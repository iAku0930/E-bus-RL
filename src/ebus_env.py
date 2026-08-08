"""E-bus 环境（现实化）：状态、动作掩码、三阶段转移、奖励。

充电简化：桩同质，动作=本段充几辆(0..n)，环境按"电量最低优先"分配空闲桩。
客流：泊松随机。详见 DESIGN.md / config.py。"""
import numpy as np
import config as C


class EBusEnv:
    def __init__(self, w1=1.0, w2=1.0, w3=1.0, seed=None, num_days=None):
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.rng = np.random.default_rng(seed)
        self.m, self.n = C.M_BUSES, C.N_CHARGERS
        self.num_days = C.NUM_DAYS if num_days is None else num_days
        self.total_steps = self.num_days * C.T_PERIODS
        self.reset()

    def reset(self):
        m = self.m
        self.bus_pos = np.zeros(m, dtype=int)        # 0=O 1=E 2=途中
        self.bus_rem = np.zeros(m, dtype=int)        # 运行剩余时段
        self.bus_last = np.zeros(m, dtype=int)
        self.bus_batt = np.full(m, C.BATTERY_FULL, dtype=float)
        self.bus_stat = np.zeros(m, dtype=int)       # 0空闲 1运行 2充电
        self.bus_chg_rem = np.zeros(m, dtype=int)    # 充电剩余时段
        for i in range(m):
            if i < C.M_O:
                self.bus_pos[i] = 0; self.bus_last[i] = 1
            else:
                self.bus_pos[i] = 1; self.bus_last[i] = 0
        self.g = 0                                   # 全局步数（跨天）
        self.t = 0                                   # 当天内时段
        self.day = 0
        self.waiting = float(self._draw_arrival(0))
        self.wait_time = 0.0
        self._carried = 0.0
        self._cost = 0.0
        self.sp_carried = 0.0
        self.ce_cost = 0.0
        self.n_charge = 0
        self.n_dispatch = 0
        self.n_lost = 0.0
        self.wait_cost_sum = 0.0
        return self.get_state()

    def _draw_arrival(self, t):
        """泊松随机到达（t 为当天内时段）。"""
        if t >= C.T_PERIODS:
            return 0.0
        return float(self.rng.poisson(C.ARRIVALS_MEAN[t]))

    def _free_chargers(self):
        return self.n - int(np.sum(self.bus_stat == 2))

    # ---------------- 状态 ----------------
    def get_state(self):
        s = np.zeros(C.STATE_DIM, dtype=np.float32)
        idx = 0
        for i in range(self.m):
            s[idx + self.bus_pos[i]] = 1; idx += 3
            s[idx] = self.bus_rem[i] / C.RUN_TIME_PEAK; idx += 1
            s[idx] = self.bus_batt[i]; idx += 1
            s[idx] = self.bus_last[i]; idx += 1
            s[idx + self.bus_stat[i]] = 1; idx += 3
            s[idx] = self.bus_chg_rem[i] / C.CHARGE_TIME; idx += 1
        s[idx] = C.price_of(self.t) / C.PRICE_MAX; idx += 1
        s[idx] = min(self.waiting, C.WAIT_THRESH) / C.WAIT_THRESH; idx += 1
        s[idx] = min(self.wait_time, 20.0) / 20.0; idx += 1
        s[idx] = self._free_chargers() / self.n; idx += 1
        return s

    # ---------------- 动作掩码 ----------------
    def _end_mask(self, side):
        m = np.zeros(self.m + 1, dtype=np.float32)
        m[0] = 1.0
        if self.waiting > 0:
            want = 0 if side == 'O' else 1
            for i in range(self.m):
                if (self.bus_stat[i] == 0 and
                        self.bus_batt[i] >= C.TRIP_CONSUMPTION and
                        self.bus_pos[i] == want):
                    m[i + 1] = 1.0
        return m

    def dispatch_mask_O(self):
        return self._end_mask('O')

    def dispatch_mask_E(self):
        return self._end_mask('E')

    def charge_mask(self):
        """合法充车辆数 0..min(空闲桩, 可充车数)。"""
        mask = np.zeros(C.CHARGE_DIM, dtype=np.float32)
        free = self._free_chargers()
        chargeable = int(np.sum((self.bus_stat == 0) &
                                (self.bus_batt < C.BATTERY_FULL - 1e-3) &
                                ((self.bus_pos == 0) | (self.bus_pos == 1))))
        maxk = min(free, chargeable, self.n)
        mask[:maxk + 1] = 1.0
        return mask

    # ---------------- 三阶段转移 ----------------
    def _dispatch_one(self, i):
        dep = self.bus_pos[i]
        self.bus_last[i] = dep
        self.bus_pos[i] = 2
        self.bus_rem[i] = C.RUN_TIME_PEAK if C.is_peak(self.t) else C.RUN_TIME_NORMAL  # 拥堵动态耗时
        self.bus_stat[i] = 1
        self.bus_batt[i] -= C.TRIP_CONSUMPTION
        carried = min(self.waiting, C.CAPACITY)
        self.waiting -= carried
        self.wait_time = 0.0
        self.n_dispatch += 1
        return carried

    def apply_dispatch(self, a_O, a_E):
        self._carried = 0.0
        if self.waiting > 0:
            if a_O >= 1:
                i = a_O - 1
                if (self.bus_stat[i] == 0 and
                        self.bus_batt[i] >= C.TRIP_CONSUMPTION and
                        self.bus_pos[i] == 0):
                    self._carried += self._dispatch_one(i)
            if a_E >= 1 and self.waiting > 0:
                i = a_E - 1
                if (self.bus_stat[i] == 0 and
                        self.bus_batt[i] >= C.TRIP_CONSUMPTION and
                        self.bus_pos[i] == 1):
                    self._carried += self._dispatch_one(i)
        return self._carried

    def apply_charge(self, a_c):
        """a_c = 本段要充的辆数；按电量最低优先分配空闲桩。返回 cost。"""
        self._cost = 0.0
        free = self._free_chargers()
        # 候选：非满电、空闲、在站
        cand = [i for i in range(self.m)
                if self.bus_stat[i] == 0
                and self.bus_batt[i] < C.BATTERY_FULL - 1e-3
                and self.bus_pos[i] in (0, 1)]
        cand.sort(key=lambda i: self.bus_batt[i])           # 电量最低优先
        k = min(int(a_c), free, len(cand))
        for i in cand[:k]:
            self.bus_stat[i] = 2
            self.bus_chg_rem[i] = C.CHARGE_TIME
            self._cost += C.price_of(self.t) * C.CHARGE_COST_K
            self.n_charge += 1
        return self._cost

    def apply_dispatch_heur(self, a_dO, a_dE):
        """启发式发车：电量最高的空闲车优先（联合智能体用，体现"选车交给规则"）。"""
        self._carried = 0.0
        for side, a_d in (('O', a_dO), ('E', a_dE)):
            if a_d >= 1 and self.waiting > 0:
                want = 0 if side == 'O' else 1
                cand = [i for i in range(self.m)
                        if self.bus_stat[i] == 0
                        and self.bus_batt[i] >= C.TRIP_CONSUMPTION
                        and self.bus_pos[i] == want]
                if cand:
                    i = max(cand, key=lambda x: self.bus_batt[x])   # 电量最高优先
                    self._carried += self._dispatch_one(i)
        return self._carried

    def apply_heuristic(self, a_dO, a_dE, a_c):
        """源论文方法：一次性启发式发车+充电（单 RL 同状态决策后调用）。"""
        self.apply_dispatch_heur(a_dO, a_dE)
        self.apply_charge(a_c)
        return self._carried, self._cost

    def advance(self):
        # 运行车到达
        for i in range(self.m):
            if self.bus_stat[i] == 1:
                self.bus_rem[i] -= 1
                if self.bus_rem[i] <= 0:
                    self.bus_stat[i] = 0
                    self.bus_pos[i] = 1 - self.bus_last[i]
                    self.bus_rem[i] = 0
            elif self.bus_stat[i] == 2:                     # 充电中
                self.bus_chg_rem[i] -= 1
                if self.bus_chg_rem[i] <= 0:
                    self.bus_stat[i] = 0
                    self.bus_batt[i] = C.BATTERY_FULL
                    self.bus_chg_rem[i] = 0
        # 时间步进 + 客流（支持多天：电量跨天继承）
        self.g += 1
        new_t = self.g % C.T_PERIODS
        new_day = self.g // C.T_PERIODS
        if new_day > self.day:                       # 跨入新一天
            self.day = new_day
            self.t = 0
            self.waiting = self._draw_arrival(0)     # 新一天首批到达
            self.wait_time = 0.0
            # 车电量/状态跨天保持（连续运营的现实）
        else:
            self.t = new_t
            if self.g < self.total_steps:
                self.waiting += self._draw_arrival(new_t)
            if self.waiting > 0:
                self.wait_time += 1.0
        # 乘客流失：等待超过忠诚线的部分按比例流失（现实：等太久转其他交通）
        lost = 0.0
        if self.waiting > C.LOYALTY_THRESH:
            lost = (self.waiting - C.LOYALTY_THRESH) * C.LOSS_RATE
            lost = min(lost, self.waiting)
            self.waiting -= lost
            self.n_lost += lost
        # 奖励（元）
        reward = (self.w1 * C.P_TICKET * self._carried
                  - self.w2 * C.C_WAIT * min(self.waiting, 100.0)
                  - self.w3 * self._cost
                  - C.LOST_PENALTY * lost)
        if self.waiting > C.WAIT_THRESH:
            reward -= C.WAIT_PENALTY_EXTRA
        self.sp_carried += self._carried
        self.ce_cost += self._cost
        self.wait_cost_sum += C.C_WAIT * self.waiting
        done = self.g >= self.total_steps
        if done:
            # terminal：末剩余电量价值计入总收益（剩余电量是资产）
            reward += float(self.bus_batt.sum()) * C.SOC_VALUE_PER_UNIT
        self._carried = 0.0
        self._cost = 0.0
        return float(reward), done

    def step(self, a_O, a_E, a_c):
        self.apply_dispatch(a_O, a_E)
        self.apply_charge(a_c)
        reward, done = self.advance()
        return self.get_state(), reward, done
