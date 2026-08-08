"""全局配置：实例规模、客流/电价、reward、PPO 超参。"""
import numpy as np

# ---------------- 实例规模 ----------------
M_BUSES = 12          # 公交车数（起点 6、终点 6）
M_O = 6
N_CHARGERS = 4        # 充电桩数（起点 2、终点 2）—— 桩/车比降低，充电排队更紧
N_O = 2
T_PERIODS = 48        # 一天时段数（每段 30min，6:00 起）
NUM_DAYS = 5          # 一个 episode 的连续天数（电量跨天继承，让充电有长期意义）
RUN_TIME_NORMAL = 3   # 正常一趟 3 段（1.5h）
RUN_TIME_PEAK = 4     # 高峰堵车一趟 4 段（2h）
CHARGE_TIME = 4       # 一次满充时段数（=2h，更长→离岗久→规划价值大）
CAPACITY = 25         # 单车容量（中型公交）
BATTERY_FULL = 1.0
TRIP_CONSUMPTION = 0.25    # 单趟耗电（满电约 4 趟）—— 之前验证有效，电池不过紧

M_E = M_BUSES - M_O
N_E = N_CHARGERS - N_O

# ---------------- 分时电价（逐小时 24 维，峰谷比 ~3:1） ----------------
E_PRICE = np.array([1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 2,
                    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1], dtype=float)
PRICE_MAX = float(E_PRICE.max())

# ---------------- 客流（现实日内曲线 + 泊松随机）----------------
def _realistic_arrivals():
    """48 段×30min(6:00 起)。陡峰，峰接近运力(2×CAP=50)，制造调度压力。"""
    prof = np.zeros(T_PERIODS)
    prof[0:2] = 10                                 # 早班
    prof[2:6] = [35, 45, 45, 38]                   # 早高峰 7-9（陡）
    prof[6:8] = 24
    prof[8:20] = 16                                # 平峰
    prof[20:22] = 26
    prof[22:26] = [40, 45, 42, 34]                 # 晚高峰 17-19（陡）
    prof[26:30] = 28
    prof[30:34] = 16
    prof[34:48] = 5                                # 夜班
    return prof
ARRIVALS_MEAN = _realistic_arrivals()
ARR_NOISE = 'poisson'     # 到达量 = Poisson(ARRIVALS_MEAN[t])，计数过程更现实
WAIT_THRESH = 120         # 等待人数过多阈值（额外惩罚）

# 乘客流失（现实：等太久转其他交通）—— 适中阈值，让"发车载客"成为智能体自发的正收益来源
LOYALTY_THRESH = 50       # 等待人数超过此线（≈2倍CAP）开始流失
LOSS_RATE = 0.2           # 超出部分每时段流失比例
LOST_PENALTY = 2.0        # 每流失一人损失的收益（≈一张票价）

# ---------------- reward 经济标定（各项均为"元"，比例 = 现实核校）----------------
P_TICKET = 2.0            # 接送 1 乘客收益（元/人）；一趟25人≈50元
C_WAIT = 0.5              # 乘客等待社会成本（元/(人·段)）；发车动力主要来自载客收益+流失罚
CHARGE_COST_K = 60.0      # 单次充电电费 = price × 60 元（60/120/180）；price3(180)>3.3趟票收(165)→高价充净亏
WAIT_PENALTY_EXTRA = 30.0  # waiting 超阈值额外惩罚（元）
SOC_VALUE_PER_UNIT = 0.0   # 末电量价值：不放训练 reward（避免诱导过度充电），仅作评估指标公平计入


def is_peak(t):
    """高峰时段（早7-9/晚17-19）→ 一趟耗时更长（拥堵）。"""
    h = int((6 + t * 0.5)) % 24
    return h in (7, 8, 17, 18)

# ---------------- 状态/动作维度 ----------------
# per-bus: pos_oh(3)+rem_run(1)+batt(1)+last_stop(1)+status_oh(3)+chg_rem(1) = 10
BUS_FEATURE = 10
GLOBAL_FEATURE = 4        # price + waiting + wait_time + 空闲桩数
STATE_DIM = BUS_FEATURE * M_BUSES + GLOBAL_FEATURE
DISPATCH_HEAD_DIM = M_BUSES + 1     # 发车每头：选某辆车 or 不发
CHARGE_DIM = N_CHARGERS + 1         # 充电：本段充几辆 (0..n)

# ---------------- PPO 超参 ----------------
ACTOR_LR = 1e-3
CRITIC_LR = 1e-2
HIDDEN = 128
GAMMA = 0.98
LMBDA = 0.95
PPO_EPOCHS = 10
EPS_CLIP = 0.2
BATCH_SIZE = 64
BUFFER_SIZE = 10000
ENTROPY_COEF = 0.03

# ---------------- 训练 ----------------
NUM_EPISODES = 2000
UPDATE_INTERVAL = 10
SEED = 0
W = (1.0, 1.0, 1.0)      # 基础策略：纯经济比例

WEIGHT_SETS = [
    (1.0, 1.0, 1.0),
    (1.5, 1.0, 1.0),
    (1.0, 1.5, 1.0),
    (1.0, 1.0, 1.5),
    (0.7, 1.0, 1.3),
]


def hour_of(t):
    return int((6 + t * 0.5)) % 24


def price_of(t):
    return E_PRICE[hour_of(t)]
