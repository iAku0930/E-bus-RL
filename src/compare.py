"""最终对比实验：联合智能体(集中式双PPO) vs 源方法(单多头PPO+启发式)。
多 seed，生成清晰的收敛曲线 + 指标对比 + 充电电价分布图。"""
import os, sys, time, collections, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as C
import train

# ---- 绘图全局设置：清晰、专业 ----
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 120
rcParams['savefig.dpi'] = 150
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.1

EP = 400
SEEDS = [0, 1]
RES = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RES, exist_ok=True)
COL = {'双PPO联合': '#2c7fb8', 'dual-stage': '#41ab5d'}     # 蓝/绿
SOC_VAL_UNIT = 120.0


def mv(a, w=31):
    w = min(w, len(a))
    return np.convolve(a, np.ones(w) / w, mode='valid') if w > 1 else a


def eval_policy(method, agent, n=5):
    SP, LO, CE, SOC = [], [], [], []
    cdist = collections.Counter()
    for sd in range(n):
        env = train.EBusEnv(1, 1, 1, seed=500 + sd)
        s = env.reset(); done = False
        while not done:
            if method == 'joint':
                dO = env.dispatch_mask_O(); dE = env.dispatch_mask_E()
                m_dO = np.array([1.0, 1.0 if (dO[1:].any() and env.waiting > 0) else 0.0])
                m_dE = np.array([1.0, 1.0 if (dE[1:].any() and env.waiting > 0) else 0.0])
                [a_dO, a_dE], _, _ = agent.select_d(s, [m_dO, m_dE])
                env.apply_dispatch_heur(a_dO, a_dE)
                s1 = env.get_state(); mc = env.charge_mask()
                ac, _, _ = agent.select_c(s1, mc)
                if ac >= 1:
                    cdist[C.price_of(env.t)] += 1
                env.apply_charge(ac); r, done = env.advance(); s = env.get_state()
            else:
                dO = env.dispatch_mask_O(); dE = env.dispatch_mask_E()
                m_dO = np.array([1.0, 1.0 if (dO[1:].any() and env.waiting > 0) else 0.0])
                m_dE = np.array([1.0, 1.0 if (dE[1:].any() and env.waiting > 0) else 0.0])
                mc = env.charge_mask()
                acts, _, _ = agent.select(s, [m_dO, m_dE, mc])
                a_dO, a_dE, ac = acts
                if ac >= 1:
                    cdist[C.price_of(env.t)] += 1
                env.apply_heuristic(a_dO, a_dE, ac); r, done = env.advance(); s = env.get_state()
        tot = env.sp_carried + env.n_lost
        SP.append(env.sp_carried / C.NUM_DAYS); CE.append(env.ce_cost / C.NUM_DAYS)
        SOC.append(float(env.bus_batt.sum())); LO.append(100 * env.n_lost / tot)
    tc = sum(cdist.values())
    sp, ce, lo = np.mean(SP), np.mean(CE), np.mean(LO)
    soc_val = SOC_VAL_UNIT * np.mean(SOC) / C.NUM_DAYS
    profit = sp * C.P_TICKET - ce + soc_val - lo / 100 * sp / max(0.01, 1 - lo / 100) * C.LOST_PENALTY
    return dict(SP=sp, LO=lo, CE=ce, SOC=np.mean(SOC) / C.M_BUSES, profit=profit,
                low=100 * cdist.get(1, 0) / max(1, tc), high=100 * cdist.get(3, 0) / max(1, tc))


def main():
    t0 = time.time()
    methods = [('双PPO联合(集中式)', 'joint', train.train_joint),
               ('dual-stage(单多头+启发式)', 'source', train.train_source)]
    curves, finals = {m[0]: [] for m in methods}, {m[0]: [] for m in methods}
    for name, tag, fn in methods:
        for sd in SEEDS:
            ts = time.time()
            log, agent = fn(1, 1, 1, num_episodes=EP, seed=sd, verbose=False)
            curves[name].append(np.array(log['return']))
            finals[name].append(eval_policy(tag, agent, n=5))
            m = finals[name][-1]
            print(f'{name} seed{sd}: R={np.mean(log["return"][-30:]):.0f} '
                  f'SP={m["SP"]:.0f} CE={m["CE"]:.0f} 总收益={m["profit"]:.0f}  ({time.time()-ts:.0f}s)')

    names = [m[0] for m in methods]
    short = ['双PPO联合', 'dual-stage']

    # ========= 图1：训练收敛曲线 =========
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, sh in zip(names, short):
        cs = np.array([mv(c) for c in curves[name]])
        xp = np.arange(cs.shape[1])
        ax.plot(xp, cs.mean(0), color=COL[sh], lw=2.6, label=sh)
        ax.fill_between(xp, cs.mean(0) - cs.std(0), cs.mean(0) + cs.std(0),
                        color=COL[sh], alpha=0.16)
    ax.set_xlabel('Episode (滑动平均)', fontsize=13); ax.set_ylabel('总 Return (元)', fontsize=13)
    ax.set_title(f'训练收敛对比 ({EP} ep × {len(SEEDS)} seed，均值±std)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'cmp_return.png')); plt.close(fig)

    # ========= 图2：末段指标对比（分组柱，标数值） =========
    metrics = ['SP', 'CE', 'profit', 'SOC']
    titles = ['日均服务量 SP (人)', '日均充电花费 CE (元)', '日均总收益 (元)', '末剩余电量 (比例)']
    means = {mk: [np.mean([f[mk] for f in finals[n]]) for n in names] for mk in metrics}
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w/2, [means[mk][0] for mk in metrics], w,
                color=COL['联合'], label='联合', edgecolor='white')
    b2 = ax.bar(x + w/2, [means[mk][1] for mk in metrics], w,
                color=COL['源方法'], label='源方法', edgecolor='white')
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f'{r.get_height():.0f}', (r.get_x() + r.get_width()/2, r.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(titles, fontsize=11)
    ax.set_title('两方法末段指标对比（多 seed 平均）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'cmp_metrics.png')); plt.close(fig)

    # ========= 图3：充电电价分布（堆叠柱） =========
    fig, ax = plt.subplots(figsize=(8, 5.5))
    low = [np.mean([f['low'] for f in finals[n]]) for n in names]
    high = [np.mean([f['high'] for f in finals[n]]) for n in names]
    mid = [100 - l - h for l, h in zip(low, high)]
    xi = np.arange(len(names))
    ax.bar(xi, low, 0.5, color='#2ca02c', label='低谷 price1')
    ax.bar(xi, mid, 0.5, bottom=low, color='#ff7f0e', label='平峰 price2')
    ax.bar(xi, high, 0.5, bottom=[l+m for l, m in zip(low, mid)], color='#d62728', label='高价 price3')
    for i in range(len(names)):
        ax.annotate(f'低谷{low[i]:.0f}%', (xi[i], low[i]/2), ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')
    ax.set_xticks(xi); ax.set_xticklabels(short, fontsize=12)
    ax.set_ylabel('充电次数占比 (%)', fontsize=13)
    ax.set_title('充电时机与分时电价（绿色=低谷充电，越多越好）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'cmp_charge_price.png')); plt.close(fig)

    # ========= 汇总表 =========
    print('\n===== 最终对比汇总 =====')
    print(f'{"方法":<22}{"SP":>7}{"流失%":>7}{"CE":>8}{"末电量":>7}{"总收益":>8}{"低谷充%":>8}')
    for n, sh in zip(names, short):
        f = {mk: np.mean([s[mk] for s in finals[n]]) for mk in metrics + ['LO', 'low']}
        print(f'{sh:<22}{f["SP"]:>7.0f}{f["LO"]:>7.1f}{f["CE"]:>8.0f}{f["SOC"]:>7.2f}'
              f'{f["profit"]:>8.0f}{f["low"]:>8.0f}')
    print(f'\n用时 {time.time()-t0:.0f}s. 图: cmp_return.png, cmp_metrics.png, cmp_charge_price.png')


if __name__ == '__main__':
    main()
