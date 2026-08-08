"""全天策略可视化（dual-stage）。训练 → 取第 2 天数据 → 4 子图：车辆甘特/电量/客流电价/发车充电时机。"""
import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as C
import train

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 120
rcParams['savefig.dpi'] = 150
rcParams['font.size'] = 12

RES = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RES, exist_ok=True)
EP = 600


def smooth(a, w=31):
    w = min(w, len(a))
    return np.convolve(a, np.ones(w) / w, mode='valid') if w > 1 else a


def run_day(agent):
    """跑一个多天 episode，返回【第2天】(索引 48..95)的逐时段记录。"""
    env = train.EBusEnv(1, 1, 1, seed=7)
    s = env.reset(); done = False
    rec = dict(pos=[], stat=[], batt=[], waiting=[], price=[], carried=[], chg=[], nD=[], nC=[])
    while not done:
        if env.day == 1:                               # 只记第2天（避开初始满电）
            rec['pos'].append(env.bus_pos.copy())
            rec['stat'].append(env.bus_stat.copy())
            rec['batt'].append(env.bus_batt.copy())
            rec['waiting'].append(env.waiting)
            rec['price'].append(C.price_of(env.t))
            rec['carried'].append(env.sp_carried)
            nd0, nc0 = env.n_dispatch, env.n_charge
        dO = env.dispatch_mask_O(); dE = env.dispatch_mask_E()
        m_dO = np.array([1.0, 1.0 if (dO[1:].any() and env.waiting > 0) else 0.0])
        m_dE = np.array([1.0, 1.0 if (dE[1:].any() and env.waiting > 0) else 0.0])
        mc = env.charge_mask()
        acts, _, _ = agent.select(s, [m_dO, m_dE, mc])
        env.apply_heuristic(*acts)
        if env.day == 1:
            rec['nD'].append(env.n_dispatch - nd0)
            rec['nC'].append(env.n_charge - nc0)
            rec['chg'].append(env.n_charge - nc0)
        r, done = env.advance(); s = env.get_state()
    return rec, env


def main():
    t0 = time.time()
    print(f'训练 dual-stage {EP}ep ...')
    log, agent = train.train_source(1, 1, 1, num_episodes=EP, seed=0, verbose=True)

    # ---- 训练 reward 曲线 ----
    R = np.array(log['return']) / C.NUM_DAYS           # 日均
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(R, color='#bdd7e7', lw=0.8, alpha=0.7, label='每 episode 日均 reward')
    ax.plot(smooth(R), color='#41ab5d', lw=2.6, label='滑动平均')
    ax.set_xlabel('Episode', fontsize=13); ax.set_ylabel('日均 Return (元)', fontsize=13)
    ax.set_title('源方法 训练收敛曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'reward_curve.png')); plt.close(fig)

    # ---- 全天策略（第2天）----
    rec, env = run_day(agent)
    T = len(rec['pos']); tt = np.arange(T)
    STAT = np.array(rec['stat']); BATT = np.array(rec['batt'])

    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    # 子图1：车辆状态甘特
    cmap = plt.matplotlib.colors.ListedColormap(['#eeeeee', '#4C72B0', '#DD8452'])
    axes[0].imshow(STAT.T, aspect='auto', cmap=cmap, vmin=0, vmax=2,
                   extent=[-0.5, T - 0.5, -0.5, C.M_BUSES - 0.5], interpolation='nearest')
    axes[0].set_ylabel('车辆', fontsize=12); axes[0].set_yticks(range(C.M_BUSES))
    axes[0].set_title('车辆状态调度（灰=空闲在站  蓝=运行中  橙=充电中）', fontsize=13, fontweight='bold')
    axes[0].legend(handles=[Patch(color='#eeeeee', label='空闲'), Patch(color='#4C72B0', label='运行'),
                            Patch(color='#DD8452', label='充电')], loc='upper right', fontsize=10, ncol=3)
    # 子图2：各车电量轨迹
    for i in range(C.M_BUSES):
        axes[1].plot(tt, BATT[:, i], marker='o', ms=3, lw=1.3, label=f'车{i}')
    axes[1].set_ylabel('电量', fontsize=12); axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('各车电量轨迹', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=7, ncol=6, loc='lower right'); axes[1].grid(alpha=0.3)
    # 子图3：等待人数 + 分时电价（双轴）
    ax3 = axes[2]
    ax3.bar(tt, rec['waiting'], width=1.0, color='#9ecae1', label='等待人数')
    ax3.set_ylabel('等待乘客数', fontsize=12, color='#3182bd')
    ax3b = ax3.twinx()
    ax3b.plot(tt, rec['price'], color='#e6550d', lw=2.4, marker='s', ms=4, label='分时电价')
    ax3b.set_ylabel('电价', fontsize=12, color='#e6550d'); ax3b.set_ylim(0, 4)
    ax3.set_title('乘客等待 & 分时电价', fontsize=13, fontweight='bold'); ax3.grid(alpha=0.3)
    # 子图4：发车/充电计数 + 充电时刻标在电价上
    ax4 = axes[3]
    ax4.bar(tt, rec['nD'], width=1.0, color='#4C72B0', alpha=0.45, label='本段发车数')
    ax4b = ax4.twinx()
    ax4b.plot(tt, rec['price'], color='#e6550d', lw=1.6, alpha=0.5)
    ax4b.set_ylabel('电价', fontsize=12, color='#e6550d'); ax4b.set_ylim(0, 4)
    # 标充电时刻
    for i in range(T):
        if rec['chg'][i] > 0:
            p = rec['price'][i]
            c = '#2ca02c' if p == 1 else ('#ff7f0e' if p == 2 else '#d62728')
            ax4.axvline(i, color=c, alpha=0.6, lw=2.5)
    ax4.set_ylabel('发车数', fontsize=12, color='#4C72B0'); ax4.set_ylim(0, 3)
    ax4.set_xlabel('时段（第2天，0=06:00，47=次日05:30）', fontsize=12)
    ax4.set_title('发车节奏 & 充电时机（竖线=充电：绿=低价段 红=高价段）', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10); ax4.grid(alpha=0.3)
    xticks = np.arange(0, T, 4)
    axes[3].set_xticks(xticks)
    axes[3].set_xticklabels([f'{int((6 + x*0.5)) % 24:02d}:00' for x in xticks])
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'day_strategy.png')); plt.close(fig)

    print(f'\n第2天: 累计接送={rec["carried"][-1]-rec["carried"][0]:.0f}  末电量均值={np.mean(BATT[-1]):.2f}')
    print(f'用时 {time.time()-t0:.0f}s. 图: reward_curve.png, day_strategy.png')


if __name__ == '__main__':
    main()
