"""训练循环。三种方法：
- train_joint：双 PPO 联合（发车 actor + 充电 actor，集中式 critic，启发式选车）
- train_source：dual-stage（单多头 PPO，启发式选车）
- train_single：单多头 PPO 选具体车（早期 baseline）
耦合：每段先发车，充电在发车之后的状态上决策。
"""
import numpy as np
import torch
from tqdm import tqdm

import config as C
from ebus_env import EBusEnv
from ppo import PPO, MultiHeadPPO, JointCentralized, Memory, MultiHeadMemory


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _new_log():
    return {'return': [], 'sp': [], 'ce': [], 'wait': [], 'n_charge': [], 'n_dispatch': []}


# ---------------- 联合智能体（两 actor 耦合 + 启发式选车 + 集中式 critic）----------------
def train_joint(w1, w2, w3, num_episodes=C.NUM_EPISODES, seed=C.SEED,
                verbose=True, device=None):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EBusEnv(w1, w2, w3, seed=seed)
    # 集中式：发车 actor(2头) + 充电 actor(1头) 共享全局 critic
    agent = JointCentralized(C.STATE_DIM, [2, 2], C.CHARGE_DIM,
                             C.HIDDEN, C.ACTOR_LR, C.CRITIC_LR, C.GAMMA, C.LMBDA,
                             C.PPO_EPOCHS, C.EPS_CLIP, device)
    mem_d = MultiHeadMemory(2)
    mem_c = Memory()

    log = _new_log()
    it = range(num_episodes)
    if verbose:
        it = tqdm(it, desc=f'Joint w=({w1},{w2},{w3})', ncols=92)
    for ep in it:
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            dO = env.dispatch_mask_O(); dE = env.dispatch_mask_E()
            m_dO = np.array([1.0, 1.0 if (dO[1:].any() and env.waiting > 0) else 0.0])
            m_dE = np.array([1.0, 1.0 if (dE[1:].any() and env.waiting > 0) else 0.0])
            [a_dO, a_dE], [lp_dO, lp_dE], v_d = agent.select_d(s, [m_dO, m_dE])
            env.apply_dispatch_heur(a_dO, a_dE)                     # 启发式发车
            s1 = env.get_state()                                    # 耦合点：充电看发车后状态
            mc = env.charge_mask()
            a_c, lp_c, v_c = agent.select_c(s1, mc)
            env.apply_charge(a_c)                                   # 启发式充电
            r, done = env.advance()
            s_next = env.get_state()
            mem_d.push(s, [m_dO, m_dE], [a_dO, a_dE], [lp_dO, lp_dE], v_d, r, done)
            mem_c.push(s1, mc, a_c, lp_c, v_c, r, done)
            s = s_next
            ep_ret += r
        log['return'].append(ep_ret)
        log['sp'].append(env.sp_carried)
        log['ce'].append(env.ce_cost)
        log['wait'].append(env.wait_cost_sum)
        log['n_charge'].append(env.n_charge)
        log['n_dispatch'].append(env.n_dispatch)
        if ep % C.UPDATE_INTERVAL == 0 and len(mem_d) > 0:
            agent.update(mem_d, mem_c, C.BATCH_SIZE)
            mem_d.clear(); mem_c.clear()
        if verbose:
            it.set_postfix(R=f'{ep_ret:.1f}', SP=f'{env.sp_carried:.0f}',
                           CE=f'{env.ce_cost:.1f}', nD=f'{env.n_dispatch}',
                           nC=f'{env.n_charge}')
    return log, agent


# ---------------- 单智能体（3头）----------------
def train_single(w1, w2, w3, num_episodes=C.NUM_EPISODES, seed=C.SEED,
                 verbose=True, device=None):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EBusEnv(w1, w2, w3, seed=seed)
    agent = MultiHeadPPO(C.STATE_DIM,
                         [C.DISPATCH_HEAD_DIM, C.DISPATCH_HEAD_DIM, C.CHARGE_DIM],
                         C.HIDDEN, C.ACTOR_LR, C.CRITIC_LR, C.GAMMA, C.LMBDA,
                         C.PPO_EPOCHS, C.EPS_CLIP, device)
    mem = MultiHeadMemory(3)
    log = _new_log()
    it = range(num_episodes)
    if verbose:
        it = tqdm(it, desc=f'Single w=({w1},{w2},{w3})', ncols=92)
    for ep in it:
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            mO = env.dispatch_mask_O(); mE = env.dispatch_mask_E()
            mc = env.charge_mask()
            acts, lps, v = agent.select(s, [mO, mE, mc])           # 同状态同时决策
            a_O, a_E, a_c = acts
            env.apply_dispatch(a_O, a_E)
            env.apply_charge(a_c)
            r, done = env.advance()
            s_next = env.get_state()
            mem.push(s, [mO, mE, mc], acts, lps, v, r, done)
            s = s_next
            ep_ret += r
        log['return'].append(ep_ret)
        log['sp'].append(env.sp_carried)
        log['ce'].append(env.ce_cost)
        log['wait'].append(env.wait_cost_sum)
        log['n_charge'].append(env.n_charge)
        log['n_dispatch'].append(env.n_dispatch)
        if ep % C.UPDATE_INTERVAL == 0 and len(mem) > 0:
            agent.update(mem, C.BATCH_SIZE)
            mem.clear()
        if verbose:
            it.set_postfix(R=f'{ep_ret:.1f}', SP=f'{env.sp_carried:.0f}',
                           CE=f'{env.ce_cost:.1f}', nD=f'{env.n_dispatch}',
                           nC=f'{env.n_charge}')
    return log, agent


# ---------------- 源论文方法（dual-stage：RL 决定数量 + 启发式选车）----------------
def train_source(w1, w2, w3, num_episodes=C.NUM_EPISODES, seed=C.SEED,
                 verbose=True, device=None):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EBusEnv(w1, w2, w3, seed=seed)
    # 3 头：O端发不发(0/1)、E端发不发(0/1)、充几辆(0..n)；具体车由启发式选
    agent = MultiHeadPPO(C.STATE_DIM, [2, 2, C.CHARGE_DIM],
                         C.HIDDEN, C.ACTOR_LR, C.CRITIC_LR, C.GAMMA, C.LMBDA,
                         C.PPO_EPOCHS, C.EPS_CLIP, device)
    mem = MultiHeadMemory(3)
    log = _new_log()
    it = range(num_episodes)
    if verbose:
        it = tqdm(it, desc=f'Source w=({w1},{w2},{w3})', ncols=92)
    for ep in it:
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            dO = env.dispatch_mask_O(); dE = env.dispatch_mask_E()
            # RL 只决定"发不发/充几辆"，合法性由掩码给出（启发式兜底具体车）
            m_dO = np.array([1.0, 1.0 if (dO[1:].any() and env.waiting > 0) else 0.0])
            m_dE = np.array([1.0, 1.0 if (dE[1:].any() and env.waiting > 0) else 0.0])
            mc = env.charge_mask()
            acts, lps, v = agent.select(s, [m_dO, m_dE, mc])
            a_dO, a_dE, a_c = acts
            env.apply_heuristic(a_dO, a_dE, a_c)                   # 启发式执行
            r, done = env.advance()
            s_next = env.get_state()
            mem.push(s, [m_dO, m_dE, mc], acts, lps, v, r, done)
            s = s_next
            ep_ret += r
        log['return'].append(ep_ret)
        log['sp'].append(env.sp_carried)
        log['ce'].append(env.ce_cost)
        log['wait'].append(env.wait_cost_sum)
        log['n_charge'].append(env.n_charge)
        log['n_dispatch'].append(env.n_dispatch)
        if ep % C.UPDATE_INTERVAL == 0 and len(mem) > 0:
            agent.update(mem, C.BATCH_SIZE)
            mem.clear()
        if verbose:
            it.set_postfix(R=f'{ep_ret:.1f}', SP=f'{env.sp_carried:.0f}',
                           CE=f'{env.ce_cost:.1f}', nD=f'{env.n_dispatch}',
                           nC=f'{env.n_charge}')
    return log, agent


if __name__ == '__main__':
    print('== smoke test: joint, 400 ep ==')
    log, _ = train_joint(0.4, 0.4, 0.2, num_episodes=400, verbose=True)
    import numpy as np
    print('末30 ep: R=%.2f SP=%.0f CE=%.1f nD=%.1f nC=%.1f' %
          (np.mean(log['return'][-30:]), np.mean(log['sp'][-30:]),
           np.mean(log['ce'][-30:]), np.mean(log['n_dispatch'][-30:]),
           np.mean(log['n_charge'][-30:])))
