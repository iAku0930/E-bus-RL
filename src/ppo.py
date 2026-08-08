"""PPO（带动作掩码）+ 多头 PPO（联合/单智能体封装）。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import config as C


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _masked(logits, mask):
    return logits + (mask - 1) * 1e9


class Memory:
    """单头经验。"""
    def __init__(self):
        self.clear()

    def push(self, state, mask, action, logp, value, reward, done):
        self.states.append(state); self.masks.append(mask)
        self.actions.append(action); self.logps.append(logp)
        self.vals.append(value); self.rewards.append(reward); self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states, self.masks, self.actions = [], [], []
        self.logps, self.vals, self.rewards, self.dones = [], [], [], []


class PPO:
    """单头 PPO（用于联合中的充电网络）。"""

    def __init__(self, state_dim, action_dim, hidden, lr_a, lr_c,
                 gamma, lmbda, epochs, eps, device):
        self.actor = PolicyNet(state_dim, action_dim, hidden).to(device)
        self.critic = ValueNet(state_dim, hidden).to(device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma, self.lmbda, self.epochs, self.eps = gamma, lmbda, epochs, eps
        self.device = device

    @torch.no_grad()
    def select(self, state, mask):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = Categorical(logits=_masked(self.actor(s), m))
        a = dist.sample()
        return a.item(), dist.log_prob(a).item(), self.critic(s).squeeze(1).item()

    def update(self, mem, batch_size):
        if len(mem) == 0:
            return 0.0
        d = self.device
        S = torch.as_tensor(np.array(mem.states), dtype=torch.float32, device=d)
        M = torch.as_tensor(np.array(mem.masks), dtype=torch.float32, device=d)
        A = torch.as_tensor(mem.actions, dtype=torch.long, device=d)
        OL = torch.as_tensor(mem.logps, dtype=torch.float32, device=d)
        V = torch.as_tensor(mem.vals, dtype=torch.float32, device=d)
        R = torch.as_tensor(mem.rewards, dtype=torch.float32, device=d)
        D = torch.as_tensor(mem.dones, dtype=torch.float32, device=d)
        T = len(mem)
        adv = torch.zeros(T, device=d); gae = 0.0
        for t in reversed(range(T)):
            nxt = V[t + 1] if t + 1 < T else 0.0
            delta = R[t] + self.gamma * nxt * (1.0 - D[t]) - V[t]
            gae = delta + self.gamma * self.lmbda * (1.0 - D[t]) * gae
            adv[t] = gae
        returns = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for b in range(0, T, batch_size):
                bi = idx[b:b + batch_size]
                dist = Categorical(logits=_masked(self.actor(S[bi]), M[bi]))
                logp = dist.log_prob(A[bi])
                v = self.critic(S[bi]).squeeze(1)
                ratio = torch.exp(logp - OL[bi])
                surr1 = ratio * adv[bi]
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv[bi]
                loss = (-torch.min(surr1, surr2).mean()
                        + 0.5 * ((returns[bi] - v) ** 2).mean()
                        - C.ENTROPY_COEF * dist.entropy().mean())
                self.opt_a.zero_grad(); self.opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_a.step(); self.opt_c.step()
        return float(loss.item())


class MultiHeadPPO:
    """多头 PPO：共享 critic，若干个独立动作头。

    head_dims: 各头动作维度列表。
    用于：联合的发车(2头：起点/终点)、单智能体(发车2头 + 充电1头)。
    """
    def __init__(self, state_dim, head_dims, hidden, lr_a, lr_c,
                 gamma, lmbda, epochs, eps, device):
        self.H = len(head_dims)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()).to(device)
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in head_dims]).to(device)
        self.critic = ValueNet(state_dim, hidden).to(device)
        self.opt = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.heads.parameters()) +
            list(self.critic.parameters()), lr=lr_a)
        self.gamma, self.lmbda, self.epochs, self.eps = gamma, lmbda, epochs, eps
        self.device = device

    def _logits(self, s):
        h = self.trunk(s)
        return [head(h) for head in self.heads]

    @torch.no_grad()
    def select(self, state, masks):
        """返回 (actions:list, logps:list, value)。"""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self._logits(s)
        actions, logps = [], []
        for h, lg, mk in zip(range(self.H), logits, masks):
            m = torch.as_tensor(mk, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = Categorical(logits=_masked(lg, m))
            a = dist.sample()
            actions.append(a.item()); logps.append(dist.log_prob(a).item())
        v = self.critic(s).squeeze(1).item()
        return actions, logps, v

    def update(self, mem, batch_size):
        if len(mem) == 0:
            return 0.0
        d = self.device
        S = torch.as_tensor(np.array(mem.states), dtype=torch.float32, device=d)
        M = [torch.as_tensor(np.array(mk), dtype=torch.float32, device=d) for mk in mem.masks]
        A = [torch.as_tensor(mem.actions_h[h], dtype=torch.long, device=d) for h in range(self.H)]
        OL = [torch.as_tensor(mem.logps_h[h], dtype=torch.float32, device=d) for h in range(self.H)]
        V = torch.as_tensor(mem.vals, dtype=torch.float32, device=d)
        R = torch.as_tensor(mem.rewards, dtype=torch.float32, device=d)
        D = torch.as_tensor(mem.dones, dtype=torch.float32, device=d)
        T = len(mem)
        adv = torch.zeros(T, device=d); gae = 0.0
        for t in reversed(range(T)):
            nxt = V[t + 1] if t + 1 < T else 0.0
            delta = R[t] + self.gamma * nxt * (1.0 - D[t]) - V[t]
            gae = delta + self.gamma * self.lmbda * (1.0 - D[t]) * gae
            adv[t] = gae
        returns = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for b in range(0, T, batch_size):
                bi = idx[b:b + batch_size]
                logits = self._logits(S[bi])
                logps, ents = [], []
                for h in range(self.H):
                    dist = Categorical(logits=_masked(logits[h], M[h][bi]))
                    logps.append(dist.log_prob(A[h][bi]))
                    ents.append(dist.entropy().mean())
                ratio = torch.exp(sum(logps) / self.H - sum(OL[h][bi] for h in range(self.H)) / self.H)
                surr1 = ratio * adv[bi]
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv[bi]
                v = self.critic(S[bi]).squeeze(1)
                loss = (-torch.min(surr1, surr2).mean()
                        + 0.5 * ((returns[bi] - v) ** 2).mean()
                        - C.ENTROPY_COEF * (sum(ents) / self.H))
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trunk.parameters(), 0.5)
                self.opt.step()
        return float(loss.item())


class MultiHeadMemory:
    """多头经验：masks/actions/logps 按 head 分组存。"""
    def __init__(self, n_heads):
        self.H = n_heads
        self.clear()

    def push(self, state, masks, actions, logps, value, reward, done):
        self.states.append(state)
        for h in range(self.H):
            self.masks[h].append(masks[h])
            self.actions_h[h].append(actions[h])
            self.logps_h[h].append(logps[h])
        self.vals.append(value); self.rewards.append(reward); self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states = []
        self.masks = [[] for _ in range(self.H)]
        self.actions_h = [[] for _ in range(self.H)]
        self.logps_h = [[] for _ in range(self.H)]
        self.vals, self.rewards, self.dones = [], [], []


class JointCentralized:
    """集中式 critic：发车 actor(多头) + 充电 actor(单头) 共享一个全局价值网络。
    两个 actor 解耦 + 耦合（充电看发车后状态），但训练时基于同一全局 V(s) 算优势，
    从而协调一致——这是让"联合智能体"真正发挥耦合价值的关键。"""
    def __init__(self, state_dim, d_dims, c_dim, hidden, lr_a, lr_c,
                 gamma, lmbda, epochs, eps, device):
        self.H = len(d_dims)
        self.trunk_d = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()).to(device)
        self.heads_d = nn.ModuleList([nn.Linear(hidden, d) for d in d_dims]).to(device)
        self.actor_c = PolicyNet(state_dim, c_dim, hidden).to(device)
        self.critic = ValueNet(state_dim, hidden).to(device)           # 共享全局 critic
        params_a = (list(self.trunk_d.parameters()) + list(self.heads_d.parameters())
                    + list(self.actor_c.parameters()))
        self.opt_a = torch.optim.Adam(params_a, lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma, self.lmbda, self.epochs, self.eps = gamma, lmbda, epochs, eps
        self.device = device

    @torch.no_grad()
    def select_d(self, state, masks):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.trunk_d(s)
        actions, logps = [], []
        for head, mk in zip(self.heads_d, masks):
            m = torch.as_tensor(mk, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = Categorical(logits=_masked(head(h), m))
            a = dist.sample(); actions.append(a.item()); logps.append(dist.log_prob(a).item())
        return actions, logps, self.critic(s).squeeze(1).item()

    @torch.no_grad()
    def select_c(self, state, mask):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = Categorical(logits=_masked(self.actor_c(s), m))
        a = dist.sample()
        return a.item(), dist.log_prob(a).item(), self.critic(s).squeeze(1).item()

    def _gae(self, vals, rewards, dones):
        V = torch.as_tensor(vals, dtype=torch.float32, device=self.device)
        R = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        D = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        T = len(R); adv = torch.zeros(T, device=self.device); gae = 0.0
        for t in reversed(range(T)):
            nxt = V[t + 1] if t + 1 < T else 0.0
            delta = R[t] + self.gamma * nxt * (1.0 - D[t]) - V[t]
            gae = delta + self.gamma * self.lmbda * (1.0 - D[t]) * gae
            adv[t] = gae
        return adv, adv + V

    def update(self, mem_d, mem_c, bs):
        if len(mem_d) == 0:
            return 0.0
        d = self.device
        S_d = torch.as_tensor(np.array(mem_d.states), dtype=torch.float32, device=d)
        M_d = [torch.as_tensor(np.array(mk), dtype=torch.float32, device=d) for mk in mem_d.masks]
        A_d = [torch.as_tensor(mem_d.actions_h[h], dtype=torch.long, device=d) for h in range(self.H)]
        OL_d = [torch.as_tensor(mem_d.logps_h[h], dtype=torch.float32, device=d) for h in range(self.H)]
        adv_d, ret_d = self._gae(mem_d.vals, mem_d.rewards, mem_d.dones)
        adv_d = (adv_d - adv_d.mean()) / (adv_d.std() + 1e-8)

        S_c = torch.as_tensor(np.array(mem_c.states), dtype=torch.float32, device=d)
        M_c = torch.as_tensor(np.array(mem_c.masks), dtype=torch.float32, device=d)
        A_c = torch.as_tensor(mem_c.actions, dtype=torch.long, device=d)
        OL_c = torch.as_tensor(mem_c.logps, dtype=torch.float32, device=d)
        adv_c, ret_c = self._gae(mem_c.vals, mem_c.rewards, mem_c.dones)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        T = len(mem_d); idx = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for b in range(0, T, bs):
                bi = idx[b:b + bs]
                # actor_d
                h = self.trunk_d(S_d[bi]); logits = [head(h) for head in self.heads_d]
                lpd = []; entd = 0.0
                for hi in range(self.H):
                    dist = Categorical(logits=_masked(logits[hi], M_d[hi][bi]))
                    lpd.append(dist.log_prob(A_d[hi][bi])); entd += dist.entropy().mean()
                ratio_d = torch.exp(sum(lpd) / self.H - sum(OL_d[hi][bi] for hi in range(self.H)) / self.H)
                a_loss_d = -torch.min(ratio_d * adv_d[bi],
                                      torch.clamp(ratio_d, 1 - self.eps, 1 + self.eps) * adv_d[bi]).mean()
                # actor_c
                distc = Categorical(logits=_masked(self.actor_c(S_c[bi]), M_c[bi]))
                lpc = distc.log_prob(A_c[bi])
                ratio_c = torch.exp(lpc - OL_c[bi])
                a_loss_c = -torch.min(ratio_c * adv_c[bi],
                                      torch.clamp(ratio_c, 1 - self.eps, 1 + self.eps) * adv_c[bi]).mean()
                # 集中式 critic：同时拟合 s0 和 s1 的全局价值
                vd = self.critic(S_d[bi]).squeeze(1); vc = self.critic(S_c[bi]).squeeze(1)
                c_loss = ((ret_d[bi] - vd) ** 2).mean() + ((ret_c[bi] - vc) ** 2).mean()
                ent = 0.5 * entd / self.H + 0.5 * distc.entropy().mean()
                loss = a_loss_d + a_loss_c + 0.5 * c_loss - C.ENTROPY_COEF * ent
                self.opt_a.zero_grad(); self.opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trunk_d.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.actor_c.parameters(), 0.5)
                self.opt_a.step(); self.opt_c.step()
        return float(loss.item())
