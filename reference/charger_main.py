import numpy as np
import torch
from matplotlib import pyplot as plt
import openpyxl
import hucl_PPO
import charge_env
import rl_utils

actor_lr = 1e-3
critic_lr = 1e-2
state_dim = 8
action_dim = 3
hidden_dim = 64
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
seed = 1 # 随机数种子
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
buffer_size = 10000
num_episodes = 1000
batch_size = 64
M = 5
env = charge_env.env(M)

def train(env, agent, num_episodes):
    r_list = []
    for i_episode in range(num_episodes):
        state = env.reset()
        r_day = 0
        alist = []
        for time in range(72):
            a, probs, value = agent.take_action(state)
            state, r, a = env.step(state,a)
            memory.push(state, a, probs, value, r)
            alist.append(a)
            r_day += r
        r_list.append(r_day)
        if i_episode % 10 == 0:
            agent.update()
            print('episode: ', i_episode, 'reward: ', r_day)
    return r_list , alist

memory = hucl_PPO.PPOMemory(buffer_size)
agent = hucl_PPO.PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, memory)

r_list ,alist = train(env, agent, num_episodes)

episodes_list = list(range(len(r_list)))
plt.plot(episodes_list, r_list, label = 'R', color = 'b')
plt.legend(loc=0)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('reward')
plt.show()


T_list = list(range(len(alist)))
plt.plot(T_list, alist, label = 'R', color = 'b')
plt.legend(loc=0)
plt.xlabel('T')
plt.ylabel('Action')
plt.title('action')
plt.show()