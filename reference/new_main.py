import numpy as np
import torch
from matplotlib import pyplot as plt
import openpyxl
import hucl_PPO
import new_env
import rl_utils

actor_lr = 1e-3
critic_lr = 1e-2
state_dim = 7
action_dim1 = 2
action_dim2 = 2
hidden_dim = 4
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
seed = 1 # 随机数种子
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
buffer_size = 10000
num_episodes = 5000
batch_size = 64
NN,M,C0 = 20,20,10
MAX=45


env = new_env.env(NN,M,C0,MAX)

def train(env, agent1, agent2,num_episodes, w1, w2, w3):
    r_list = []
    r1_list , r2_list, r3_list = [],[],[]
    for i_episode in range(num_episodes):
        state = env.env_reset()
        aa1,aa2 = 0,0
        r_day = 0
        r1_day,r2_day,r3_day = 0,0,0
        for i in range (7):
            state[i] = state[i]/100
        for j in range(48):
            t=100*state[0]
            if t < 24:
                a1, probs1, value1 = agent1.take_action(state)
                a2, probs2, value2 = agent2.take_action(state)
                state2 = np.zeros(7)
                for i in range(7):
                    state2[i] = 100 * state[i]
                state_new,reward,aa1,aa2,r1,r2,r3=env.env_step(state2,a1,a2,w1,w2,w3)
                memory1.push(state, a1, probs1, value1, reward)
                memory2.push(state, a2, probs2, value2, reward)
                for i in range(7):
                    state_new[i] = state_new[i] / 100
                state = state_new
                r_day += reward/5
                r1_day += r1
                r2_day += r2
                r3_day += r3

            else:
                break
        r_list.append(r_day)
        r1_list.append(r1_day)
        r2_list.append(r2_day)
        r3_list.append(r3_day)
        if i_episode%10 == 0:
            agent1.update()
            agent2.update()
            print('episode: ', i_episode, 'reward: ', r_day)
           # print('a1: ',aa1)
            #print('a2: ',aa2)

    return r_list,r1_list,r2_list,r3_list
memory1 = hucl_PPO.PPOMemory(buffer_size)
memory2 = hucl_PPO.PPOMemory(buffer_size)
agent1 = hucl_PPO.PPO(state_dim, hidden_dim, action_dim1, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, memory1)
agent2 = hucl_PPO.PPO(state_dim, hidden_dim, action_dim2, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, memory2)

r_list, r1_list, r2_list, r3_list =train(env,agent1,agent2,num_episodes, 0.2,0.2,0.2)

episodes_list = list(range(len(r_list)))
mv_return = rl_utils.moving_average(r_list, 39)
plt.plot(episodes_list, mv_return, label = '1', color = 'b')
plt.legend(loc=0)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('reward')
plt.show()

episodes_list = list(range(len(r_list)))
mv_return1 = rl_utils.moving_average(r1_list, 39)
plt.plot(episodes_list, mv_return1, label = '1', color = 'b')
mv_return2 = rl_utils.moving_average(r2_list, 39)
plt.plot(episodes_list, mv_return2, label = '2', color = 'r')
mv_return3 = rl_utils.moving_average(r3_list, 39)
plt.plot(episodes_list, mv_return3, label = '3', color = 'y')
plt.legend(loc=0)
plt.xlabel('Episodes')
plt.ylabel('r')
plt.title('R')
plt.show()
