import random
import numpy as np
import math

# 充电部分
# 假设每个时刻生成固定数量的待充电公交车，共有5个充电桩，不考虑充电量
# 假设进行一次充电需要花费40分钟
# 生成公交车的时间为早上6点到晚上10点   48个时间段    一天中的任意时间都可以充电
# 分时电价
e_price=[1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2, 2, 2,  1.5, 1.5, 1.5, 1.5, 2, 2, 2.5, 2.5, 2, 1.5, 1]
bus_appear=[ random.randint(0,1) for i in range (72) ]      # 出现待充电公交车的数量
EMPTY_PENALTY = 150
EXCEED_PENALTY = 150
MESS_PENALTY = 100
'''for i in range(72):
    if i<19:
        bus_appear[i]=0
    elif 19 <= i < 22:
        bus_appear[i]=1
    elif 22 <= i < 25:
        bus_appear[i] = 2
    elif 25<=i<34:
        bus_appear[i]=1
    elif 34<=i<37:
        bus_appear[i]= 2
    elif 37<=i<43:
        bus_appear[i]=1
    elif 43<=i<48:
        bus_appear[i]=1
    elif 48<=i<52:
        bus_appear[i]=1
    elif 52<=i<56:
        bus_appear[i]=1
    elif 56 <= i < 62:
        bus_appear[i] = 2
    elif 62 <= i < 72:
        bus_appear[i] = 0'''

class env:
    def __init__(self,M):  # 充电桩数目
        self.M = M
        self.state = []
        self.bus_appear = bus_appear

    def reset(self):
        bus_now = self.bus_appear[0]
        charger_num = self.M
        charger=np.zeros(5)
        t=0
        self.state = [bus_now, charger_num, t,charger[0],charger[1],charger[2],charger[3],charger[4]]   # 记录每一个充电桩的状态
        return self.state

    def state_one(self, bus_now, charger_num, t, charger1, charger2, charger3, charger4, charger5):  # 等待人数   等待时间
        self.state = [bus_now, charger_num, t, charger1, charger2, charger3, charger4, charger5]
        return self.state

    # 决策部分
    # 假设每个时间段为20min
    def step(self, state,action):
        bus_now = self.state[0]
        charger_num = self.state[1]
        t = self.state[2]
        charger=np.zeros(5)
        for m in range(5):
            charger[m] = state[m+3]
        k= t // 60 % 24
        price=e_price[k]
        if action != 0:      # action 为某一时间段 决定的充电公交车的数量
            # 若充电
            reward = - action*price*2/3  # 充电扣费
            bus_now = bus_now - action
            charger_num -= action
            for x in range(action):
                j = 0
                while charger[j]!=0:      # 找到未被使用的充电桩
                    j += 1
                    if j==5:      # 所有的充电桩均被占用
                        break
                if j==5:
                    charger_num += 1
                    bus_now += 1
                    '''reward = reward - MESS_PENALTY + price*2/3          # 减去分配混乱的惩罚，加上多扣除的费用'''
                else:
                    charger[j]=40
            '''if bus_now < 0:
                reward -= EMPTY_PENALTY   # 惩罚1: 待充电车的数量<0不合理'''
        else:
            reward = 0
        if bus_now >= 5:
            reward -= EXCEED_PENALTY # 惩罚2： 防止待充电车累积过多
        t += 20
        bus_now += bus_appear[t // 20 % 72]
        for j in range(5):
            if charger[j] != 0:
                charger[j] -= 20
        self.state = [bus_now, charger_num, t, charger[0], charger[1], charger[2], charger[3], charger[4]]
        return self.state, reward, action         # 求reward的最小值