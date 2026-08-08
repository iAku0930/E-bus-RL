import random
import numpy as np
import math

# 假设起点和终点各有10辆公交车； 各有5个充电桩，每辆公交车充满电时可以沿规定路线运行 5 趟（也就是说，每次运行消耗电动公交车电量的20%）
# 假设乘坐公交车的费用一致，无论在哪个站点上车都需付一定的费用
# 假设充电时间为 2h ，每完成一次充电电动公交车电量增加40%   路线一趟运行1h
# 假设公交车路线上中途有若干个站点

# 充电桩 分时电价
E_price_1 = [1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1]    # 公交 电费

# 定义上限
# 每个时间段客流量增加量上限 NN = 20      起点站和终点站分别的公交数 M = 20       起点站和终点站分别的充电桩数 C=10     公交车的客容量 MAX=45

# 环境
class env:
    def __init__(self, NN, M, C0,MAX):
        self.NN = NN
        self.M = M
        self.C0 = C0
        self.MAX = MAX

    def state_one(self,t,people_now,L,E,C,waiting_time):       # c_charger为当前空闲充电桩数 n为可充电的公交车数
        state=[]
        k= math.floor(t)     # 计算当前时段电价
        e_1=E_price_1[int(k)]
        state.append(t)
        state.append(people_now)
        state.append(e_1)
        state.append(L)
        state.append(E)
        state.append(C)
        state.append(waiting_time)
        return state

# 定义初始状态

    def env_reset(self):
        E=np.zeros(20)
        L=np.zeros((20,3))
        C=np.zeros((10,2))
        people_now=random.randint(1,self.NN)
        for i in range(0,20):   #L有3个分量（当前车站，运行剩余时间，上一站点的位置，可使用状态）电动车电量
            E[i]=1
            if i<=10:
                L[i][0]= 0
                L[i][1]= 0
                L[i][2]= 1
            else:
                L[i][0] = 1
                L[i][1] = 0
                L[i][2] = 0
        for j in range(0,10):                        #C有两个分量（可使用状态，剩余使用时间）
                C[j][0]=0
                C[j][1]=0
        waiting_time=0
        e_1=E_price_1[0]
        t=0
        state = [t,people_now,e_1,L,E,C,waiting_time]
        return state

# 决策部分
    # 假设每个时间段为 0.5h
    def env_step(self,state,a1,a2,w1,w2,w3):    # a1 为0/1变量 表示是否发车  #  a2 充电动作
        s=0    #所运送总人数
        reward=0   #奖励
        test1=test2=0
        t,people_now=state[0],state[1]
        e_1=state[2]
        L=state[3]
        E=state[4]
        C=state[5]
        waiting_time=state[6]

        for i in range(0,M,1):     #发车部分状态转移
            test1 += a1[i]
            if L[i][3]==1 and a1[i]==1:            #单一状态
                reward-=5
            if a1[i]==1:
                L[i][0]=2
                L[i][1]=1/0.5
                if L[i][2]==0:
                    L[i][2]=1
                elif L[i][2]==1:
                    L[i][2]=0
                L[i][3]=1        #正在使用
                E[i]-=0.2
                s = people_now
                people_now = 0
                waiting_time = 0
                break
        if test1  > 1:
            reward -= 3

        for k in range(0, C0):         #充电部分状态转移
            for i in range(0,M):
                test2+=a2[i][k]
                if (L[i][3]==1 and a2[i][k]==1) or(C[k][0]==1 and a2[i][k]==1):
                    reward-=5
                if a2[i][k]==1:
                    C[k][0]=1
                    C[k][1]=2/0.5
                    C[k][2]=i
                    L[i][3]=1
                    t0=0.5
            if test2>1:               #同一个充电桩不能给多个公交车供电
                reward-=5

        t+=0.5                          #环境状态转移
        for i in range (0,M):
            if L[i][1]!=0:
                L[i][1]-=1
                if L[i][1]==0:
                    L[i][3]=0
                    if L[i][2]==0:
                        L[i][0]=1
                    elif L[i][2]==1:
                        L[i][0]==0
        people_now+= random.randint(1,self.NN)
        waiting_time+=0.5
        for k in range(0,C0):
            if C[k][0]==1:
                C[k][1]-=1
                f=t0*e_1
                if C[k][1]==0:
                    C[k][0]=0
                    E[C[k][2]] += 0.4
                    L[C[k][2]][3] = 0
        e_1 = E_price_1[math.floor(t / 1)]
        state_new=self.state_one(t,people_now,e_1,L,E,C,waiting_time)
        #奖励
        r1=s*5
        r2=waiting_time*2
        r3=f/2
        reward+=w1*r1-w2*r2+w3*r3       # r越大越好
        if people_now >= MAX:
            reward -= 3

        return state_new,reward,a1,a2,r1,r2,r3