# 实现说明 (DESIGN)

研究过程见 `docs/成果总结.md`，论文定位见 `docs/论文框架.md`。

## 定位

把 E-taxi dual-stage RL（本组前作，IEEE T-ASE 2025）的思路用到 E-bus 上，并和"双 PPO 联合"对比。结论：dual-stage（单多头 PPO + 启发式）在 E-bus 同质车队下更好。

## 实例和环境

| 参数 | 值 | 说明 |
|---|---|---|
| m / n | 12 车 / 4 桩 | 同质车队，桩/车比 0.33 |
| CAP | 25 | 单车容量 |
| 满电续航 | 约 4 趟（耗电 0.25/趟） | |
| 充电时间 | 4 段 (2h) | |
| 运行时间 | 正常 3 段(1.5h) / 高峰 4 段(2h) | 拥堵 |
| 天数 | 5 天/episode | 电量跨天继承 |
| 客流 | 日内曲线 + 泊松随机 | 早 7-9 / 晚 17-19 高峰 |
| 电价 | 分时，峰谷比约 3:1 | |

reward（元）：`+2×carried − 0.5×waiting − price×60×充车辆 − 2×流失`。末剩余电量只做评估指标，不进 reward（放进 reward 会让双 PPO 方案屯电）。

## 状态 / 动作

- 状态（定长 124 维）：每车 [pos_oh, rem_run, batt, last_stop, status_oh, chg_rem] + 全局 [price, waiting, wait_time, 空闲桩数]。
- 发车：O 端发不发 / E 端发不发，启发式选电量最高的车。
- 充电：充几辆 ∈ {0..n}，启发式选电量最低的车。
- 动作掩码：没有可动车/桩时强制不发/不充，用来处理动态决策时机。

## 两种方法

**方法 A（dual-stage）**：单多头 PPO（发车 2 头 + 充电 1 头），同状态一次决策；启发式执行（电量最高发车、电量最低充电）。

**方法 B（双 PPO 联合）**：发车 actor + 充电 actor，充电看发车后状态；共享一个全局 critic（集中式训练）。

## 实验结果

| 方法 | 日均SP | 流失% | 日均CE | 末电量 | 日均总收益 | 低谷充电% |
|---|---|---|---|---|---|---|
| B: 双PPO联合 | 848 | 3.8 | 1015 | 0.53 | 768 | 65 |
| A: dual-stage | 861 | 2.4 | 766 | 0.60 | 1084 | 91 |

## 代码

`config.py` 参数；`ebus_env.py` 环境；`ppo.py`（PPO / 多头 / JointCentralized）；`train.py`（train_joint / train_source）；`compare.py` 对比实验；`visualize.py` 策略可视化。

## 运行

```
pip install -r ../requirements.txt
cd src && python compare.py
cd src && python visualize.py
```
