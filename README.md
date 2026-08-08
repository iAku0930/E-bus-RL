# E-bus 调度与充电强化学习

电动公交（E-bus）调度 + 充电的强化学习实验。对比两种方法：

- **dual-stage**：单多头 PPO 决定发车/充电的时机和数量，启发式按电量排序选具体车。思路来自本组 E-taxi 工作 [1]，针对公交场景做了适配。
- **双 PPO 联合**：发车和充电各用一个 actor，共享一个全局 critic（集中式训练）。

[1] *Charge or Pick up? Optimizing E-Taxi Management: A Dual-Stage Heuristic Coordinated RL Approach*, IEEE T-ASE 2025.

## 结果

设置：同质车队 12 车 4 桩、连续 5 天运营、高峰拥堵、泊松客流、分时电价（EP400，2 seed 平均）。

| 方法 | 日均服务量 | 日均充电花费 | 日均总收益 | 低谷充电占比 |
|---|---|---|---|---|
| 双 PPO 联合 | 848 | 1015 | 768 | 65% |
| dual-stage | 861 | 766 | 1084 | 91% |

dual-stage 全面更好一些。同质车队里按电量排序选车已经接近最优，用 RL 去学"选哪辆车"意义不大，反而不如规则；真正需要 RL 的是发车和充电的时机判断。

## 环境

- 固定线路，同质车队
- 客流按真实日内曲线 + 泊松随机
- 分时电价，峰谷比约 3:1
- 高峰拥堵：单趟 1.5h → 2h
- 乘客等待过久会流失
- 多日连续运营，电量跨天继承

## 运行

```
pip install -r requirements.txt
cd src
python compare.py      # 对比实验，图在 results/cmp_*.png
python visualize.py    # 全天策略图 results/day_strategy.png
```

## 目录

- `src/` — 代码：config / ebus_env / ppo / train / compare / visualize
- `reference/` — 早期参考实现
- `docs/` — 成果总结.md、论文框架.md、草稿/

优化目标是日总收益最大化：接客收入 − 等待成本 − 流失损失 − 充电花费（末剩余电量作为评估指标，不计入 reward）。
