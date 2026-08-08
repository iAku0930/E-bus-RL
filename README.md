# E-bus 调度与充电协同管理强化学习

将 **dual-stage 强化学习方法**（启发式协调 + RL 决策时机/数量）应用于**电动公交（E-bus）调度 + 充电**场景，并与"双 PPO 联合智能体"方案进行系统对比。

> 本工作受本组 E-taxi 前作（*"Charge or Pick up? ... A Dual-Stage Heuristic Coordinated RL Approach"*, IEEE T-ASE 2025）启发，将其迁移并适配到 E-bus 同质车队、固定线路、连续多日运营的场景，并给出对比发现。

## 核心结论

在 E-bus **同质车队**场景下，**dual-stage 方法（单多头 PPO 决策时机/数量 + 启发式选车）优于双 PPO 联合智能体**：

| 方法 | 日均服务量 SP | 日均充电花费 CE | 日均总收益 | 低谷充电占比 |
|---|---|---|---|---|
| 双 PPO 联合 | 848 | 1015 | 768 | 65% |
| **dual-stage（主方法）** | **861** | **766** | **1084** | **91%** |

> 详见 [`docs/成果总结.md`](docs/成果总结.md)。

## 场景与方法

- **现实建模**：同质车队、固定线路、聚合客流（泊松）、分时电价（峰谷比~3:1）、**高峰拥堵**（1.5h→2h）、**乘客流失**、**连续多日运营**（电量跨天继承）。
- **主方法（dual-stage）**：单多头 PPO 同状态决策"发车时机/数量 + 充电时机/数量"，启发式按电量排序选具体车。
- **对比方法（双 PPO 联合）**：发车 actor + 充电 actor 解耦耦合，集中式 critic。

## 目录结构

```
src/
  config.py        # 实例规模、客流/电价、reward 经济参数、超参
  ebus_env.py      # 环境（状态/动作/转移/启发式执行/多天/拥堵/流失）
  ppo.py           # PPO / 多头PPO / JointCentralized（集中式critic）
  train.py         # train_joint / train_source / train_single
  compare.py       # 主对比实验 → results/cmp_*.png
  visualize.py     # 全天策略可视化 → results/day_strategy.png
  DESIGN.md        # 实现设计说明
  results/         # 实验图表
docs/
  成果总结.md       # 过程、结论、代码结构、复现步骤
  论文框架.md       # 论文定位、贡献点、章节结构
  *.docx           # 论文草稿
```

## 快速开始

```bash
# 依赖（Python 3.10）
pip install -r requirements.txt

# 跑主对比实验（联合 vs 源方法）
cd src && python compare.py

# 策略可视化
cd src && python visualize.py
```

## 优化目标

最大化 **日总收益 = 接客收入 − 乘客等待成本 − 流失损失 − 充电花费 + 末剩余电量价值**

## 相关工作说明

本工作的 dual-stage 思想源自本组 E-taxi 论文（IEEE T-ASE 2025）。本工作的贡献在 **E-bus 场景建模、方法适配、对比发现**，与源论文的创新边界详见 [`docs/论文框架.md`](docs/论文框架.md)。
