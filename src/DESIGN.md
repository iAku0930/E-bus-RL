# E-bus 调度+充电强化学习 —— 实现设计 (DESIGN)

> 本文档为最终实现说明。研究过程与方法演进见 `docs/成果总结.md`、论文定位见 `docs/论文框架.md`。

## 一、项目定位

将源论文（本组 E-taxi dual-stage RL, IEEE T-ASE 2025）方法**迁移适配到 E-bus**，并与双 PPO 联合方案对比。最终结论：**源方法（单多头 PPO + 启发式）在 E-bus 同质车队场景下更优**，作为主方法；双 PPO 联合为对比方案。

## 二、实例与环境（现实化）

| 参数 | 值 | 说明 |
|---|---|---|
| m / n | 12 车 / 4 桩 | 同质车队，桩/车比 0.33 |
| CAP | 25 | 单车容量 |
| 满电续航 | ~4 趟（耗电 0.25/趟） | |
| 充电时间 | 4 段 (2h) | |
| 运行时间 | 正常 3 段(1.5h) / 高峰 4 段(2h) | 拥堵 |
| 天数 | 5 天/episode | 电量跨天继承 |
| 客流 | 现实日内曲线 + 泊松随机 | 早7-9/晚17-9高峰 |
| 电价 | 分时，峰谷比 ~3:1 | |

**reward（经济标定，元）**：`+2×carried − 0.5×waiting − price×60×充车辆 − 2×流失`；
末剩余电量作**评估指标**（不进 reward，避免诱导屯电）。

## 三、状态 / 动作

- **状态**（定长 124 维）：每车[pos_oh, rem_run, batt, last_stop, status_oh, chg_rem] + 全局[price, waiting, wait_time, 空闲桩数]。
- **发车动作**：O 端发不发 / E 端发不发（启发式选电量最高车）。
- **充电动作**：充几辆 ∈ {0..n}（启发式选电量最低车）。
- **动作掩码**：实现"动态决策时机"——无可行车/桩时强制不发/不充。

## 四、两方法

### 主方法：源方法（dual-stage + 启发式）
- 单多头 PPO（发车 2 头 + 充电 1 头），同状态一次决策；
- 启发式执行：电量最高发车、电量最低充电。

### 对比：双 PPO 联合（集中式 critic）
- 发车 actor（2 头）+ 充电 actor（1 头），充电看发车后状态（耦合）；
- **共享全局 critic**（集中式训练）。

## 五、实验结论

| 方法 | 日均SP | 流失% | 日均CE | 末电量 | 日均总收益 | 低谷充电% |
|---|---|---|---|---|---|---|
| 联合(双PPO) | 848 | 3.8 | 1015 | 0.53 | 768 | 65 |
| **源方法(主)** | **861** | **2.4** | **766** | 0.60 | **1084** | **91** |

源方法全面更优。详见 `docs/成果总结.md`。

## 六、代码结构（src/）

- `config.py` 参数；`ebus_env.py` 环境；`ppo.py`（PPO/多头/JointCentralized）；`train.py`（train_joint/train_source）。
- `compare.py` 主对比（→ results/cmp_*.png）；`visualize.py` 策略可视化（→ day_strategy.png）。
- `archive/` 过程脚本。

## 七、运行

```bash
pip install -r ../requirements.txt   # torch/numpy/matplotlib/openpyxl/tqdm
cd src && python compare.py     # 对比实验+图
cd src && python visualize.py   # 策略可视化
```
