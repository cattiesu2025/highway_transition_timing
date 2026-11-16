# B / BAL / S10：100K 训练结果检查

日期：2026-10-14。Run：`full_B_BAL_S10_seed4190_local_retry1`，Katana job `9510008`。

**结论：奖励已实际触发并进入学习，但最终模型在本次无探索评估中没有完成“超越后返回原车道”。不能把训练成功比例当作最终策略成功率。**

## 设定与完整性

- B / BAL / seed4190，S10：完整超车奖励 +30、不因成功提前终止。换道动作惩罚仍为 −0.2。
- 定义 `complete_overtake_return_v1`：完成左换道 → 左侧净领先初始前车至少 5 m → 回原车道中心附近并保持净领先 1 s；无碰撞或驶出道路。
- 100,000 步，1,015 个已结束 episode，另有 1 个预算截断 episode。各 episode 步数之和等于训练预算。
- `complete.json`、`progress.json` 完成；模型 ZIP 完整性检查通过。训练 manifest 中四项源文件 SHA256 与当前本地文件一致。
- 日志训练耗时约 456.86 s；首次成功 step781。348 条成功 transition 入库，训练采样命中 8,871 次，覆盖 341 条不同成功 transition。

## 带探索的训练统计

排除最后一个预算截断 episode；无前车 episode 不进入“完整超车成功率”分母。

| 样本范围 | 成功 / episode | 比例 |
| --- | ---: | ---: |
| 全程有前车且已结束 | 348 / 811 | 42.9% |
| 全程 visible slow-front | 314 / 406 | 77.3% |
| 最后 20K 步 visible slow-front | 74 / 79 | 93.7% |

全程已结束 episode 中有 38 次碰撞；最后 20K 步无碰撞。训练末期仍有 epsilon=0.05 的随机探索，且训练中策略持续更新。这些比例不能代替保存后的最终策略评估。

## 最终模型：关闭探索

同一最终模型、确定性动作；成功奖励不影响固定策略动作，评估 detector 仅记录成功。固定 development 场景继续观察到 120 s；训练场景复测使用原 20 s 窗口。

| 评估集 | 完整成功 | 碰撞 | 首次换道 onset 中位数 |
| --- | ---: | ---: | ---: |
| 原 development 36 场景，20 s 内 | 0 / 36 | 0 / 36 | 2.8 s |
| 同一 development 场景，120 s 内 | 0 / 36 | 0 / 36 | 2.8 s |
| 训练最后 20K 步的 79 个 slow-front 原场景，20 s 内 | 0 / 79 | 0 / 79 | 0.8 s |

development 中 36/36 完成左换道且在左侧净领先 ≥5 m，0/36 返回并领先。全部 21,600 个 policy steps 中，36 次 LANE_LEFT、89 次 IDLE、21,475 次 FASTER，**没有 LANE_RIGHT**。失败阶段是返回原车道，并非没有超过前车。首次 onset 范围为 0.4–6.6 s；尚无配对 S00 完整训练，不能据此推断奖励令 onset 提前或推迟。

训练场景复测逐项核对 scene spec 与原训练日志一致，因此不能简单把差异归因于 development 场景不同。随机探索、训练期间策略变化/最终策略退化等仍是待检验解释；当前日志不足以识别训练中每次返回来自贪心动作还是随机动作。

## 本地评估与复现边界

使用 `evaluate_trained.py`，仅支持 Arm B，未使用 held-out 场景。模拟器 highway-env=1.12.0、gymnasium=1.3.0 与训练一致；安装于临时隔离目录，没有替换项目 venv。SB3 均为 2.9.0。本地 Python=3.13 / Torch=2.14.1 与集群 Python=3.11 / Torch=2.13.0+cpu 不同，最终结论仍建议在原集群环境复核。

跨 Python 版本加载训练时 cloudpickle 的 replay 类曾导致本地崩溃；评估脚本用本地同名 replay 类和未使用的训练 schedule 覆盖，仅分配一个 replay 槽。加载后逐项验证所有 policy state_dict 张量与 ZIP 中保存值完全相等；未修改网络权重。

结果位于 `outputs/overtake_success_return_pilot/full_B_BAL_S10_seed4190_local_retry1/`：

- `training_summary.json`、`training_episodes.csv`：训练原始结果。
- `evaluation_development_verified/`：36 个场景、120 s 的确定性评估。
- `evaluation_training_tail_verified/`：79 个训练原场景、20 s 的确定性复测。
- 每个评估目录包含 manifest、episodes、压缩逐步轨迹、summary、完成标记。早期诊断目录保留，以上两个目录为最终复核输出。

在与训练匹配的环境中，从仓库根目录运行（输出目录必须不存在）：

```bash
PYTHONPATH=src python experiments/overtake_success/evaluate_trained.py \
  --run-dir outputs/overtake_success_return_pilot/full_B_BAL_S10_seed4190_local_retry1 \
  --out outputs/overtake_success_return_pilot/B_BAL_S10_eval_development

PYTHONPATH=src python experiments/overtake_success/evaluate_trained.py \
  --run-dir outputs/overtake_success_return_pilot/full_B_BAL_S10_seed4190_local_retry1 \
  --out outputs/overtake_success_return_pilot/B_BAL_S10_eval_training_tail \
  --training-tail
```

## 后续判断

本次回答了“奖励能否被触发并被采样”：可以，而且不止偶发一次。尚未证明最终策略学会完整超车，也未回答四个 RQ 的组间效应。保持换道惩罚不变；后续应先在原运行环境复核，并加入训练中固定场景的确定性 checkpoint 评估，区分中途学会后退化与始终依赖探索，再决定是否扩展正式四组实验。
