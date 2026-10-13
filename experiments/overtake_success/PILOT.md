# Success-trigger Pilot

> 历史报告：此文件使用已废弃的“超越后在左车道保持 1 s”定义，数字不能代表完整超车成功率。当前定义为先左换道、超越、返回原车道并稳定领先 1 s，重跑报告见 [PILOT_RETURN.md](PILOT_RETURN.md)。本文件命令是历史运行记录；当前脚本采用新定义，应使用新报告中的独立输出目录。

日期：2026-10-13。这是原始场景上的奖励可触发性诊断，不是正式四组实验，也不使用 sealed held-out 数据。

## Scope

1. 重跑现有 selected policies：B 4100/4101、C 4200/4201，各 FD/BAL/SP，每模型 36 个 original development scenes。最多 120 s，首次严格成功或环境结束时停止，共 432 episodes。它回答旧模型是否触发拟定判据，不用于 final-100K 因果比较或成功后的安全结论。
2. 从头短训练：B/4190、C/4290，BAL，S00/S10 各 2,000 steps。使用原始 stratified 训练场景，不增加课程、不筛选可行场景、不添加 dense shaping。它直接检查训练探索是否出现严格成功，以及事件是否进入 replay 并被优化器采样。
3. 100K 扩展 runner/PBS 已准备，当前机器无 `qsub`，未提交。扩展为 B/C × 每 arm 两个预留 seeds × FD/BAL/SP × S00/S10，共 24 models。先诊断奖励触发，再考虑正式四组 pilot；不自动启动 README 中的 48-model 全四组方案。

## Implementation

- [pilot.py](pilot.py) 固定初始目标身份，在每个 `road.step` 动力学及碰撞处理后被动检查几何条件，policy step 末发一次性事件。
- 保留 README 的严格判据：车尾净领先 5 m、左车道中心偏差 ≤0.5 m、连续 1 s、无碰撞/驶出道路。没有为了提高触发率修改定义。
- 训练的 success marker 随实际 transition 进入 replay。单环境 replay buffer 同步记录真实 sample indices，统计带成功事件的采样次数和至少被采样一次的独立成功 transition 数，不额外采样或消费随机数。
- S00 记录同一事件但不加 bonus；S10 在该步额外加 30。成功次数不代表 S00 实际获得成功奖励。
- 2K 短训练使用正式 100K 的探索日程前缀：epsilon 在 25K steps 内从 1 降至 .05，2K 时仍为 .924；learning starts=1000。不能将 exploration_fraction=.25 直接套到 2K 总预算，否则会在第 500 步就结束衰减。
- 初次接线运行 `short_*` 使用了压缩探索日程，保留原产物供审计，但不纳入下述有效结果；后续 `prefix_*` 为修正日程后的结果。collector 显式排除 `short_*`。
- training budget 到达时未结束的 episode 记为 `budget_stop`，不冒充 timeout。汇总保留 no-front N/A 和未完成 episode 数量。

## Commands

从项目根目录运行。输出目录必须不存在；脚本拒绝覆盖。

```bash
PYTHONPATH=src .venv/bin/python experiments/overtake_success/pilot.py old-policy \
  --out outputs/overtake_success_pilot/old_policy_2seed

PYTHONPATH=src .venv/bin/python experiments/overtake_success/pilot.py train \
  --arm B --agent BAL --seed 4190 --group S10 --steps 2000 \
  --out outputs/overtake_success_pilot/prefix_B_BAL_S10_seed4190

python3 experiments/overtake_success/collect_pilot.py \
  --root outputs/overtake_success_pilot

# 在 Katana 项目目录中运行（尚未提交）：
qsub experiments/overtake_success/katana_overtake_trigger_pilot.pbs
```

C 使用 seed4290；S00 将 group 改为 S00 并使用独立输出目录。完整扩展使用 B4190/4191、C4290/4291，100K 步。PBS 路径为 [katana_overtake_trigger_pilot.pbs](katana_overtake_trigger_pilot.pbs)。

## Existing-policy Results

| Arm | FD：20 s / 120 s 严格成功 | BAL：20 s / 120 s 严格成功 | SP：20 s / 120 s 严格成功 |
| --- | --- | --- | --- |
| B | 72/72 / 72/72 | 72/72 / 72/72 | 72/72 / 72/72 |
| C | 0/72 / 0/72 | 0/72 / 0/72 | 0/72 / 0/72 |

每格为 2 seeds × 36 scenes。B 合计 216/216 在 20 s 内通过严格判据，三个 agent 的成功时间中位数均为 12.6 s。C 合计 0/216 通过严格判据，但 216/216 都曾进入左车道，也都曾达到车尾净领先 5 m；15/40/70 m 三种左车 gap 下均出现该现象。两项“曾经”指标并不单独证明完整成功窗口，详细轨迹案例见下节。

两 arm 在各自诊断停止时点前均未记录碰撞。B 成功后即停止，不能据此断言随后 120 s 无碰撞；C 未严格成功，实际观察到 120 s。结果仅覆盖指定两个 seeds，不推广为全部旧模型的成功率。

逐 episode 结果见 `outputs/overtake_success_pilot/old_policy_2seed/old_policy_episodes.csv`；按左车 gap 的汇总见 `old_policy_summary.csv`。

## Short-training Results

下表仅列有效的 `prefix_*` runs，每个 run 都是 2,000 policy steps：

| Arm / seed | Group | 首次成功 step | 成功 transitions | replay 中成功采样次数（可重复） | 至少被采样一次的成功 transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| B / 4190 / BAL | S00 | 575 | 2 | 38 | 2 |
| B / 4190 / BAL | S10 | 575 | 2 | 38 | 2 |
| C / 4290 / BAL | S00 | 779 | 1 | 23 | 1 |
| C / 4290 / BAL | S10 | 779 | 1 | 23 | 1 |

B 每组有 17 个已结束且含初始前车的 episodes，其中 2 个成功；C 每组为 18 个，其中 1 个成功。另分别保留无目标场景和 1 个 budget-stop episode，不将它们混入该分母。

B 的两次成功来自 visible-slow-front 场景，初始距离约 91.9/106.8 m，在 episode 内 13.6/12.0 s 触发。C 的一次成功初始前车距离约 166.2 m、前车速度约 12.0 m/s、左车 gap 约 23.0 m，在 15.8 s 触发。

这些成功都发生在前 1,000 步随机 warm-up 内，随后确实被训练更新采样。因此本样本反驳“从头训练必然完全接触不到这个奖励”，但成功仍然稀疏。S00/S10 在本次短前缀中事件数一致，不是奖励无效的证据，也不是已经学会超车的证据。每个 arm 只有一个 seed、一个 agent、2K steps，不能据此宣称多 seed 学习稳定或 onset 已改变。

## Definition Risk Observed in Arm C

实际回放 C4200/FD/M0000：8.8 s 时 ego 在左车道，车尾已净领先慢前车约 5.99 m；9.0 s 开始向右返回，9.4 s 实际回到右车道，轨迹在检查窗口内无碰撞。由于净领先之后在左车道中心附近保持不足 1 s，严格成功奖励不触发。

这说明“从未触发严格判据”不等于“没有完成物理超车”。当前判据额外要求超过后留在左车道，会排除快速安全返回右车道的行为。

建议在正式训练前明确任务语义：若目标是“超过初始慢前车”，可考虑将确认阶段改成“已经完成左换道，且连续 1 s 保持净领先和无碰撞，允许期间返回右车道”。这将是显式的新协议版本，需要重新评估，不能用该建议重标本次严格判据结果。本轮没有改成功定义，也没有加入课程或 shaping。

## Validation and Limits

- 26 项相关测试通过：包括车身净距离、横向偏差、持续时间、碰撞优先、无目标、事件一次性、replay 实际采样，以及被动 detector 不改变环境轨迹。
- PBS 语法及 git diff whitespace 检查通过。
- 四个有效短训练模型均已重新加载，并完成 predict/step 检查。
- 本诊断未执行完整可行性 witness 搜索，没有证明所有 Arm C 场景均可完成。
- 旧策略使用 maintained selected models；短训练是新初始化 BAL，不将两者当作同一学习曲线。
- 结果文件位于 `outputs/overtake_success_pilot/`，源码和简要结果记录在本实验目录，生成产物不入版本控制。
