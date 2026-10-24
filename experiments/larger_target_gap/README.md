# Larger target gap pilot

独立开发 pilot；设计见 [DESIGN.md](DESIGN.md)。不更改原 A/B/C 或 held-out v2。

## Broad-distance retraining v2

新一轮重训协议与命令见 [BROAD_V2.md](BROAD_V2.md)：白车距离 40–150 m，独立输出前缀，9 个全新初始化的 100k-step 模型。Katana 提交 `qsub scripts/katana_larger_target_gap_broad_v2.pbs`；收集时使用 `collect.py --profile broad-v2`。`by_distance.csv` 按白车距离报告发生率和时间，未观察到换道及碰撞均保留。下方 v1 命令仍可复现原 pilot。

## Frozen pilot v1

- 9 个模型：seed 5200、5201、5202 × FD/BAL/SP，index 按 seed 优先依次映射 0–8。
- 每模型 100,000 steps；训练和评估每 episode 20 s；5 Hz 决策、15 Hz 仿真。使用 final fixed-step checkpoint，不按排序或 gate 重选模型。
- 原车道距离 40–80 m；黄车距离为原车道距离 + 30–80 m。ego 28 m/s，慢白车 10–20 m/s，黄车 29 m/s。
- 每 10 个 reset：6 个慢前车、2 个无白车、2 个白车同速场景，按 seed/block 独立洗牌。无白车时黄车仍保留，距离从同一关联分布生成。
- 同一 seed 的奖励条件共享按 reset index 定义的场景序列；由于 episode 长度受策略影响，100k steps 内实际经历的 reset 数量可能不同，训练场景 CSV 保留这些差异。
- 奖励沿用现有 MAIN_REWARD_WEIGHTS，公共 slowdown=0.2、lane-change=0.2、collision-risk=3.0，碰撞权重沿用各条件共同默认值 2.0；其余 DQN 超参数由保存的 config.json 完整记录。
- 27 场景：D_o = 45/60/75 m，ΔD = 35/55/75 m，白车速度 = 10/14/18 m/s。
- 每模型评估 original、no-front、open-target-lane、matched-speed-front，共 108 episodes。ΔD 剂量比较已包含于主网格。
- 这是开发诊断，pilot 不做正式假设检验，不根据排序是否显著决定保留结果。正式 held-out 与正式训练另行冻结。

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_larger_target_gap.py tests/test_occupied_target_lane.py
.venv/bin/python experiments/larger_target_gap/run.py --index 0 --execute --preflight-only
.venv/bin/python experiments/larger_target_gap/run.py --index 0 --execute --smoke
```

Preflight 包括 27 个开发场景和训练最紧边界，检查车辆可见、位置观测未裁剪，并验证立即换道脚本能够完成换道而无碰撞。IDLE 对照不减速也不换道；它的碰撞不代表训练策略会碰撞。

Smoke 仅训练 200 steps、评估每 variant 3 个场景，只验证流程，不用于评价学习效果。输出不覆盖；重复验证需指定新的 `--out-root`。

## Katana execution

在含本项目完整源码的 Katana checkout 根目录执行：

```bash
qsub scripts/katana_larger_target_gap_pilot.pbs
```

脚本复用 `/srv/scratch/$USER/venvs/highway-transition`，输出到 `/srv/scratch/$USER/highway_transition_timing/outputs/larger_target_gap_pilot_<agent>_100k_seed<seed>`。申请每任务 4 CPU、16 GB、5 小时。提交前需确认该环境存在；每任务都会独立执行 preflight。

## Collection and interpretation

```bash
python3 experiments/larger_target_gap/collect.py \
  --root /path/to/outputs --out /path/to/new_pilot_summary
```

Collector 列出全部 9 个预期任务的 complete/incomplete/missing 状态，缺失任务不默默剔除。`summary.csv` 包含每 seed/condition/variant 的发生率、碰撞、未换道未碰撞数量、首步动作数量和换道前空间检查。平均时间仅针对已观察到换道的样本，是描述统计，不据此直接比较策略优劣。

逐步日志保留白车/黄车的固定身份及前后状态；净间距为中心距离减 5 m。车辆已被超过或干预中不存在时，不把它当成当前前车参与空间优势比例。完整轨迹仍保留其带符号相对位置。首次物理换道按 post-step 车道索引变化计时，不等同于完全居中。

每个任务保存 protocol/config/provenance、开发网格、前 100 个计划 reset、实际训练场景与训练日志、检查点、逐步评估、episode 汇总和 complete 标记。最终学习效果必须等 9 个真实 pilot 模型完成后报告。
