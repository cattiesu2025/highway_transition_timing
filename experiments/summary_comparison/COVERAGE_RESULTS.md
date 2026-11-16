# A–D clip/onset 覆盖率结果

2026-10-28 核验完成。补齐 DISAGREEMENTS，并新增 Arm D（C distance-affine-v1）。
本表统计所选 clip 的公共时间窗口是否包含对应真实 leader episode 的 onset；不是 clip 间 IoU。

| Arm | HIGHLIGHTS 原版 max-min | HIGHLIGHTS Huber max-second | DISAGREEMENTS last-state |
| --- | --- | --- | --- |
| A：单车道 | 14/300（4.7%） | 28/300（9.3%） | 334/600（55.7%） |
| B：空目标车道 | 129/300（43.0%） | 94/300（31.3%） | 248/600（41.3%） |
| C：占用目标车道 | 73/300（24.3%） | 86/300（28.7%） | 150/600（25.0%） |
| D：新增距离变体 | 214/290（73.8%） | 199/290（68.6%） | 109/560（19.5%） |

HIGHLIGHTS 每个 seed/策略选 5 个 clips；DISAGREEMENTS 每个 seed/有方向的策略对选 5 个，保留两个 leader 方向。
A/B/C 各 60 个模型；D 使用 58 个通过筛选的模型，FD/4209 和 BAL/4205 仍标记为失选，未使用临时模型替代。
因此 D 有 112/120 个可用有方向策略对，其他 arms 各有 120 个。所有可用策略对均有至少 5 个候选。

为匹配 DISAGREEMENTS 的 seed/leader/对手支持集，重复汇总对应 HIGHLIGHTS 选择后：A/B/C 比例不变；
D 原版为 417/560（74.5%），Huber 版为 391/560（69.8%），DISAGREEMENTS 仍为 109/560（19.5%）。
D 的差距不是由两项失选模型导致的分母不一致解释的。

## 解读范围

DISAGREEMENTS 在 A 中比两种 HIGHLIGHTS 更常覆盖 onset；B 中介于两种 HIGHLIGHTS 之间；
C 中与 HIGHLIGHTS 原版接近、低于 Huber 版；D 中明显更少覆盖 onset。
这是已观察到的选择位置差异，未做跨方法优越性或人类理解效果的推断。
ONSET 的片段按定义包含 onset，不能以这个指标本身证明 ONSET 更好。

沿用 P0 的 `target_onset_t`：A 为持续减速 onset；B/C/D 为被实际车道变化确认的、方向匹配的动作起点，
不是越过车道线或换道完成的时刻。保留 t=0 的 onset。
D 与这里的历史 A/B/C 同时存在奖励参数和几何设置差异，不能据此作纯几何干预的因果解释。

主指标沿用 P0 的公共窗口 `range(max(t-10,0), min(t+10,L-1))`。
若改用 DISAGREEMENTS 原生窗口，A/B/C/D 分别为 54.8%、40.8%、23.8%、19.5%；定性关系未改变。
仅检验真实 leader 轨迹的 onset，不把它套用到反事实 follower 分支上。
这是每 seed 的诊断扩展，不是预注册人类实验的跨 seed 池化摘要。

## 完整性与复现

- 238/238 个可用模型完成，各 36 个场景，共 8,568 个 episode；另 2 个模型明确失选。
- 最终 4,740 个所选 clips，472 个有方向策略对，全部窗口标记重新计算核验通过。
- 回放的位置、速度、车道、长度及动作核验通过；本次全量计算的数值 argmax 平局例外为 0。
- A–C 两种 HIGHLIGHTS 的 1,800 个所选片段身份与旧 P0 结果逐条一致。
- 输入 onset 表 SHA256 与本地原始资料一致；执行源码与保存的 v1/v2 快照一致。
- 相关测试、模型失选处理及聚合边界测试通过；详细测试结果记录于执行记录。

输出目录：`outputs/summary_comparison/clip_onset_coverage_20261027_v1/`。
`REPORT.md` 为完整汇总，`summary.csv` 包含逐策略/策略对结果，`paired_summary.csv` 和
`paired_by_seed.csv` 为匹配支持集的结果，`all_selections.csv` 为逐 clip 明细，
`validation.json` 为最终核验，`provenance/` 保存源码快照与修正版 onset 输入。

方法详见 [COVERAGE_PROTOCOL.md](COVERAGE_PROTOCOL.md)，作业与恢复记录详见
[coverage_submission.json](coverage_submission.json)。B/C 使用独立的修正版 onset 输入目录；
D 两个含失选模型的 seeds 使用仅修正失选状态识别的 v2 脚本，选片和覆盖率算法未改变。

```bash
PYTHONPATH=src .venv/bin/python scripts/aggregate_clip_onset_coverage.py \
  outputs/summary_comparison/clip_onset_coverage_20261027_v1
PYTHONPATH=src .venv/bin/python scripts/validate_clip_onset_coverage.py \
  outputs/summary_comparison/clip_onset_coverage_20261027_v1
```
