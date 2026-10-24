# Broad-distance retraining pilot v2

2026-10-24。用户授权重新训练。独立于 pilot v1、原 A/B/C 和 held-out v2；从头初始化，不继续微调旧模型。

## Frozen setup

- 3 个配对 seeds：5200–5202 × FD/BAL/SP，共 9 个新模型，各 100k steps。复用 seed 编号便于开发比较，独立输出前缀 `larger_target_gap_broad_v2`。不是新增 3 个独立于 v1 的统计重复。
- 原车道白车中心距离连续均匀抽样 D_o ∈ [40,150] m；不围绕 120–150 m 提高抽样权重。
- 先抽 D_o，再均匀抽 ΔD ∈ [30,min(80,195−D_o)] m，黄车中心距离 D_t=D_o+ΔD。白车较远时额外间距支持收窄，黄车始终更远，初始中心距离最大 195 m。实际观测可见性和裁剪由 preflight 验证。
- 这使额外间距分布依赖白车距离；与 v1 的差别不只是单一距离参数，不能把新旧模型差异全部归因于 D_o。该约束旨在保留观测范围和近距离原有额外间距支持，不根据训练结果挑参数。
- ego 初速 28 m/s，原生目标速度控制保持不变；白车慢速场景均匀 10–20 m/s；黄车 29 m/s。
- 每个独立洗牌的 10-reset block：60% 慢前车、20% 无白车、20% 同速白车。无白车仍保留黄车；各条件共享 reset-index 场景序列，实际 reset 数可随 episode 长度不同。
- 奖励权重、公共惩罚、gamma=0.95、DQN 超参数、20 s episode、5 Hz 决策、15 Hz 仿真均沿用 v1。不加动作门槛、不增加“等一等”奖励。Final fixed 100k checkpoint，不按差异大小或排序选择模型。

## Evaluation

主网格 D_o=45/60/90/120/135/150 m，ΔD=30/45 m，白车速度=10/14/18 m/s，共 36 场景。覆盖近、中、远距离，135 m 提供局部诊断；按距离报告，不只给网格总体均值。

每模型评估 original/no-front/open-target-lane/matched-speed-front，共 144 episodes。全部 9 模型共 1,296 episodes。保持记录首次左换道指令、实际车道变化、两条车道固定身份的前后间距、碰撞与未观察到事件。空目标车道仍是训练占用状态之外的控制。

结果接受立即换道、等待、减速、未换道和无组间差异，不因排序不符合预期更换模型或 seed。本轮仍是开发 pilot，不能冒充预先封存的验证实验。

## Implementation and verification

`run.py --profile broad-v2` 使用明确的 profile 参数切换训练支持、评估网格、输出名和 provenance。未指定 profile 时保留 v1 场景及默认行为。`collect.py --profile broad-v2` 校验每 variant 的 36 个唯一 exposure，拒绝把 smoke 当真实结果。

提交前检查：两版本场景回归、宽分布覆盖与条件间距上限、配对可重复性、36 场景全因子及实际可见性、脚本换道 preflight、200-step 端到端 smoke、PBS 语法。长训练只在 Katana 运行。

## Run

```bash
# Katana 项目根目录，代码更新后：
qsub scripts/katana_larger_target_gap_broad_v2.pbs

# 本地 smoke（不产生科学结果）：
.venv/bin/python experiments/larger_target_gap/run.py --profile broad-v2 --index 0 --smoke --execute

# 下载到本地项目根目录：
rsync -avP 'z5535967@kdm.restech.unsw.edu.au:/srv/scratch/z5535967/highway_transition_timing/outputs/larger_target_gap_broad_v2_*_100k_seed*' ./outputs/

# 汇总：
.venv/bin/python experiments/larger_target_gap/collect.py --profile broad-v2 \
  --root outputs --out outputs/larger_target_gap_broad_v2_summary
```

所有输出目录拒绝覆盖。运行/提交状态记入 `dev_log.md`；代码完成不等于训练已提交。
