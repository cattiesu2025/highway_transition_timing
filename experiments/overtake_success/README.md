# Arm B/C Overtake Success Experiment

状态：**正式实验仍处于设计阶段；奖励可触发性诊断已实现并完成本地 pilot，未提交集群任务。**
设计日期：2026-10-13。协议版本：draft-v5（完整超车须返回原车道并稳定领先 1 s）。成功定义 ID：`complete_overtake_return_v1`。

本实验研究：在现有 Arm B/C 中添加一次性超车成功奖励后，换道 onset 时间是否变化，以及 FD/BAL/SP 之间的 onset 差异是否保留、扩大、缩小或反转。同时用成功后终止的消融组，区分奖励和 episode 结束规则的影响。沿用现有道路结构，但本实验的主任务场景须通过超车可行性验证；必要时调整本实验的车辆位置采样，不覆盖原实验场景。

本文是独立的 development 实验设计，不覆盖现有 B/C 结果，不改变 sealed held-out v2。用户已授权并完成初步奖励可触发性 pilot，当前定义的方案、命令、结果与限制见 [PILOT_RETURN.md](PILOT_RETURN.md)；它不是正式四组实验。历史左车道保持判据的结果见 [PILOT.md](PILOT.md)。

设计修订、实现、验证与运行历史统一记录在本实验的 [dev_log.md](dev_log.md)；项目级日志仅保留日期索引和链接。

本实验的 Python、设计及日志位于本目录，PBS 统一放在本目录的 `scripts/` 子目录。先跑一次 B/BAL/S10 完整训练时，从仓库根目录执行：

```bash
qsub experiments/overtake_success/scripts/katana_overtake_B_BAL_S10.pbs
```

单任务脚本为 [katana_overtake_B_BAL_S10.pbs](scripts/katana_overtake_B_BAL_S10.pbs)，24-job 扩展为 [katana_overtake_trigger_pilot.pbs](scripts/katana_overtake_trigger_pilot.pbs)；不需要同时提交两者。PBS 使用提交时的仓库根目录作为工作目录。

如果依赖导入在共享环境读取处超时，使用 [本地环境重试](scripts/katana_overtake_local_retry.pbs)，打包和运行步骤见 [PILOT_RETURN.md](PILOT_RETURN.md) 的 Local-environment retry 小节。它保持实验参数和依赖版本，改为在计算节点 `/tmp` 读取环境快照。

## 1. Research Questions

四个 RQ 并列研究，不划分主要、次要或核心问题：分别考察成功奖励、成功终止、两者组合，以及 FD/BAL/SP 的 onset 差异。所有对应比较均按训练 seed 配对，完整报告方向、幅度和不确定性，不根据结果选择性保留某个 RQ。

| 问题 | 对照 | 解释目标 |
| --- | --- | --- |
| RQ1：添加超车奖励后，onset 时间如何变化？ | 同 arm/agent/seed 的 S10 − S00 | 单独添加奖励的方向和幅度 |
| RQ2：成功终止是否进一步改变 onset？ | S01 − S00；S11 − S10 | 无奖励和有奖励时的终止效应 |
| RQ3：奖励与终止组合如何影响 onset？ | S11 − S00；四组差分之差 | 组合总效应及交互 |
| RQ4：新设定下 FD/BAL/SP 的 onset 差异是否保留？ | 各 arm/组内 SP−FD、BAL−FD、SP−BAL，并与 S00 对应差值比较 | 时序差异保留、扩大、缩小或反转 |

预期而非结果：超车奖励可能促使策略更早换道以更早完成超车，但 onset 是否提前需要实测，不能由奖励定义直接推出。Arm B 的空目标车道和 Arm C 的占用目标车道可能产生不同响应；FD/BAL/SP 的响应幅度也可能不同。Termination only 可能诱发推迟成功。不得因预期不成立而排除 seed 或更换参数。超车成功率、碰撞率和完成时间用于解释 onset 变化及其代价，不替代四个 RQ 中的 onset 比较。

Arm B/C 都分别估计添加奖励后的 onset 变化，不要求 Arm B 无变化，也不要求保留原有 FD/BAL/SP 排序。置信区间包含零不代表没有影响或等效。本实验不复现 Arm A，不独立宣称已验证 A/C 的跨场景 reversal。

## 2. Existing Implementation and Scope

本设计基于项目本地实现，未引入新的外部算法或文献基线。

| 项目 | 当前实现 | 本实验处理 |
| --- | --- | --- |
| Arm B | 两车道，ego 从右车道出发，左车道空 | 保持 |
| Arm C | 同上，但左车道有车辆 | 保留占用和车距权衡；联合检查左车位置、速度及右前车状态，确保主任务存在可完成的超车轨迹 |
| 策略 | FD/BAL/SP，Double DQN | 保持 |
| 奖励 | 速度、前车距离、安全和动作项 | 按组增加一次性成功奖励 |
| 训练时限 | 20 s，5 Hz，共 100K policy steps | 保持 |
| 原评估时限 | 120 s | 保留，并增加 20 s 任务指标 |
| onset | 现有实际换道确认逻辑 | 保持，不替换成超车完成时刻 |
| 成功事件 | 无显式超车完成奖励或成功终止 | 新增 |
| checkpoint | maintained 实验选最新通过 gate 的 checkpoint | 本实验主分析固定最终 100K，gate 仅审计 |

代码依据：

- [B/C runner](../multilane_open_lane_change/run.py)：环境、场景、训练与评估。
- [B/C experiment](../multilane_open_lane_change/README.md)：当前运行协议。
- [Reward weights](../../src/highway_transition_timing/rewards.py)：FD/BAL/SP 权重。
- [Reward components](../../src/highway_transition_timing/highway_adapter.py)：基础奖励计算。
- [Onset extraction](../../src/highway_transition_timing/transitions.py)：换道 onset 和失败分类。
- [Checkpoint gate](../../src/highway_transition_timing/model_selection.py)：既有选择规则。
- [Project requirements](../../docs/user_requirements.md)：当前项目约束。

本地已有场景生成、训练和分析代码；已实现 pilot 成功跟踪器、短训练 runner 和诊断 collector。正式实验 runner、完整可行性审计及四个 RQ 分析仍待实现。沿用现有依赖，不需要新下载数据集。全文中的新增参数是本实验设计选择，不声称来自论文或已有结果。

### 2.1 Arm C placement and overtaking feasibility

用户约束：**左车道车辆可以影响换道选择和时机，但主任务不能因它挡住路径而根本无法超越初始慢前车。** “存在可行轨迹”与“每个学到的策略都能成功”不同；后者仍是实验结果。

当前代码将左车放在 `s_ego + target_lane_gap`：开发评估 gap 为 15/40/70 m，训练候选范围为 10–90 m；gap 是中心纵向差，不是车身净间距。左车初始和目标速度固定为 29 m/s，原始慢前车速度范围为 10–20 m/s。这些参数作为候选起点，不能单凭“左车比慢前车快”就保证 20 s 内可完成超车。

必须联合考虑 ego—慢前车距离和相对速度、左车位置和速度、车辆长度、两次横向移动时间以及成功所需的 5 m 净超越和返回原车道后的 1 s 保持窗口。既检查进入左车道的空间，也检查超越慢前车后能否在它前方安全返回右车道。无需超过左车，更不应把左车简单移到观测范围外来消除权衡。

**冻结场景前的可行性验证：**

1. 对每个 original 评估场景，以及预生成训练 manifest 中的慢前车超车任务场景，使用实际 simulator 和相同 action set 做确定性的参考动作序列搜索。参考规划器只判断几何/动力学可行性，不使用 FD/BAL/SP 权重或训练后 onset 结果。
2. 搜索需覆盖可执行的加减速、等待和向左换道序列。接受场景必须保存至少一条完整的动作 witness：从相同初态开始，无碰撞、无驶出道路，且按第 3 节判据在 **20 s 内**确认成功。每个 simulation step 检查实际车辆间净间隙；witness 与两辆交通车的最小车身间隙至少 2 m，避免仅靠擦边通过。该额外余量仅用于场景可行性审计，不更改策略成功奖励判据。
3. 未找到 witness 的场景标记为 `feasibility_unverified`，不能声称已证明物理上不可行，也不能进入主超车任务 manifest。可按固定候选顺序增加左车 gap，在原 10–90 m 范围内重新验证；若仍未通过，则按预定种子重采样同一场景层。不得根据任何实验组的表现选择位置。
4. 保留所有候选、调整/拒绝原因、参考动作、成功时间、最小净间隙与左右车辆轨迹。正式运行前冻结搜索规则、预算、位置选择顺序及 manifest hash；当前仅规定验收标准，尚未执行审计或验证任何具体 gap。
5. 尽量保留近/中/远左车位置层。若某层无法达到可行性标准，明确修订并冻结新场景表；不得静默把所有场景变成远距离左车，也不得通过放宽成功定义强行通过。

同一 arm/seed 的 S00/S10/S01/S11 和 FD/BAL/SP 必须共用审核后的场景。B 同样验证 20 s 主任务可达性，避免将距离过远造成的不可完成误归因于 C 的左车。训练中的 no-front、matched-speed、non-closing 等控制家族仍保留并单独标注，不能因控制场景不适合超车而强行移车；它们不属于上述“保证可超车”的主任务集合。评估控制从冻结的 original 场景按既有规则派生，不要求所有控制均在 20 s 完成。

若可行性筛选或位置调整改变训练分布，相应旧 baseline 不能直接复用：必须在相同新 manifest 上重训 S00，否则无法将 onset 变化归因于奖励。历史原始场景结果仅作单独参考。

## 3. Success Definition

### 3.1 Target identity

每次 reset 并完成场景替换后，保存初始右车道前车的对象身份为 `overtake_target`。整个 episode 固定该目标，不得在换道后用当前车道最近前车替换它。

Arm C 左车道车辆继续影响碰撞、前车距离和风险项，但不是必须超越的目标。**完整超车必须先从原来的右车道进入左车道、超过固定目标，再返回原来的右车道并稳定处于目标前方。** 留在左车道超过前车只算中间阶段，不算任务完成。不强制选择 FASTER 动作或速度必须增加：已有速度足以超过目标时也可以完成超车。

有初始前车的 matched-speed、non-closing 等场景仍使用同一判据；无初始前车时 `success_applicable=false`，不发奖励、不因成功终止。目标不存在不能当作已经超越。

### 3.2 Geometry and persistence

沿道路纵向坐标定义净超越距离：

```text
clearance = ego_rear_s - target_front_s
```

前后端位置由车辆实际长度和朝向在道路纵向的投影确定，不能把车中心差直接当作净距离。当前直线路段上可直接投影到公共道路方向。

按以下顺序判定，不能只检查最终车道或某个动作命令：

1. **左换道完成**：ego 实际 lane index 为初始车道左侧车道，且中心距左车道中心线 ≤ **0.5 m**。保存首次完成时间。
2. **左侧超越目标**：已经完成上述左换道，实际仍在左车道时达到 `clearance >= 5.0 m`。保存首次达到时间；此阶段不要求额外停留 1 s。
3. **返回原车道并领先**：完成前两阶段后，ego 实际 lane index 回到 reset 时的原车道，中心距原车道中心线 ≤ **0.5 m**，同时 `clearance >= 5.0 m`。
4. **稳定确认**：第 3 步的回归和领先状态连续保持 **1.0 s**，全过程无碰撞、未驶出道路，目标身份有效。此时才记录一次完整超车成功并触发奖励/成功终止。

第 3 步状态被打断时重置其持续窗口，已完成的左换道和左侧超越阶段保留；碰撞或驶出道路则不允许本 episode 再获得成功。仅从右侧直接超过、尚未超越就返回右侧，或超越后始终留在左车道，都不满足完整成功。

在每个 simulation step 检查几何及碰撞状态，policy step 末处理奖励和终止。若实现只能检查 policy 边界，必须先修订协议，不能将离散抽样宣称为全程保持。

`success_time` 是首次满足完整持续窗口后、可交付 reward 的 policy-step 结束时刻；同时保存几何条件首次成立时间和内部确认时间。使用仿真时间差，不把 5 次观测误写成 5 个完整的 0.2 s 区间。若同一个 policy step 内成功确认后又发生碰撞，该 step 按失败处理，不发奖励。

5 m、0.5 m、1 s 是本协议固定参数。任何修改都需新协议版本并说明原因，不能按干预 onset 效应或 FD/SP 差值调节。

### 3.3 State machine

```text
not_applicable                                      # reset 时无目标
pending -> left_completed -> passed_in_left
                                  |
                                  v
                         returned_ahead -> success   # 原车道稳定领先 1 s
                                  |
                                  +-> passed_in_left # 窗口中断，重新计时
```

`success` 一旦确认即锁存。Reward only 和 Baseline 后续再次超越也不产生第二次事件。成功后碰撞记录为 `post_success_collision`，不抹去历史成功事件。必须区分“首次成功”和“整个固定窗口内无碰撞地完成任务”。

## 4. Factorial Design

| ID | 名称 | β | 成功终止 | 与核心 baseline 的区别和意义 |
| --- | --- | ---: | --- | --- |
| S00 | Baseline | 0 | 否 | 现有任务规则，增加被动成功日志；核心比较基线 |
| S10 | Reward only | 30 | 否 | 只增加任务奖励 |
| S01 | Termination only | 0 | 是 | 只增加任务终点 |
| S11 | Reward + termination | 30 | 是 | 用户提出的完整新设定 |

每组均为 Arm B/C × FD/BAL/SP。所有组都运行相同成功检测器；S00 中检测器不能影响 observation、reward、随机数流或环境转移。

四组使用同样的场景生成规则和基础奖励。其他奖励敏感性、fixed-sum、risk-penalty 实验不并入本实验，避免同时改变多个因素。

## 5. Reward and Termination

### 5.1 Reward

```text
reward_total = reward_base + beta * first_success_this_step
```

`reward_base` 完全复用当前 B/C 算式，不重新归一化，不在成功步清除原有安全或动作惩罚。所有 agent 的 β 相同。

| 项目 | FD | BAL | SP |
| --- | ---: | ---: | ---: |
| speed weight | 0.45 | 0.70 | 1.00 |
| front-distance weight | 1.00 | 0.70 | 0.25 |
| collision penalty | 2.00 | 2.00 | 2.00 |
| collision-risk penalty | 3.00 | 3.00 | 3.00 |
| lane-change action penalty | 0.20 | 0.20 | 0.20 |
| slow-down action penalty | 0.20 | 0.20 | 0.20 |
| right-lane reward | 0 | 0 | 0 |

β=30 的量级依据：基础正奖励上限分别为 1.45、1.40、1.25；γ=0.95 时无限时域正收益上限为 29、28、25。小额 terminal bonus 可能不足以补偿提前结束损失的后续收益。30 是预设起点，不是最优性或安全保证；它可能压过偏好差异，也可能增加冒险行为。

仅在另行开展敏感性扩展时，使用预设 β∈{15,30,60}，对 S10/S11 全部报告。主实验固定 30；不得从扩展中挑分离最大的 β 替换主结果。

### 5.2 Terminal priority

每个 policy step 先计算仿真结果，再按以下优先级处理：

1. 碰撞：`terminated=true`，reason=`collision`，成功奖励为零。
2. 其他底层真实终止：保留终止状态和具体原因，不覆写为成功；成功奖励为零。
3. 首次成功且本组开启成功终止：`terminated=true`，`truncated=false`，reason=`overtake_success`。
4. 达到时间上限且没有真实终止：`terminated=false`，`truncated=true`，reason=`duration`。
5. 否则继续。

碰撞、成功、时间上限同一步发生时碰撞优先；成功与时间上限同一步发生时，开启成功终止的组以成功结束。未开启成功终止的组可在 horizon 步获得首次奖励，同时以 duration 截断。

保持底层 off-road 等配置，不借本实验额外引入一种失败终止；但驶出道路不能满足成功条件。实现时验证实际底层终止类型并写入 manifest。

成功是真实任务终止，Bellman target 不 bootstrap。纯 time-limit truncation 沿用当前算法的超时处理并检查 replay buffer 的 timeout 标记，防止误把成功当作可 bootstrap 的 timeout。

## 6. Training Protocol

### 6.1 Fixed configuration

| 参数 | 设定 |
| --- | --- |
| Algorithm / policy | 当前 SB3-compatible Double DQN / MlpPolicy |
| Training budget | 每个模型 100,000 policy steps |
| Training horizon | 20 s |
| Policy / simulation frequency | 5 / 15 Hz |
| γ / learning rate | 0.95 / 0.0005 |
| Replay buffer / batch | 50,000 / 32 |
| Learning starts | 1,000 steps |
| Train frequency / gradient steps / n-step | 1 / 1 / 1 |
| Target update | 250 steps |
| Exploration | initial 1.0，final 0.05，fraction 0.25 |
| Checkpoint interval | 5,000 steps |
| Observation / action / network | 当前 B/C 配置；记录实际解析值和依赖版本 |
| Scenario profile | 当前 stratified，near-matched-speed delta=2.0 |
| Arm C target traffic | 29 m/s；原 gap 范围作为候选，经第 2.1 节审计后冻结本实验场景 |

以现有 stratified scene generator 为候选生成器，按第 2.1 节审核慢前车任务，保留家族配额及控制场景，冻结实际训练 manifest；不要用 `no_front_train_fraction=0.2` 代替分层生成器的实际场景比例。manifest 按最多 100K 次 reset 预备或使用等价的确定性审计缓存，确保任何实际用到的任务场景均有 witness；不得耗尽后回退为未验证采样。

### 6.2 Seeds and pairing

- 正式实验：B 4100–4119，C 4200–4219，每 arm 20 seeds。
- 同一 arm/seed 内四组和三种奖励条件使用一致的初始化种子及按 reset_index 索引的场景计划。
- 相同 policy-step 预算下，成功终止会改变 episode 数、接触到的场景数量及探索阶段覆盖。这是该干预的总效应之一，不能宣称训练经历逐步完全匹配。
- 记录每个模型 reset 数、各场景家族数量、episode 长度和成功/碰撞频率。配对的是种子和场景序列规则，不是每一条 transition。
- B/C seed 不构成同一场景的跨 arm 配对，统计时分 arm 分析。

### 6.3 Pilot and protocol freeze

先进行本地代码 smoke，再用正式 seeds 之外的 B 4190–4191、C 4290–4291 做完整 100K pilot：2 arms × 2 seeds × 3 agents × 4 groups = 48 models。pilot 不进入正式推断。

pilot 用于验证奖励是否接入、是否出现成功样本、episode 长度和 replay 终止标记是否正确，以及资源量级。不得用 pilot 的 onset 排序筛选 β、场景或 seeds。若成功几乎无法达到，应报告失败并解释是几何条件、20 s 可达性、探索还是奖励问题；必要时形成新版本，不悄悄延长训练或时限。

在正式运行前冻结 protocol、job manifest、development scene manifests 和源码 hash。协议参数修改均保留版本记录。若需未来确认性泛化实验，应另建未用于开发的新评估集，不调用既有 sealed held-out v2。

## 7. Checkpoint Policy and Baseline Reuse

主分析统一使用最终 **100K checkpoint**，所有模型同训练年龄，不因 gate 失败而丢弃。这样可以避免成功目标改变 control specificity 后，组间差异混入 checkpoint 选择效应。

这是本独立实验相对 maintained 协议的明确变更；不能把结果标成 maintained selected-policy 结果。现有 gate 在固定 120 s 延续评估上作附加审计，报告各项通过率，不用于主模型替换。

旧 baseline 仅在满足以下全部条件时复用：

1. 最终 step_100000 checkpoint 实际存在，不能直接假定 `*_main.zip` 是最终模型。
2. 基础配置、代码行为、依赖、训练预算和 seed 与新实验兼容且可核验。
3. 记录源文件路径和 checksum；只读使用，禁止覆写。
4. 用新被动检测器重新评估，并完成 detector 开关不改变 rollout 的回归检查。

若不满足，则重新训练对应 S00。旧 gate-selected 模型只可列为历史参考，不能与新组的 final-100K 混入主对照。默认资源预算按四组全部新训计算。

## 8. Evaluation Protocol

所有模型 deterministic evaluation，使用完全匹配、预先导出的本实验 development scenes。以当前 development generator 的 36 exposures/variant 为候选规模；original 须通过第 2.1 节审计，控制从冻结场景派生。导出时审计 B/C 场景数、车辆初始状态及 checksum。若可行性要求导致规模或分层改变，正式运行前修订 manifest 和第 13 节数量估计，不沿用未经验证的旧场景表。

### 8.1 Variants

| Variant | B | C | 用途 |
| --- | --- | --- | --- |
| original | 是 | 是 | 主任务和 timing 对比 |
| no-front | 是 | 是 | 无目标控制；超车完成指标 N/A |
| matched-speed-front | 是 | 是 | 追近需求控制 |
| far-front | 是 | 是 | 距离控制 |
| open-target-lane | 否 | 是 | 占用目标车道机制控制 |

不得把 no-front 的 N/A 记为成功率 0，也不能并入原始场景的成功分母。各控制独立报告，不通过它们筛选出“表现好”的 original 样本。

### 8.2 Two evaluation modes

**Task evaluation：**使用每个组自己的成功奖励及成功终止设定，分别以 20 s 和 120 s 为最大时限评估。20 s 指标对应训练任务；120 s 描述较长机会窗口下的任务完成。S00/S10 记录首次成功后继续到碰撞/时限，S01/S11 在成功后结束。

**Fixed-horizon continuation evaluation：**所有模型统一关闭成功终止，保留一次性事件检测，运行到碰撞、其他底层终止或 120 s。用于主 onset 分析、固定观察窗口内的碰撞和超车后行为。保持各组 bonus 日志，但 reward 不更新模型，不影响 deterministic action。

两个模式必须从同一个初始场景独立 reset，不能在成功后从自动 reset 的新 episode 接着计时。相同模型两模式在首次成功/终止前应具有一致轨迹。

仅用 task evaluation 的碰撞率不能宣称 S01/S11 更安全，因为提前成功会缩短暴露时间。安全比较以统一 continuation 窗口为依据，并附实际暴露时间。

## 9. Endpoints and Statistical Analysis

### 9.1 Outcomes

| 指标 | 定义与用途 |
| --- | --- |
| Original physical lane-change onset | 现有确认算法，记录 onset 和 confirmation；不能换成成功时刻 |
| First overtake success | 按第 3 节首次确认的事件 |
| Success probability by 20/120 s | 所有 applicable episodes 为分母，碰撞前未成功计未完成 |
| Success time | 记录事件时间；仅成功样本的中位数是描述性统计 |
| Collision / other terminal / timeout | 全部分开统计；另记录成功前/后碰撞 |
| Collision-free completion | continuation 中成功且整个观察窗口无碰撞 |
| Lane change without overtake | 有确认换道但窗口内无超车成功 |
| No confirmed onset | 保留失败原因和未观察到 onset 的样本 |
| Extra lane changes / right return | continuation 中成功后的行为诊断 |
| Reward decomposition | base、success、total 分开记录，不跨组按 total return 排名 |

### 9.2 All-episode timing estimand and RQ contrasts

主分析使用 original、120 s continuation，τ=120 s。对每个 episode 定义：

```text
L_onset = confirmed onset latency, if a valid onset exists by tau
          tau, otherwise
```

采用当前算法的有效 onset 定义；onset 后碰撞仍保留已发生 onset，同时计入碰撞率。未发生有效 onset 的碰撞/其他终止取 τ。这是“窗口内未完成目标事件的时间负担”复合指标，不是对碰撞后潜在 onset 时间的插补，也不是声称失败者真的在 120 s 换道。

每个 arm/seed/agent/group 先对全部固定 exposures 求 `L_onset` 均值，再在同 arm/seed/agent 内计算 RQ1 的干预效应：

```text
Delta_reward = mean(L_onset | S10) - mean(L_onset | S00)
```

分别报告 B/C × FD/BAL/SP 共六个效应，不先合并不同 agent。负值表示添加奖励后 onset 更早和/或更容易发生，正值表示更晚和/或更少发生。必须同时报告 onset 发生率和实际观测时间，不能把复合指标单独解释为纯时间变化。

为直接展示 onset 的秒数变化，同时报告各组 observed-onset median，以及同 seed、同 exposure 在 S10/S00 均有有效 onset 时的配对时间差 `t_onset(S10) - t_onset(S00)`，先在 seed 内汇总再跨 seed 报告。列出双方均发生、仅 baseline 发生、仅 reward 组发生、双方均未发生的 exposure 数量；配对时间差只描述双方均发生 onset 的子集，不能替代全部样本结果。不存在 onset 或有效配对时为 N/A，不填零、不静默删 seed。只有发生率和配对覆盖支持时才作纯 timing 解释。

四个 RQ 使用一致的指标和报告规则：RQ1 报告六个 `Delta_reward`；RQ2 报告 S01−S00、S11−S10；RQ3 报告 S11−S00 和四组交互；RQ4 报告组内条件差异及其变化。每项均给出全部样本时间负担、对应实际 onset 配对时间差、发生率、配对覆盖和不确定性。

**RQ4 正式分析：**每个 arm/group/seed 计算 SP−FD、BAL−FD、SP−BAL 的 `L_onset` 差值，并报告对应双方都有 onset 的 exposure 配对秒数差、各自发生率和完整分母。再计算这些条件差值相对 S00 的变化，例如 `(SP−FD)_S10 − (SP−FD)_S00`。三种条件比较均报告，不仅展示最分离的一对。用方向、幅度与区间判断时序差异是否保留、扩大、缩小或可能反转；不以区间包含零断言等效，也不要求必须保留原排序。

### 9.3 Factorial effects

对任一 seed-level 指标 Y，记 Y00、Y10、Y01、Y11 为四组结果：

```text
Reward effect      = ((Y10 - Y00) + (Y11 - Y01)) / 2
Termination effect = ((Y01 - Y00) + (Y11 - Y10)) / 2
Interaction        = Y11 - Y10 - Y01 + Y00
```

按 arm/agent 分别计算 onset 时间负担、任务完成和安全的因子效应。跨终止条件平均的 Reward effect 与 RQ1 的 S10−S00 回答不同问题，应分别报告；有交互时尤其不能将两者混称。不同 agent 的 Delta_reward 差值用于描述响应差异。主 bonus 下的因子效应是本训练协议的总效应，包含因 episode 长度改变带来的数据分布变化。

### 9.4 Uncertainty, failures and interpretation

- 训练 seed 是独立重复单位；36 exposures 不是 36 个独立训练样本。
- 每个 arm 对 seed block 做 10,000 次配对 bootstrap，固定统计 RNG seed=20261013；同一次抽样保留所有 agent/group/exposure 的配对关系，给出 95% percentile intervals。
- 本实验为 development 奖励/终止干预分析，报告效应量和区间，不以大量逐格显著性检验宣布确认性发现；区间不作跨所有比较的同时覆盖保证。
- 事件曲线采用累积发生比例，碰撞/其他真实终止为竞争结局，不作为独立无信息删失。固定窗口 timeout 单独报告。
- crashed-before-success、timeout、no-onset 均保留分母；仅 infra failure 可同 seed/config 重试并保留日志。训练数值发散为算法失败，不换 seed 补成功结果。
- 缺失模型时报告缺失数量与原因；涉及配对效应时列出完整配对数量，并对缺失导致的选择偏差作限制说明。不得静默丢弃或将 infra 缺失当作驾驶失败。
- 任一 arm/agent 的干预效应区间包含零，只说明当前证据无法明确变化方向，不证明没有影响。是否提前或推迟必须结合实际 onset 时间、发生率、配对覆盖及失败率判断，不以 Arm C 是否保留 SP−FD<0 作为主结论。

## 10. Logging and Reproducibility

拟定输出目录：`outputs/overtake_success_v1/arm{B,C}/S{00,10,01,11}/seed{seed}/`。一个 run 包含 FD/BAL/SP；已有目录默认拒绝覆盖。generated outputs 不进入 source control。

| 文件 | 必要内容 |
| --- | --- |
| protocol.json / config.json | 版本、成功几何及持续规则、β、终止模式、实际基础参数 |
| source_manifest.json | 源码及场景 hash、依赖版本、模型 checksum、复用来源 |
| training_scenarios.csv | reset_index、完整初始场景、场景家族、目标身份 |
| training_episodes.csv | steps、结束原因、成功次数、base/bonus return、碰撞 |
| checkpoints/ | 各 checkpoint 和 final 100K 文件 |
| evaluation_steps.csv.gz | pre/post 时间状态、target 位置、clearance、lane offset、持续窗口、奖励分解、terminated/truncated |
| evaluation_episodes.csv | applicability、首次成功/换道时间、碰撞时间、结束原因、总观察时间、成功后碰撞 |
| gate_audit.csv | final 100K 原 gate 审计结果，不用于替换模型 |
| seed_metrics.csv / contrasts.csv | seed 级指标、配对效应、分母与缺失 |
| complete.json | 所有规定产物验证后最后写入 |

每条 evaluation 记录包含 arm、seed、agent、group、variant、mode、horizon、exposure_id、protocol_version。明确时间单位为秒，原始 policy-step index 单独保留。

成功不是失败：collector 必须识别 `overtake_success`。不得直接沿用只理解 collision/duration 的终止映射。任务完成结果与现有 onset outcome 分列，不能把二者压成一个布尔 success。

## 11. Implementation Boundaries and Validation

后续建议增加独立 runner、成功检测组件、job manifest、collector 和 PBS；共享模块只在必要时提供默认关闭的扩展点。不能通过改全局默认值影响旧实验。

实现后的必要验证：

1. 几何与顺序：仅换道但未超越、超越后未返回、右侧直接超过、超越前返回、车中心超过但车尾未超过、刚跨 lane boundary、恰好 5 m、0.5 m 边界。
2. 时间与身份：持续不足/达到 1 s、中断重计、目标车不随最近前车变化、reset 清空状态。
3. 事件语义：无目标 N/A、一次性奖励、超车后再被反超不得重发、成功后碰撞单独记录。
4. 终止优先级：碰撞与成功同 step、成功与 horizon 同 step、timeout bootstrap、真正成功不 bootstrap。
5. S00 被动检测回归：固定 actions/seed 下 obs、reward、flags 与原环境一致，检测不消费 RNG。
6. 模式一致性：task 与 continuation 的首次事件前轨迹一致；训练和评估使用同一个 detector。
7. 保存/加载与 collector：短训练完成后重载模型、跑全部 variants、检查分母/缺失和成功终止分类。
8. 人工可控动作 rollout：至少一条 B 成功、一条 C 成功、一条碰撞和一条 timeout，核验几何日志。

本地验证只检查接线和事件正确性，不作为已学会超车的证据。长训练在 Katana 上运行。

## 12. Reporting Plan

计划产物：

1. 结果表：按 B/C × FD/BAL/SP 报告四组的 onset 时间、发生率、L_onset，以及 RQ1–RQ4 对应的配对效应；附 seed 数、observed/total 和双方均有 onset 的配对覆盖。
2. 效应图：按 RQ1–RQ4 展示奖励、终止、组合/交互及 FD/BAL/SP 差异的效应与区间，区分全部样本时间负担和双方均发生 onset 的实际秒数差。
3. 场景审计表：左车位置分层、可行性 witness 和候选场景调整记录。
4. 任务与安全表/曲线：20/120 s 内成功和碰撞的累积发生率，用于解释 onset 变化是否伴随完成率或安全变化。
5. 训练诊断：episode 长度、reset 数、基础/成功奖励、成功事件随训练进度变化。
6. 定性轨迹：按预设规则展示成功、失败、超车后碰撞等案例，不能只挑最能分离偏好的案例。

若后续绘图，沿用项目近期新增实验的 R 工作流；本阶段不生成图或模拟结果。

## 13. Resource Estimate and Execution Stages

| 阶段 | 模型数 | 训练 policy steps | 说明 |
| --- | ---: | ---: | --- |
| 本地 smoke | 少量短 runs | 不计正式预算 | 接线、终止、保存加载 |
| 独立 pilot | 48 | 4.8M | B/C 各 2 个非正式 seeds |
| 正式四组 | 480 | 48M | 2 arms × 20 seeds × 3 agents × 4 groups |
| 若全部 S00 可复用 | 新训 360 | 新训 36M | 须通过第 7 节完整审计 |
| 可选 β=15/60 扩展 | 480 | 48M | 仅 S10/S11，2 个额外 β；不属主实验 |

正式每种 horizon/mode：B 240 models × 4 variants × 36 scenes = 34,560 episodes；C 240 × 5 × 36 = 43,200；合计 77,760。20 s task、120 s task、120 s continuation 共 233,280 个逻辑评估 episodes。只有规则完全相同的评估才可去重并记录复用关系，不可混用不同 horizon。

训练小时数、CPU/GPU 内存和日志空间尚未测量；用 pilot 实测后估算，不凭空给出硬件耗时。

执行顺序：协议冻结 → 独立实现 → 本地验证 → pilot → 固定版本正式训练 → 全模型评估 → seed 级统计与报告。若 pilot 促成协议修改，先记录新版本再启动正式实验。

当前已完成设计及独立可触发性 pilot；已提供 runnable pilot 命令和 Katana PBS，但尚未提交集群任务或运行正式实验。早期 pilot 使用左车道保持判据，其 B/C 成功率不能视为本版本的完整超车成功率。用户明确完整超车须返回原车道后，已按本版本更新 detector 并重新评估；当前结果见 [PILOT_RETURN.md](PILOT_RETURN.md)，历史结果保留于 PILOT.md。
