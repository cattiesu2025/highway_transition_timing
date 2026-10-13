# Development Log — Overtake Success Experiment

本文件记录本实验的设计决策、实现、验证和运行历史，按日期追加。
当前协议见 [README.md](README.md)；项目级日志只保留本实验的日期索引和链接，不重复详细记录。

## 2026-10-13 — 实验设计与日志建立

### 已完成

- 创建独立实验目录及完整设计文档，当前协议为 draft-v4。
- 设计 Arm B/C × FD/BAL/SP 的四组对照：S00 baseline、S10 成功奖励、S01 成功终止、S11 奖励与终止组合。
- 明确四个 RQ 并列研究，不分主要、次要或核心：奖励对 onset 的影响、终止对 onset 的影响、两者组合与交互、FD/BAL/SP 的 onset 差异是否保留或改变。
- 规定超车目标为初始右车道慢前车，固定目标身份；实际进入左车道、车尾领先目标车头至少 5 m、距车道中心不超过 0.5 m，并连续保持 1 s。无需返回右车道或超过 Arm C 左车道车辆。
- 拟定一次性成功奖励 β=30，明确碰撞优先、成功真实终止、时间上限截断和一次性发奖规则。参数尚未经过训练验证。
- 加入 Arm C 场景可行性约束：左车可以影响时机，但不能使主超车任务无法完成。正式场景须保存实际 simulator 中 20 s 内无碰撞成功的参考动作轨迹；必要时按固定规则调整位置或重采样。B 的主任务也检查可达性。
- 明确四组共用审核后的场景；若训练分布改变，相关 S00 必须重训，不能直接使用旧 baseline 归因奖励效应。
- 拟定 final-100K checkpoint 对照、task/continuation 两类评估、seed 配对统计、完整失败分母、日志结构、实现验证清单和资源预算。

### 设计修订经过

| 版本 | 修订内容 |
| --- | --- |
| draft-v1 | 建立四组设计、成功判据、训练评估及统计协议 |
| draft-v2 | 将研究目标明确为添加奖励后的 onset 变化，调整前后配对比较 |
| draft-v3 | 恢复 FD/BAL/SP 条件间差异为正式研究问题，并加入超车可行性约束 |
| draft-v4 | 根据用户要求，四个 RQ 并列，统一分析和报告安排 |

早期版本曾将 RQ4 降为辅助分析，该安排已被用户纠正并由 draft-v4 取代；后续以当前 README 为准，不沿用旧优先级。

### 验证与当前状态

- 已检查设计文档本地链接、代码围栏配对及模型/评估数量计算。
- 未实现 runner、成功检测器、可行性搜索或 collector；未实际验证任何候选场景的可行性。
- 未训练、提交 Katana 任务或产生新实验结果；本次文档检查不构成行为证据。
- 现有 maintained 输出和 sealed held-out v2 未修改。

### 后续工作

按当前设计实施可行性审计与实验组件、本地验证和独立 pilot；正式训练前冻结搜索规则、场景 manifest、配置和源码版本。后续设计变更、检查结果与运行记录追加到本文件。

## 2026-10-13 — 奖励可触发性 pilot

用户要求先验证成功奖励是否能被探索到，授权运行本次诊断。详细方法、可复现命令及结果见 [PILOT.md](PILOT.md)。

- 实现 `pilot.py`：按 simulation step 被动跟踪初始慢前车、严格成功判据、训练 bonus、episode 日志与实际 replay 采样审计；不改共享实验默认行为。
- 实现 `collect_pilot.py` 和 24-job Katana PBS（B/C × 2 seeds × FD/BAL/SP × S00/S10）。本机无 qsub，未提交；正式四组实验也未运行。
- 重跑 B4100/4101、C4200/4201 selected models，全部 FD/BAL/SP 和每模型 36 个 development scenes，共 432 episodes。B 严格成功 216/216（均在 20 s 内）；C 严格成功 0/216，但全部曾进入左车道且净超过初始前车 5 m。未打开 held-out 数据。
- 核验 C4200/FD/M0000：8.8 s 已净超过约 5.99 m，随后返回右车道，未达到左车道中心附近的 1 s 持续要求。因此严格判据未触发不能简单解释为不会超车；尚未修改任务定义。
- 从头运行 BAL 的 B4190/C4290 × S00/S10，每模型 2K steps。保留 100K 的 epsilon 衰减日程前缀，learning starts=1000。B 每组 2 次成功，首次 step575，2 条成功经验被采样共 38 次；C 每组 1 次成功，首次 step779，被采样 23 次。全部成功发生于随机 warm-up，随后实际进入训练更新；不构成已稳定学会超车的证据。
- 初次 `short_*` 接线运行使用压缩到 2K 的探索日程，发现后改为固定 25K 衰减并重跑 `prefix_*`；旧产物保留且 collector 明确排除，不混入有效结果。
- 验证：26 项相关测试通过、四个有效短训练模型 reload/predict/step 成功、PBS 语法及 whitespace 检查通过。输出在 `outputs/overtake_success_pilot/`，不纳入版本控制。
- 结论边界：已证明所测 B/C 短探索可触发奖励，尚未证明多 seed 稳定学习、所有场景可行或奖励改变了 onset。正式长训练前需先明确是否允许确认窗口内返回右车道。

## 2026-10-13 — 完整超车定义确认与重新运行

用户明确超车必须“左换道、超过原车道前车、再回原车道”，要求写入设计并重跑。协议更新为 draft-v5，定义 ID 为 `complete_overtake_return_v1`。本条取代前条中尚待确定的成功语义；旧结果保留为历史记录。

- 成功 detector 改为有序阶段：实际左换道完成 → 左侧净超越至少 5 m → 返回初始车道中心附近（≤0.5 m） → 保持净领先和原车道位置 1 s，全程无碰撞/驶出道路。不强制 FASTER 动作。
- 加入各阶段时间及定义 ID 日志；collector 拒绝混合旧定义，PBS 切换独立输出目录。输出在 `outputs/overtake_success_return_pilot/`，未覆盖历史数据。
- 重跑相同旧 selected models 和 432 个 development episodes：B 的 FD/BAL/SP 均 0/72 在 120 s 内完整成功；C 均 72/72 在 120 s 内成功，20 s 内分别 64/72、64/72、69/72。C 总计 197/216 在 20 s 内、216/216 在 120 s 内。B 全部曾完成左侧超越，但没有达到返回原车道并稳定领先的完整成功条件。
- 重跑 B4190/C4290 × BAL × S00/S10 各 2K steps，仍用正式探索日程前缀。B 每组成功 1 次，首次 step781，被采样 27 次；C 每组 2 次，首次 step485，被采样 49 次。全部独立成功 transitions 都实际被训练采样；不宣称已稳定学习或已改变 onset。
- 29 项相关测试通过，四个短训练模型 reload/predict/step 检查通过，汇总、文档链接、PBS 语法及 whitespace 检查通过。所有重跑轨迹均未记录 off-road。
- 当前完整结果见 [PILOT_RETURN.md](PILOT_RETURN.md)。未运行正式四组实验、完整可行性搜索或提交 Katana 任务；项目级 dev_log 继续只保留指向本日志的索引。

## 2026-10-13 — 单次完整训练交付

- 用户明确本实验先保持换道惩罚不变，完整实验之后再考虑调整；先准备 B/BAL/S10/seed4190 的 100K 从头训练，成功奖励 +30，换道动作惩罚 −0.2。
- 用户要求上传 GitHub，由用户在 Katana 自行运行。新增单任务脚本 `scripts/katana_overtake_B_BAL_S10.pbs`，避免误提交整个 24-job array；运行说明见 PILOT_RETURN.md。
- 脚本输出训练日志、成功事件及 replay 统计和最终模型；不把训练探索比例当作训练后评估成功率。deterministic evaluation 尚不包含在该单任务脚本中。
- 本次上传范围仅为本实验目录、两个相关 PBS 和本实验测试。模型及生成输出保持忽略；项目级 `docs/dev_log.md` 位于原有 gitignore 的 `docs/` 内，继续本地保留链接索引，不将整个历史文档强制加入提交。

## 2026-10-13 — PBS 集中管理

- 按用户建议，将两个 `katana_overtake_*.pbs` 从项目级 `scripts/` 移至 `experiments/overtake_success/`，与 runner、设计和日志放在一起。
- 更新 README、pilot 文档中的提交命令和相对链接。上条日志的 `scripts/` 路径为历史位置，以本条为准。
- PBS 内容及训练参数不变；仍从仓库根目录提交。检查两个 PBS 的 shell 语法、文档链接及 diff whitespace。

## 2026-10-13 — 实验内 scripts 子目录

- 按用户要求，将两个 PBS 进一步整理到 `experiments/overtake_success/scripts/`，更新所有当前提交命令和文档链接。
- 历史记录保留原位置；当前从仓库根目录执行 `qsub experiments/overtake_success/scripts/katana_overtake_B_BAL_S10.pbs`。训练参数和脚本内容不变。
- 两个 PBS 的 shell 语法、文档链接及 whitespace 检查通过。

## 2026-10-14 — Katana 启动与进度诊断

- 用户报告 job9508396 运行约 2 h 15 min，累计 CPU 3 s、cpupercent=0、mem 约 253 MiB；输出目录仅有 manifest.json，PBS `.o` 尚不可见。说明执行到过 manifest 写入，但不能据此确定具体阻塞函数，根因尚未证实。
- 两个 PBS 接入已有 `scripts/katana_headless_setup.sh` 的有界依赖导入预检、私有 fontconfig/cache；stdout/stderr 直接输出到 scratch 下按 job ID 命名的实时日志，并打印 shell 阶段。
- runner 增加阶段 progress.json、每 1,000 步/约 30 s 的训练心跳与 partial episode CSV；120 s 无阶段/心跳更新时输出 Python 栈。所有诊断不改变动作、奖励、seed、训练预算或探索日程。
- 单任务 PBS 支持 OVERTAKE_RUN_NAME，为后续重试保留独立目录，仍拒绝覆盖。未取消、重启或登录用户当前 Katana 作业；新日志只对后续启动的程序生效。
- 34 项相关测试通过；本地 1,100 步 smoke 覆盖依赖导入、环境/模型创建、真实梯度更新、心跳、保存和完成。PBS shell 语法与 whitespace 检查通过。诊断改动不是对集群阻塞根因的确认或已解决声明。
