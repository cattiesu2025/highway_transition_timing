# Complete-overtake Pilot — Return to Original Lane

日期：2026-10-13。成功定义：`complete_overtake_return_v1`，对应设计文档 draft-v5。

## Definition and Scope

用户明确“超车”指完整过程，本轮采用以下有序判据：

1. 实际进入左车道，车中心距左车道中心线 ≤0.5 m。
2. 此后在左车道内，ego 车尾净超过 reset 时原车道前车的车头至少 5 m。
3. 返回 reset 时的原车道，车中心距该车道中心线 ≤0.5 m，并保持上述净领先。
4. 回归和领先状态连续维持 1 s；全程无碰撞、未驶出道路。

不强制 FASTER 动作，也不要求超过左车道交通车。只超过但不返回，不计完整超车成功。检测每个 simulation step，奖励和终止在确认后的 policy step 末处理。目标身份固定。

本轮重跑与前次相同的模型和场景，不根据新定义筛选成功样本：

- 旧策略：B4100/4101、C4200/4201，各 FD/BAL/SP × 36 original development scenes，最多 120 s，首次成功或环境终止时停止，共 432 episodes。
- 从头短训练：B4190、C4290，各 BAL × S00/S10 × 2K steps。保留 100K 训练的 25K-step epsilon 衰减日程，learning starts=1000，不加课程、不做场景可行性筛选、不加 dense shaping。
- 输出使用全新目录 `outputs/overtake_success_return_pilot/`，不覆盖历史 pilot。manifest、episode rows 和 training summary 显式标注定义 ID，collector 拒绝混合旧定义。

本轮仍为可触发性诊断，不是四个 RQ 的正式训练实验。未使用 sealed held-out 数据，未提交 Katana 任务。

## Existing-policy Results

每个 agent 的 72 episodes = 2 个训练 seeds × 36 original development scenes。

| Arm | Agent | 20 s 内完整超车 | 120 s 内完整超车 | 成功时间中位数 | 成功时间范围 |
| --- | --- | ---: | ---: | ---: | --- |
| B | FD | 0/72 | 0/72 | N/A | N/A |
| B | BAL | 0/72 | 0/72 | N/A | N/A |
| B | SP | 0/72 | 0/72 | N/A | N/A |
| C | FD | 64/72 | 72/72 | 15.4 s | 10.8–24.4 s |
| C | BAL | 64/72 | 72/72 | 15.2 s | 10.4–23.0 s |
| C | SP | 69/72 | 72/72 | 14.2 s | 10.4–21.4 s |

- B 的 216/216 都完成了左换道及左侧净超越阶段，但 0/216 达到“返回原车道中心附近并净领先”的确认起点，故完整成功为 0/216。
- C 的 216/216 都完成完整序列，197/216 在 20 s 内，另外 19 条在 20–120 s 内。对 C 可以确认：这两个 seeds 的全部受测轨迹确实在超越后返回了原车道并稳定领先，而不只是发送了换道动作。
- 所有 episode 在诊断停止前均无碰撞/驶出道路；C 成功即停止，不能据此断言成功后余下 120 s 仍无碰撞。
- 这是指定旧 selected models 的描述性复评，不推广为全部 seeds/场景的 100%，也不将成功时间中位数当作换道 onset 时间。FD/BAL/SP 的 onset 研究仍按原事件定义进行。

逐 episode、各阶段时间及 gap 分层结果分别位于 `outputs/overtake_success_return_pilot/old_policy_2seed/old_policy_episodes.csv` 和 `old_policy_summary.csv`。

## Short-training Results

| Arm / seed / agent | Group | 首次成功 step | 完整超车成功 transitions | replay 成功采样次数 | 至少被采样一次的成功 transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| B / 4190 / BAL | S00 | 781 | 1 | 27 | 1 |
| B / 4190 / BAL | S10 | 781 | 1 | 27 | 1 |
| C / 4290 / BAL | S00 | 485 | 2 | 49 | 2 |
| C / 4290 / BAL | S10 | 485 | 2 | 49 | 2 |

B 每组已结束且有目标的 episodes 为 17 个，成功 1 个；C 为 18 个，成功 2 个。每组另有 1 个 budget-stop episode；无目标场景与预算停止记录均保留，不并入上述分母。

B 的完整超车在 episode 内 14.8 s 确认；C 分别在 17.0/19.8 s 确认。事件发生于随机 warm-up，随后全部被实际 replay 采样用于更新。S00 只记录事件，S10 额外给 +30。

这些结果说明所测 B/C 的早期探索能触发“返回原车道”的完整超车奖励，不能说明奖励已经改变策略、所有 seed 都能学会或成功经验足够密集。每 arm 仅一个 seed、一个 agent，S00/S10 事件数相同，不作因果效果判断。未完成 episode、无目标和失败均保留。

## Reproduction

### First full training on Katana

本次先单独跑 B / BAL / S10 / seed4190 / 100K steps。保持换道动作惩罚 −0.2、成功 bonus +30、不因成功终止，使用完整超车返回判据。从头训练，不续接本地 2K 模型，也不调整课程或其他奖励。

在 Katana 的仓库根目录执行：

```bash
git pull --ff-only origin main
qsub experiments/overtake_success/scripts/katana_overtake_B_BAL_S10.pbs
```

这份脚本只提交一个任务。输出目录为 `/srv/scratch/$USER/highway_transition_timing/outputs/overtake_success_return_pilot/full_B_BAL_S10_seed4190`，默认拒绝覆盖。检查 `complete.json`、`training_summary.json`、`training_episodes.csv` 和 `final_model.zip`。这里的成功比例是训练探索统计；本脚本尚不包含训练后 deterministic evaluation，不能用它替代最终策略成功率。

完整 24-job PBS 仍保留，当前不需要提交它。长训练由用户在 Katana 启动，本地尚未运行 100K。

### Monitoring and diagnosed retry (2026-10-14)

PBS 将实时 stdout/stderr 直接写到 `/srv/scratch/$USER/highway_transition_timing/outputs/overtake_success_return_pilot/startup_logs/$PBS_JOBID.log`，无需等待 PBS 回传 `.o` 文件。日志含 shell 阶段和主机名，复用项目的无显示环境预检（单独限时 300 s）。

runner 写 `progress.json` 标记加载模块、创建环境、创建模型、进入训练、保存结果及完成阶段。训练中每 1,000 步或经过约 30 s 的下一次 callback 更新一次，附成功次数和已结束 episode 数，同时刷新 `training_episodes.partial.csv`。若 simulator 的单步本身阻塞，callback 无法运行；120 s Python 栈诊断帮助定位该阻塞，并不自动终止训练。

```bash
# 将 JOBID 替换为 qsub 返回的完整 job ID，例如 9508396.kman：
tail -f "/srv/scratch/$USER/highway_transition_timing/outputs/overtake_success_return_pilot/startup_logs/JOBID.log"
cat "/srv/scratch/$USER/highway_transition_timing/outputs/overtake_success_return_pilot/full_B_BAL_S10_seed4190/progress.json"
```

这些日志仅对新版新启动的任务有效，不会给已运行的旧 Python 进程追加心跳。原作业 9508396 的截图显示已运行约 2 h 15 min、cput 3 s、cpupercent 0，目录仅有 manifest；这是阻塞嫌疑，尚无调用栈证实根因。

需要重试时先处理旧作业，不删除或覆盖原目录。使用不同 run name，例如：

```bash
qsub -v OVERTAKE_RUN_NAME=full_B_BAL_S10_seed4190_retry1 \
  experiments/overtake_success/scripts/katana_overtake_B_BAL_S10.pbs
```

retry 从头训练，不是 checkpoint resume；训练配置和 seed 不变。观察 retry 的 progress 时也使用对应目录名。

### Local-environment retry after import timeout

作业 9509030 以 Exit_status=124 结束，walltime 5:52。提供的调用栈显示 `Preflight importing matplotlib.pyplot` 后，在 NumPy 导入链的 `importlib ... get_data` 读取模块代码处等待；共享环境路径为 `/srv/scratch/.../venvs/highway-transition/`。这支持进一步排查共享依赖读取，而不是认定字体缓存配置已解决；截图尚未给出确切阻塞文件或文件系统根因。

使用本实验的 [本地环境重试 PBS](scripts/katana_overtake_local_retry.pbs)。从 Katana 仓库根目录执行：

```bash
git pull --ff-only origin main
bash experiments/overtake_success/scripts/pack_training_env.sh
```

打包完成后会打印包含真实 `VENV_BUNDLE` 路径的 qsub 命令，直接执行该行即可。已有完整的 `venv.tar` 和 `venv.tar.sha256` 快照时可以复用，无需重新打包：

```bash
qsub -v VENV_BUNDLE=/实际快照目录 \
  experiments/overtake_success/scripts/katana_overtake_local_retry.pbs
```

该流程只归档现有环境，复制到计算节点 `/tmp`，校验 SHA256 后解包，并验证关键依赖实际来自本地环境；不安装/升级包。基础 Python 仍来自原来的 module，项目源码和输出仍在原路径，不能承诺绕过所有共享存储访问。复制/解包及导入预检均有超时诊断。

默认独立输出 `full_B_BAL_S10_seed4190_local_retry1`，拒绝覆盖；仍然是 B/BAL/S10/seed4190/100K、换道惩罚 −0.2、成功奖励 +30。日志仍在 startup_logs 下按完整 PBS job ID 命名，正常应依次看到环境复制、校验、解包、`Preflight passed.`、`training_started` 和 steps 增长。此前作业已经结束，无需再次 qdel。这个方案已经完成本地静态/测试检查，但尚未在用户集群验证成功。

### Diagnostic commands

从项目根目录运行；输出目录必须不存在。

```bash
PYTHONPATH=src .venv/bin/python experiments/overtake_success/pilot.py old-policy \
  --out outputs/overtake_success_return_pilot/old_policy_2seed

PYTHONPATH=src .venv/bin/python experiments/overtake_success/pilot.py train \
  --arm B --agent BAL --seed 4190 --group S10 --steps 2000 \
  --out outputs/overtake_success_return_pilot/prefix_B_BAL_S10_seed4190

python3 experiments/overtake_success/collect_pilot.py \
  --root outputs/overtake_success_return_pilot

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_overtake_success_pilot.py \
  tests/test_multilane_open_lane_change_experiment.py tests/test_highway_adapter.py
```

C 短训练使用 seed4290；S00 使用独立目录。100K 扩展 PBS 已切换到新定义输出目录：`experiments/overtake_success/scripts/katana_overtake_trigger_pilot.pbs`，仍未提交。

## Validation

29 项相关测试通过，新增检查包括：超越后留在左车道不算成功、未左换道直接在右侧超过不算成功、超越前返回不算成功；保留几何边界、持续时间、碰撞优先、无目标、被动环境等价和真实 replay 采样检查。

四个短训练模型均完成 reload/predict/step 检查；collector 汇总 432 条旧策略轨迹和 4 个短训练 runs，PBS 语法、文档链接及 whitespace 检查通过。未运行长训练或正式四组统计。

旧报告 [PILOT.md](PILOT.md) 的 B216/216 与 C0/216 是“左车道保持”事件的统计，不能作为当前完整超车成功率。历史日志保留原记录，由本报告和当前设计文档取代其成功语义。
