# Fixed-sum reward-ratio experiment

This separate, development-only follow-up fixes speed + front-spacing weights
at 1.40. It does not replace R1-R8, change their original sampling rules, or
reopen held-out v2. The question is how preference allocation changes onset
timing, not whether training returns form visually separated clusters.

## Frozen conditions

Ratio r is front spacing / speed. Groups are FD-like (r > 2), Intermediate
(0.5 <= r <= 2), and SP-like (r < 0.5). There are three configurations per group.

| Configuration | Speed | Front spacing | Group | Source |
|---|---:|---:|---|---|
| F1 | 0.28 | 1.12 | FD-like | New |
| F2 | 0.35 | 1.05 | FD-like | New |
| F3 | 0.42 | 0.98 | FD-like | Existing R2 |
| F4 | 0.56 | 0.84 | Intermediate | New |
| F5 | 0.70 | 0.70 | Intermediate | Existing BAL |
| F6 | 0.84 | 0.56 | Intermediate | New |
| F7 | 0.98 | 0.42 | SP-like | New |
| F8 | 1.05 | 0.35 | SP-like | New |
| F9 | 1.12 | 0.28 | SP-like | New |

Original FD/SP are not extra conditions. F3/F5 are relabelled only in the new
collection; their source files and models remain in their existing directories.
Keep all nine configurations regardless of the resulting timing or return.
Internal FD/BAL/SP agent labels select existing code paths; F identifiers and
their actual coefficients identify the experimental conditions.

## Protocol and task mapping

- A: persistent slowdown; seeds 3100-3104.
- C: physical lane crossing; seeds 4200-4204. Lane-change action time is not
  the endpoint.
- 100K training steps, 20-second training episodes, 120-second evaluations,
  36 paired development exposures, 5 Hz policy, existing Double DQN and
  checkpoint selection rules. No model is selected using onset separation.
- Existing arm-specific safety/action costs remain unchanged: collision 2,
  risk 3; A slowdown 0 and inactive lane-change cost 0.1; C slowdown and
  lane-change costs 0.2.
- 90 condition records total: 70 new models and 20 reused R2/BAL records.
  New training budget: 7,000,000 steps. Resource requests inherit the existing
  4 CPU / 16 GB / 5-hour-per-task PBS settings; runtime is not guaranteed.

| Configuration | A array indices | C array indices |
|---|---|---|
| F1 | 0-4 | 35-39 |
| F2 | 5-9 | 40-44 |
| F4 | 10-14 | 45-49 |
| F6 | 15-19 | 50-54 |
| F7 | 20-24 | 55-59 |
| F8 | 25-29 | 60-64 |
| F9 | 30-34 | 65-69 |

Each block uses ascending seeds. `jobs_fixed_sum.json` freezes all 70 new tasks.
Execution verifies this checked-in manifest. New output directories are named
`outputs/reward_fixed_sum_arm{A,C}_F*_100k_seed*`. Existing directories are
refused for training, including incomplete ones. Inspect and archive failed
training folders before retrying; never archive a completed model just to
repeat counterfactual evaluation.

## Run on Katana

Synchronize the new runner, manifest, collector and two PBS files, plus updated
`run.py` and `collect_symmetric.py`. Keep the existing `collect.py`,
`run_symmetric.py`, arm entry points and headless setup/preflight scripts on the
same project checkout. Commands below run from the project root.

For manual Python commands on the login node, activate the maintained environment:

```bash
module load python/3.11.3
source "/srv/scratch/${USER}/venvs/highway-transition/bin/activate"
python3 --version

# Read-only command previews; neither command starts training.
python3 experiments/reward_sensitivity/run_fixed_sum.py --index 0
python3 experiments/reward_sensitivity/run_fixed_sum.py --index 69 --phase evaluate
```

The PBS files activate the environment and run the existing bounded headless
preflight automatically:

```bash
qsub scripts/katana_reward_fixed_sum.pbs
```

For the current report, this is the only required submission: the default
training phase also performs original-scene evaluation. Once all tasks finish,
proceed directly to collection. Counterfactual evaluation is not required for
the reward/onset table or the training curves, and missing counterfactual
coverage does not prevent a complete primary collection.

### Optional counterfactual evaluation (not part of the current run)

Skip this subsection for the current report. If a later study needs these
controls, only after the corresponding training tasks have completed successfully:

```bash
qsub -v SWEEP_PHASE=evaluate scripts/katana_reward_fixed_sum.pbs
```

The evaluation phase reuses selected models, does not train, and writes the
original/no-front/matched-speed-front/far-front counterfactual outputs. Repeating
evaluation regenerates the selected task's counterfactual files.

Single-task retry (replace 0 with the actual failed index):

```bash
qsub -v SWEEP_INDEX=0 scripts/katana_reward_fixed_sum_retry.pbs
# Evaluation-only retry of an existing trained model:
qsub -v SWEEP_INDEX=0,SWEEP_PHASE=evaluate scripts/katana_reward_fixed_sum_retry.pbs
```

F3/R2 currently lacks counterfactual outputs in the inspected local data. This
does not require any action for the current report. For optional controls, if
still absent on the cluster, evaluate the ten existing A/C R2 models with the
original R1-R6 runner; do not retrain them:

```bash
qsub -J 5-9,65-69 -v SWEEP_PHASE=evaluate scripts/katana_reward_sensitivity.pbs
```

These are indices of the ORIGINAL 90-task reward-sensitivity array, not the
new fixed-sum array. F5/BAL counterfactual results are already available locally;
ensure they also exist under the collection root on the cluster or collect locally.

## Collection and interpretation

```bash
python3 experiments/reward_sensitivity/collect_fixed_sum.py \
  --root "/srv/scratch/${USER}/highway_transition_timing/outputs" \
  --out outputs/reward_fixed_sum/analysis
```

Before new training is complete, add `--allow-incomplete` for a labelled partial
collection. This never fabricates missing conditions. After downloading the run
folders, local collection uses:

```bash
.venv/bin/python experiments/reward_sensitivity/collect_fixed_sum.py
```

Collection checks actual weights/protocol/scene geometry, each condition's
training reset prefix against its paired baseline, selected model presence,
new-run manifests, and exposure/rollout identifiers and seeds across all available
configurations within each arm/seed. Existing model/source directories must be
present; summary tables alone do not support these checks.

Outputs under `outputs/reward_fixed_sum/analysis/`:

- Standard training, seed and episode outcome tables, retaining censoring,
  failures, collisions and denominators. Onset and collision can coexist.
- `condition_sources.csv`: F labels, source labels, weights, groups and directories.
- `paired_episode_gaps.csv`, `paired_seed_gaps.csv`, `pair_status.csv`: four
  mirror comparisons F9-F1, F8-F2, F7-F3 and F6-F4. There are 40 planned seed
  comparisons and 1,440 planned episode pairs. The inherited `fd_*`/`sp_*`
  column names mean higher-spacing/lower-spacing preference, including the
  Intermediate F4/F6 pair. Missing onset on either side leaves a blank gap.
- `counterfactual_status.csv`: all 90 conditions, separately checked for four
  variants with 36 episodes each. Training completeness does not imply complete
  counterfactual coverage. `collection.json` reports both counts explicitly.

The primary replication unit is the training seed (five per arm), not the
episodes or the nine weight settings. Compare per-seed conditional onset
medians alongside occurrence, censoring and collision rates. Mirror gaps are
computed within matched episodes before taking each seed's median. Positive
gap means the lower-spacing-preference policy responds later.

Show onset against spacing share (20%, 25%, 30%, 40%, 50%, 60%, 70%, 75%, 80%).
Training returns remain reward-specific even with equal coefficient sums.
This is an exploratory development follow-up using previously inspected seeds;
it is not independent held-out confirmation.

## Validation

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_reward_fixed_sum.py tests/test_reward_symmetric.py tests/test_reward_sensitivity.py
bash -n scripts/katana_reward_fixed_sum.pbs scripts/katana_reward_fixed_sum_retry.pbs
```

Local previews and mocked execution checks do not establish new training or
behavioral results. No cluster jobs are submitted by the setup scripts.

## Arm B extension (2026-10-16)

The original 70-job A/C manifest and indices remain frozen. Arm B uses a
separate 35-job manifest (`jobs_fixed_sum_armB.json`), seeds 4100–4104, and
`run_fixed_sum_armB.py`. Reuse B/R2 for F3 and B/BAL for F5; do not retrain them.
All training, reward costs, checkpoint and development-evaluation settings
inherit the existing B protocol (two lanes, empty target lane, 100K steps).

| B array indices | Configuration |
|---|---|
| 0–4 | F1 |
| 5–9 | F2 |
| 10–14 | F4 |
| 15–19 | F6 |
| 20–24 | F7 |
| 25–29 | F8 |
| 30–34 | F9 |

Each block uses ascending seeds. New output folders are
`outputs/reward_fixed_sum_armB_F*_100k_seed410*`; existing folders are refused.

### Node-local environment and submission

Use `katana_reward_fixed_sum_armB_local.pbs`, not the old A/C array.
It uses `scripts/katana_local_env.py` (stdlib only, launched with module Python
`-S`). A per-user, per-node file lock is held throughout extraction, preflight,
training and cleanup. Other updated B jobs on the same node wait without making
another environment copy (at most four hours, with progress every minute).
Jobs on different nodes still run in parallel. Waiting consumes PBS walltime;
very heavily packed nodes may require a later retry. The lock file is retained
so all waiters use the same inode; only each job's own temporary environment is
removed. This does not coordinate other users or older PBS entry points.

The shared archive is checksum-verified and extracted directly into a unique
node `/tmp` folder; no second archive copy is stored there. Before extraction,
the helper estimates allocated file sizes plus 512 MiB headroom and checks
free bytes and inodes. It fails clearly before training if capacity is
insufficient. Checksum, archive metadata scanning and extraction each have a
300-second bound. Child process groups are terminated on timeout/TERM before
cleanup and lock release. The same interpreter-path checks, private font caches,
and 300-second headless preflight remain. Results stay on scratch, package
versions and training settings are unchanged. PBS walltime stays five hours.
This addresses the observed same-user contention; unrelated users/processes
can still consume node space after the capacity check.

Sync the updated project files to Katana first, including the new B runner,
manifest and two PBS files, updated `run_fixed_sum.py` and collector, and the
new `scripts/katana_local_env.py`, existing packer/headless helpers and experiment dependencies. From the project
root on Katana:

```bash
# Only if an existing unchanged snapshot is not already available:
bash scripts/katana_pack_training_env.sh
# Use the real Bundle path printed by the packer (not its old retry command):
export VENV_BUNDLE=/srv/scratch/$USER/venv_archives/highway-env.REPLACE_ME
qsub -v VENV_BUNDLE scripts/katana_reward_fixed_sum_armB_local.pbs
```

For one failed B task, after confirming it ended and inspecting/archiving its
incomplete output folder, use the B-specific retry index:

```bash
qsub -v VENV_BUNDLE,SWEEP_INDEX=0 scripts/katana_reward_fixed_sum_armB_local_retry.pbs
```

Training includes original-scene evaluation. Counterfactual evaluation is not
needed for the training-curves request. After all B jobs finish and source
folders are available locally, collect to a separate three-arm destination:

```bash
.venv/bin/python experiments/reward_sensitivity/collect_fixed_sum.py \
  --include-arm-b --out outputs/reward_fixed_sum_ABC/analysis
```

Expect 135 conditions: 105 newly trained and 30 reused across A/B/C. With no
`--include-arm-b`, the collector still handles the original 90 A/C conditions.
The existing partial preview is historical; regenerate a full three-arm plot
from the completed collection rather than treating missing B curves as data.


### Recover the 22 disk-full failures from array 9533407

Audit of all 35 PBS logs: 13 exited with status 0; 22 failed during archive
copy/extraction with `No space left on device`, before preflight or training.
The nodes have 4 GiB `/tmp`, shared by multiple colocated tasks. The previous
PBS copied and extracted an environment per task without node coordination.

After syncing the fix, reuse the already verified snapshot; no repacking is
needed. Submit this dedicated recovery array, not the full 35-job array:

```bash
export VENV_BUNDLE=/srv/scratch/$USER/venv_archives/highway-env.2DBNKs
qsub -v VENV_BUNDLE scripts/katana_reward_fixed_sum_armB_recover.pbs
```

Recovery slots 0–21 map to original B indices:
`0,1,4,5,6,7,8,9,10,11,14,15,19,20,27,28,29,30,31,32,33,34`.
The 13 successful outputs are untouched. The runner still refuses to overwrite
any existing output folder. This recovery mapping is specific to this audited
batch; later individual failures use the single-task retry PBS and original
B indices. The new helper and all three B PBS files must be synced together.

Local validation: 20 tests passed for locking, unsafe lock-path rejection,
space/inode rejection, real tar extraction, no local archive copy, cleanup on
success/failure, child timeout, recovery mapping, existing preflight and runner
guards. PBS shell syntax checks passed. This is not a Katana execution test.
