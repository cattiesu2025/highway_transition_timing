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
