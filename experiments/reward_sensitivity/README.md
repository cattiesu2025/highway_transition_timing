# Exploratory reward-coefficient comparison

Six new coefficient pairs, three arms and five paired training seeds produce
90 new models. Existing FD/BAL/SP baselines use the same five seeds per arm
(45 baseline condition records). This is a new development-only experiment;
it does not replace the maintained experiments or reopen held-out v2.

| Variant | Internal policy label | Speed | Front spacing |
|---|---|---:|---:|
| R1 | FD | 0.33 | 0.51 |
| R2 | FD | 0.42 | 0.98 |
| R3 | BAL | 0.74 | 0.92 |
| R4 | BAL | 0.88 | 0.96 |
| R5 | SP | 1.17 | 0.76 |
| R6 | SP | 1.03 | 0.48 |

The internal policy label supports existing environment/model-selection code;
**variant + arm + seed** identifies a new policy, not FD/BAL/SP alone.
Baseline weights remain (0.45,1), (0.7,0.7), and (1,0.25).

Sampling uses Python `random.Random(20261006)`, independently uniform terms in
[0.25,1.25], rounded to two decimals. Keep the first two pairs in each ratio
region: spacing/speed >1.5, <2/3, and the intervening region. This is stratified
random sampling, not six unconditioned draws. No resampling after outcomes.
`jobs.json` records the exact 90 tasks; `run.py` reproduces and checks the sample.

## Protocol

- A seeds 3100–3104; B 4100–4104; C 4200–4204.
- 100K steps, 20 s training episodes, 120 s evaluation, 5 Hz policy.
- Existing Double DQN architecture/optimizer and checkpoint eligibility rule.
- Common collision cost 2 and risk cost 3; B/C slowdown and lane-change costs
  0.2. A retains its existing costs (slowdown 0; inactive lane-change cost 0.1).
- Only speed/spacing coefficients vary. No fixed sum; ratio and strength
  relative to penalties may both change.
- Every configuration is reported, including failures/censoring. No selection
  by onset separation or by incomparable raw returns.
- Training seed is the replication unit; five seeds are exploratory.
- A endpoint: persistent slowdown onset. B/C endpoint: physical lane change,
  using `first_actual_lane_change_seconds`, not lane-change action time.
- Training curves use SB3 logged episode means. Across-seed median/IQR uses
  1K-step interpolation inside observed support; no extrapolation to 100K.
- Behavior reports episode-weighted mean speed, finite-front spacing, collision
  and onset occurrence, failures/censoring, conditional median latency. Infinite
  spacing (no front) is excluded explicitly and its episode coverage recorded.
- Independent original/no-front/matched-speed-front/far-front controls reuse
  selected models; no new checkpoint selection during counterfactual evaluation.

## Run

The project requires Python 3.10 or newer. On the Katana login node, activate
its maintained training environment before running any manual Python commands
(the system `python3` may be too old):

```bash
module load python/3.11.3
source "/srv/scratch/${USER}/venvs/highway-transition/bin/activate"
python3 --version
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
```

`SyntaxError: future feature annotations is not defined` means the selected
interpreter is too old and the script has not started. Load/activate the above
environment; do not remove the future import. The PBS script already performs
these environment setup steps automatically.

From the repository root, inspect a task without executing it:

```bash
PYTHONPATH=src python3 experiments/reward_sensitivity/run.py --index 0
PYTHONPATH=src python3 experiments/reward_sensitivity/run.py --index 89 --phase evaluate
```

Indices 0–29 are A, 30–59 B, 60–89 C; within each arm, R1–R6 each use five
consecutive seed indices. Long runs belong on Katana using the same environment
as the baselines. Inspect/update that environment deliberately rather than
silently upgrading dependencies during this comparison.

```bash
qsub scripts/katana_reward_sensitivity.pbs
```

After successful training, submit independent counterfactual evaluation:

```bash
qsub -v SWEEP_PHASE=evaluate scripts/katana_reward_sensitivity.pbs
```

Each task writes an exclusive directory and `sweep_job.json`. Existing run
folders are refused to prevent accidental overwrites. A failed task's partial
folder must be inspected and moved aside before rerunning; it is never deleted
automatically. Evaluation verifies the job manifest and selected model.

After synchronizing outputs to this checkout:

```bash
python3 experiments/reward_sensitivity/collect.py
Rscript experiments/reward_sensitivity/plot_convergence.R
```

The collector checks weights, common costs, training protocol, scene geometry,
and each new run's shared reset prefix against its paired baseline. Missing
runs are recorded in `run_status.csv`; the default complete export refuses them.
Counterfactual summaries are included when available and retain their original
arm-specific fields; their absence does not block training-curve export.

To inspect available baseline data before new runs finish:

```bash
python3 experiments/reward_sensitivity/collect.py --allow-incomplete
Rscript experiments/reward_sensitivity/plot_convergence.R --allow-incomplete
```

Partial figures are prominently labelled PREVIEW. No new-coefficient curves are
fabricated. Source tables and PDF/SVG/PNG/TIFF figures are written beneath
`outputs/reward_sensitivity/`. R loads the existing optional `tmp/r-lib` library;
required packages are ggplot2, dplyr, patchwork, svglite, and ragg.

## Validation

```bash
PYTHONPATH=src python3 -m pytest -q
bash -n scripts/katana_reward_sensitivity.pbs
```

Short simulator smokes validate wiring/save/reload only. They do not establish
convergence, safety, or sensitivity findings. A full six-variant figure cannot
be generated until the 90 new models have been trained and evaluated.

## Symmetric-weight follow-up (R7/R8)

This is a targeted, exploratory follow-up designed after inspecting R1–R6,
not part of the original random sample. It tests two equal-total-weight pairs:

| Pair | Front-distance preference (speed, spacing) | Speed preference (speed, spacing) |
|---|---|---|
| Sum 1.25 | **R7 (0.25, 1.00), new** | SP (1.00, 0.25), existing |
| Sum 1.45 | FD (0.45, 1.00), existing | **R8 (1.00, 0.45), new** |

Only A and C are included: 2 new configurations × 2 arms × 5 seeds = 20 new
models. Reuse 20 existing FD/SP condition records; do not retrain baselines.
Training, checkpoint selection, penalties and development evaluation are
unchanged. This tests directional separation at two symmetric endpoint pairs,
not monotonicity throughout the reward space or generalization to new seeds.
All outcomes, including reverse gaps, non-onsets and collisions, are retained.

`jobs_symmetric.json` freezes the new task mapping:

| Array index | Arm | Variant | Seeds |
|---|---|---|---|
| 0–4 | A | R7 | 3100–3104 |
| 5–9 | A | R8 | 3100–3104 |
| 10–14 | C | R7 | 4200–4204 |
| 15–19 | C | R8 | 4200–4204 |

From the repository root, with the Python environment above active:

```bash
python3 experiments/reward_sensitivity/run_symmetric.py --index 0
python3 experiments/reward_sensitivity/run_symmetric.py --index 19 --phase evaluate
qsub scripts/katana_reward_symmetric.pbs
```

After all training tasks succeed, submit independent counterfactual evaluation:

```bash
qsub -v SWEEP_PHASE=evaluate scripts/katana_reward_symmetric.pbs
```

Synchronize the new `reward_sensitivity_arm{A,C}_R{7,8}_100k_seed*` directories,
including models, metadata and evaluation outputs. The original PBS and R1–R6
manifest remain unchanged. Existing output directories are refused, not replaced.

Collect the 40 conditions into a separate follow-up analysis directory:

```bash
python3 experiments/reward_sensitivity/collect_symmetric.py
# Preview before all new runs are present:
python3 experiments/reward_sensitivity/collect_symmetric.py --allow-incomplete
```

Outputs are under `outputs/reward_symmetric/analysis/`, separate from the
original 135-condition collection. Besides the standard seed/outcome/curve
and status tables, exports include:

- `paired_episode_gaps.csv`: matched scenario/rollout gaps, defined as
  **SP-preference onset minus FD-preference onset**. Positive means the
  speed-preference agent responds later. A uses persistent slowdown, C uses
  physical lane crossing. Collisions remain explicit even after a valid onset.
- `paired_seed_gaps.csv`: conditional median of within-scene gaps for each
  training seed, counts of positive/negative/tied gaps and excluded pairs.
  A missing onset on either side leaves the episode gap blank; it is never
  imputed as zero or as the evaluation horizon.
- `pair_status.csv`: all 20 planned seed comparisons, including missing sides.

Pairing checks scenario/rollout IDs and logged exposure/rollout seeds. Both
sides must contain the same 36 unique scene/rollout IDs. Training reset-prefix
checks and metadata checks are inherited from the original collector. Across-
seed summaries should treat the five seed medians as the replication units,
not treat 180 scenes as independent training replicates. Counterfactual exports
are included when present; their absence does not block original-scene gaps.
The original R plotting script is for R1–R6 and does not plot this follow-up.

### Font-cache startup stall and single-task retry

The symmetric PBS entry points now source `scripts/katana_headless_setup.sh`.
Each job gets private temporary Matplotlib/fontconfig caches and an isolated
fontconfig configuration referencing Matplotlib's bundled fonts. The preflight
imports Matplotlib, SB3 and highway-env, prints each stage, dumps Python stacks
if startup exceeds 60 seconds, and is limited to 300 seconds (+10 seconds kill
grace). This mitigates system/user fontconfig scanning during headless startup;
it does not prove every font-cache stall has that cause. `--no-figures` cannot
prevent plotting imports performed by dependencies. No reward, seed, training
or selection setting changes. These settings are for headless experiment jobs,
not manuscript figure generation requiring custom fonts.

For a single failed training condition, first confirm the previous task has
ended and inspect/archive its output directory. Then submit a regular job
(no singleton array syntax required):

```bash
qsub -v SWEEP_INDEX=2 scripts/katana_reward_symmetric_retry.pbs
```

The default index is 2 (A/R7/3102). Other indices 0–19 can be supplied. The retry
script uses the same startup checks and commands as the array script. Preflight
failure occurs before any new experiment folder is created; inspect its logged
stage/traceback rather than repeatedly submitting the unchanged job. A successful
local preflight does not verify cluster filesystem or fontconfig performance.

If the cluster disallows `strace`/`ptrace`, use the import-only diagnostic:

```bash
qsub scripts/katana_import_diagnostic.pbs
# After completion, substitute the numeric job ID:
tail -n 160 reward-import-check.oJOBID
```

This uses Python's own `open` audit events, enabled only by
`HEADLESS_TRACE_OPENS=1`, in the same bounded preflight. No ptrace, privilege
changes, training or seed selection occurs. The normal PBS output contains
`PYTHON_OPEN` timestamps/paths/modes, stage markers and periodic stacks, not file
contents. An open event is an attempt, not proof of successful reading; correlate
its last paths with the stack rather than assuming the last filename is faulty.
`Preflight passed` means dependency imports finished, not that training ran.
