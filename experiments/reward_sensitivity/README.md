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
