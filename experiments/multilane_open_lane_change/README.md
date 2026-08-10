# Multi-Lane Open-Lane Lane-Change Experiment

This is an isolated multi-lane sanity experiment for lane-change timing.

The controlled evaluation scene contains:

- one ego vehicle in the right lane of a two-lane road;
- one slower front vehicle in the ego lane;
- one empty adjacent lane to the left;
- no random background traffic.

The geometry is two lanes with ego lane 1, so the only escape from the slow
front vehicle is a left lane change. This removes the choice of which open lane
to take, which was a confound on lane-change timing under the earlier four-lane
geometry. Results produced before this change used four lanes and are retained
as superseded development evidence; four-lane policies cannot be evaluated
here, because the `highway-env` observation `y` normalisation range scales with
the lane count.

The goal is to test whether FD, BAL, and SP show different lane-change timing
when a lane-change escape is genuinely available.

## Why This Exists

This experiment provides a direct lane-change opportunity and records both the
first lane-change action and the first completed physical lane change. The
separation keeps action selection and vehicle motion explicit in the analysis.

This experiment is not the final Highway result. It is a clean diagnostic:

- If FD/BAL/SP separate here, then the multi-lane timing question is still
  promising under a controlled open-lane response.
- If SP and FD remain indistinguishable, then the current reward design may not
  produce the intended lane-change timing contrast.

## Run A Smoke Test

Training defaults to the same deterministic shuffled 20-reset block as the
single-lane experiment:

- 4 no-front controls;
- 2 exact matched-speed controls;
- 2 strictly non-closing-front controls;
- 2 near-closing-front scenes;
- 8 visible slow-front scenes, two in each frozen TTC/deceleration cell;
- 2 gradual boundary-visible slow-front scenes.

All front-vehicle training scenes are visible at reset. This avoids assigning
conflicting actions to the same empty observation in no-front and initially
hidden-hazard episodes; the feed-forward DQN has no memory with which to
distinguish those states.

First audit the generator without training:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py audit \
  --out outputs/multilane_open_lane_change_stratified_training_audit_seed0 \
  --num-resets 2000 \
  --seed 0
```

The audit writes every sampled scene and its family, scene seed, block
position, net gap, closing speed, TTC, required deceleration, visibility, and
difficulty labels.

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py \
  --out outputs/multilane_open_lane_change_stratified_20k_smoke \
  --timesteps 20000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --training-scenario-profile stratified \
  --near-matched-speed-delta 2.0 \
  --lane-change-penalty 0.2 \
  --slow-down-penalty 0.2 \
  --collision-risk-penalty 3.0 \
  --bootstrap-samples 500
```

For a fast code smoke:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py \
  --out outputs/multilane_open_lane_change_stratified_tiny_smoke \
  --timesteps 200 \
  --num-exposures 36 \
  --agents FD \
  --duration 20 \
  --evaluation-duration 20 \
  --training-scenario-profile stratified \
  --bootstrap-samples 10 \
  --no-figures \
  --verbose 0
```

## Five-Seed Katana Retraining

Submit the OpenPBS array from the project root on Katana:

```bash
qsub scripts/katana_multilane_open_lane_change.pbs
```

Array indices 0-4 are used as training seeds. Each task trains FD, BAL, and SP
for 100,000 steps with the same deterministic within-seed 20-reset schedule
and 36 matched development-validation exposures. Outputs are
written to:

```text
/srv/scratch/$USER/highway_transition_timing/outputs/multilane_open_lane_change_stratified_100k_seed<seed>/
```

The development-validation rollout produced by this command is not the sealed
held-out evaluation. The versioned held-out grid is fixed before five-seed
training, but model inference on it is run only after the five completed runs
pass their training-integrity and development checks.

The existing `katana_multilane_counterfactual_rollout.pbs` remains tied to the
earlier `mixed_100k` diagnostic models. Do not use it as the final evaluation
of this retraining iteration.

## 20-Second Training-Horizon Diagnostic

Before opening the held-out grid, submit the isolated 20-seed training-density
diagnostic from the project root:

```bash
qsub scripts/katana_multilane_duration20_20seed.pbs
```

The OpenPBS array maps indices 0-19 to seeds 4000-4019. It changes only the
training episode horizon from 120 to 20 physical seconds. FD, BAL, and SP each
retain 100,000 total training steps, 120-second evaluation, 1x rewards, ego
lane 1, the frozen stratified reset block, lane-change cost 0.2, collision-risk
cost 3.0, and 36 matched development exposures. At 5 Hz, the training wrapper
enforces an exact 100-policy-step episode cap; evaluation is capped at 600
policy steps.

This array is retained to reproduce the superseded four-lane results; it still
passes `--lanes-count 4` explicitly. The maintained protocol is the two-lane
selected array below.

## Two-Lane Retraining With Model Selection

```bash
qsub scripts/katana_multilane_twolane_selected_20seed.pbs
```

Array indices 0-19 map to seeds 4000-4019 and write to
`multilane_open_lane_change_twolane_selected_100k_seed<seed>`, so the earlier
four-lane directories are untouched. Each policy saves a checkpoint every 5,000
steps under `models/checkpoints/`; the run then keeps the latest checkpoint that
passes the frozen development eligibility gate and records every checkpoint it
examined in `model_selection.csv`. The gate and its thresholds are documented in
`docs/implementation.md`. The sealed held-out grid is not loaded by this array.

Each task writes to:

```text
/srv/scratch/$USER/highway_transition_timing/outputs/
  multilane_open_lane_change_duration20_100k_seed<seed>/
```

The PBS command passes `--no-figures` and only produces models, logs, CSVs, and
development analysis on Katana. After copying all 20 directories into local
`outputs/`, generate the cross-seed convergence image locally:

```bash
R_LIBS_USER=tmp/r-lib Rscript \
  figures/multilane/plot_multiseed_results.R \
  outputs \
  outputs/multilane_open_lane_change_duration20_100k_20seed/figures \
  duration20
```

The `duration20` plotting profile expects seeds 4000-4019. It always exports
the local training figure and source CSVs; it defers the counterfactual figure
until all required development-counterfactual files exist. Neither the PBS job
nor this plotting command opens the sealed held-out grid.

## Sealed Held-Out Evaluation

The final grid is the versioned file `heldout_grid_v1.csv`. It contains a
36-scene full factorial with ego speeds 25/27/29 m/s, front distances
130/170/210/250 m, and front speeds 11/15/19 m/s. All three marginal value
sets are disjoint from the 36-scene development grid, and exposure seeds
9000000-9000035 are fixed across training seeds. Its frozen SHA-256 is:

```text
d861b169fb618b0053f972f128fa498d3249d48969209db99e342af7732e6964
```

The loader refuses to run if that checksum changes. After all five new
training runs pass their development checks, submit:

```bash
qsub scripts/katana_multilane_heldout_rollout.pbs
```

Each array task evaluates one independently trained seed on the identical
sealed original/no-front/matched-speed-front/far-front scenes and writes to
`RUN_DIR/rollout_counterfactual_heldout/`. The output includes
`evaluation_grid_manifest.csv`, which records the checksum and exposure count.

## Outputs

The run writes:

- `training_runs.csv`: model paths, reward settings, observation settings.
- `training_scenarios.csv`: reset-by-reset training scene manifest and physical
  difficulty labels.
- `evaluation/steps.csv`: per-step rollout records.
- `evaluation/exposures.csv`: matched exposure settings.
- `analysis/episode_outcomes.csv`: lane-change onset outcomes.
- `analysis/gap_summary.csv`: paired FD/BAL/SP timing gaps.
- `analysis/actual_lane_change_summary.csv`: first lane-change action time and
  first actual lane-index change time.
- `analysis/action_summary.csv`: action counts by agent.
- optional figures in `analysis/`.

## Report Figures

The report plotting script is maintained locally under the ignored `figures/`
workspace and is intentionally not distributed through Git. Generate the
report-facing multi-lane figures with:

```bash
PYTHONPATH=src python figures/multilane/plot_report_figures.py \
  --run-dir outputs/multilane_open_lane_change_mixed_20k_smoke
```

This writes:

- `outputs/multilane_open_lane_change_mixed_20k_smoke/analysis/report_lane_change_latency_by_agent.png`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/analysis/report_initial_action_counterfactual_rates.png`

The script also writes the open-lane initial-action counterfactual tables if
they do not already exist:

- `outputs/multilane_open_lane_change_mixed_20k_smoke/counterfactual_open_lane/counterfactual_initial_actions.csv`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/counterfactual_open_lane/counterfactual_action_summary.csv`

## Full-Episode Counterfactual

Compare the trained policies on matched slow-front and no-front episodes:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/rollout_counterfactual.py \
  --run-dir outputs/multilane_open_lane_change_mixed_100k \
  --eval-grid development \
  --variants original no-front \
  --num-exposures 36 \
  --evaluation-duration 120
```

This writes per-step rollouts, episode-level first lane-change times, and paired
`no-front - original` summaries under
`RUN_DIR/rollout_counterfactual_open_lane/`.

`duration` and `evaluation-duration` are physical seconds. With the default
5 Hz policy frequency, a 120-second counterfactual contains at most 600 policy
steps. Episode summaries keep the first command, first observed lateral motion,
and first post-step physical lane-index change separate; the physical change is
the primary realised event for this experiment.

After running all four variants, regenerate the report figures:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/rollout_counterfactual.py \
  --run-dir outputs/multilane_open_lane_change_mixed_100k \
  --eval-grid development \
  --variants original no-front matched-speed-front far-front

PYTHONPATH=src python figures/multilane/plot_report_figures.py \
  --run-dir outputs/multilane_open_lane_change_mixed_100k
```

The full-episode figure is written to
`analysis/report_rollout_counterfactual_lane_change_response.{png,svg,pdf}`.
An event-time version is also written to
`analysis/report_rollout_counterfactual_cumulative_incidence.{png,svg,pdf}`.

## Five-Seed R Figures

After the development rollouts for seed 0 and stratified seeds 1-4 are
available locally, generate the cross-seed training and physical-event
counterfactual figures with the local, Git-ignored plotting script:

```bash
R_LIBS_USER=tmp/r-lib Rscript \
  figures/multilane/plot_multiseed_results.R
```

The script pools the lane-cost-0.2 seed-0 pilot with stratified seeds 1-4. It
does not open the sealed held-out grid. Outputs are editable SVG/PDF, 600-dpi
TIFF, 300-dpi PNG, and source-data CSV files under
`outputs/multilane_open_lane_change_stratified_100k_multiseed/figures/`.
The counterfactual figure uses physical lane-index changes in seconds,
an all-episode occurrence heatmap, and conditional median/IQR timing among
observed events. Its source-data bundle also retains empirical cumulative
incidence, pooled and seed-level T50, and the matched-speed before-2-s
development gate.

## Interpretation

Primary analysis target:

```text
lane_change_onset
```

The generic pipeline still reports persistence-based stable lane-change
evidence as a diagnostic. `actual_lane_change_summary.csv` is the primary
realised-event table because a high-level lane-change command, lateral motion,
and a completed physical lane-index change are not always the same event.
