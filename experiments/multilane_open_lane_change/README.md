# Multi-Lane Open-Lane Lane-Change Experiment

This is an isolated multi-lane sanity experiment for lane-change timing.

The controlled evaluation scene contains:

- one ego vehicle;
- one slower front vehicle in the ego lane;
- multiple empty adjacent lanes;
- no random background traffic.

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

Training uses the same negative-control mixture as the single-lane sanity
experiment:

- 80% slow-front episodes;
- 20% no-front cruise episodes.

The no-front fraction is configurable, and the actual sampled reset counts are
written to `training_runs.csv`.

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py \
  --out outputs/multilane_open_lane_change_mixed_20k_smoke \
  --timesteps 20000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --no-front-train-fraction 0.2 \
  --slow-down-penalty 0.2 \
  --collision-risk-penalty 3.0 \
  --bootstrap-samples 500
```

For a fast code smoke:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py \
  --out outputs/multilane_open_lane_change_mixed_tiny_smoke \
  --timesteps 200 \
  --num-exposures 36 \
  --agents FD \
  --duration 20 \
  --evaluation-duration 20 \
  --no-front-train-fraction 0.2 \
  --bootstrap-samples 10 \
  --no-figures \
  --verbose 0
```

## Outputs

The run writes:

- `training_runs.csv`: model paths, reward settings, observation settings.
- `evaluation/steps.csv`: per-step rollout records.
- `evaluation/exposures.csv`: matched exposure settings.
- `analysis/episode_outcomes.csv`: lane-change onset outcomes.
- `analysis/gap_summary.csv`: paired FD/BAL/SP timing gaps.
- `analysis/actual_lane_change_summary.csv`: first lane-change action time and
  first actual lane-index change time.
- `analysis/action_summary.csv`: action counts by agent.
- optional figures in `analysis/`.

## Report Figures

Generate the report-facing multi-lane figures with:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/plot_report_figures.py \
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
  --variants original no-front \
  --num-exposures 36 \
  --evaluation-duration 120
```

This writes per-step rollouts, episode-level first lane-change times, and paired
`no-front - original` summaries under
`RUN_DIR/rollout_counterfactual_open_lane/`.

After running all four variants, regenerate the report figures:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/rollout_counterfactual.py \
  --run-dir outputs/multilane_open_lane_change_mixed_100k \
  --variants original no-front matched-speed-front far-front

PYTHONPATH=src python experiments/multilane_open_lane_change/plot_report_figures.py \
  --run-dir outputs/multilane_open_lane_change_mixed_100k
```

The full-episode figure is written to
`analysis/report_rollout_counterfactual_lane_change_response.{png,svg,pdf}`.
An event-time version is also written to
`analysis/report_rollout_counterfactual_cumulative_incidence.{png,svg,pdf}`.

## Interpretation

Primary analysis target:

```text
lane_change_onset
```

The main pipeline detects stable lane-change evidence using lane-change actions
and lane-index changes. The additional
`actual_lane_change_summary.csv` is included because a lane-change action and a
completed physical lane change are not always the same event.
