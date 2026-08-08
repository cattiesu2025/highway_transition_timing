# Highway Transition Timing

This project compares FD, BAL, and SP Double DQN policies under matched
`highway-env` exposures. The maintained experiment code is intentionally
limited to two experiment directories:

- `experiments/single_lane_slow_front/`: slowdown-onset timing in a controlled
  single-lane slow-front scenario;
- `experiments/multilane_open_lane_change/`: lane-change-onset timing when an
  adjacent escape lane is open.

Vanilla Stable-Baselines3 DQN is not supported. All training and model loading
use the project's Double DQN implementation in
`src/highway_transition_timing/double_dqn.py`.

## Setup

Create or activate a Python 3.10+ environment, then install the simulator
dependencies:

```bash
python -m pip install -r requirements-pilot.txt
```

Run the test suite from the project root:

```bash
PYTHONPATH=src pytest -q
```

## Single-Lane Slow-Front Experiment

Audit the stratified reset generator before training:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py audit \
  --out outputs/single_lane_slow_front_stratified_audit_seed0 \
  --num-resets 2000 \
  --seed 0
```

Run one local experiment:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py \
  --out outputs/single_lane_slow_front_stratified_100k_seed0 \
  --timesteps 100000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --seed 0 \
  --bootstrap-samples 500
```

Submit the validated five-seed array on Katana:

```bash
qsub scripts/katana_single_lane_stratified.pbs
```

Run the isolated 20-second-training/120-second-evaluation diagnostic over 20
independent seeds with:

```bash
qsub scripts/katana_single_lane_duration20_20seed.pbs
```

This keeps 100,000 total training steps and all reward/scenario settings fixed,
using separate `single_lane_slow_front_duration20_100k_seed<seed>` outputs.

See `experiments/single_lane_slow_front/README.md` for reset strata,
counterfactual commands, and output details.

## Multi-Lane Open-Lane Experiment

Run the controlled lane-change experiment:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py \
  --out outputs/multilane_open_lane_change_stratified_100k_seed0 \
  --timesteps 100000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --training-scenario-profile stratified \
  --bootstrap-samples 500
```

Submit the five-seed multi-lane retraining array on Katana:

```bash
qsub scripts/katana_multilane_open_lane_change.pbs
```

Each array task trains FD, BAL, and SP for one seed and writes to
`/srv/scratch/$USER/highway_transition_timing/outputs/multilane_open_lane_change_stratified_100k_seed<seed>/`.

Before entering the held-out stage, submit the isolated 20-second-training,
20-seed multi-lane diagnostic:

```bash
qsub scripts/katana_multilane_duration20_20seed.pbs
```

Array indices 0-19 map to seeds 4000-4019. Each policy still receives 100,000
training steps and 120-second development evaluation; only the training episode
horizon is shortened to 20 seconds. The PBS command passes `--no-figures`.
After copying the completed run directories back under local `outputs/`, create
the 20-seed training figure locally with:

```bash
R_LIBS_USER=tmp/r-lib Rscript \
  figures/multilane/plot_multiseed_results.R \
  outputs \
  outputs/multilane_open_lane_change_duration20_100k_20seed/figures \
  duration20
```

This diagnostic uses only the development grid and does not open the sealed
held-out grid.

The existing multi-lane counterfactual PBS script targets the earlier
`mixed_100k` diagnostic models. The new stratified models use the checksum-
sealed disjoint held-out grid through:

```bash
qsub scripts/katana_multilane_heldout_rollout.pbs
```

Submit that inference array only after all five new training runs pass their
training-integrity and development checks.

See `experiments/multilane_open_lane_change/README.md` for smoke tests,
counterfactual rollouts, and report-figure generation.

## Project Layout

```text
experiments/
├── single_lane_slow_front/
└── multilane_open_lane_change/
scripts/
├── katana_multilane_heldout_rollout.pbs
├── katana_multilane_duration20_20seed.pbs
├── katana_multilane_open_lane_change.pbs
├── katana_single_lane_duration20_20seed.pbs
└── katana_single_lane_stratified.pbs
figures/                        # local plotting workspace, ignored
├── single_lane/
├── multilane/
└── shared/
src/highway_transition_timing/
tests/
outputs/                         # generated and ignored
```

Generated runs, local figure scripts, caches, dependency folders, and temporary
work products are ignored. Experiment commands recreate their required output
directories.
