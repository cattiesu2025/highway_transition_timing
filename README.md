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
sbatch scripts/katana_single_lane_stratified.slurm
```

See `experiments/single_lane_slow_front/README.md` for reset strata,
counterfactual commands, and output details.

## Multi-Lane Open-Lane Experiment

Run the controlled lane-change experiment:

```bash
PYTHONPATH=src python experiments/multilane_open_lane_change/run.py \
  --out outputs/multilane_open_lane_change_mixed_100k \
  --timesteps 100000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --bootstrap-samples 500
```

See `experiments/multilane_open_lane_change/README.md` for smoke tests,
counterfactual rollouts, and report-figure generation.

## Project Layout

```text
experiments/
├── single_lane_slow_front/
└── multilane_open_lane_change/
scripts/
├── katana_single_lane_stratified.slurm
└── plot_training_diagnostics.py
src/highway_transition_timing/
tests/
outputs/                         # generated and ignored
```

Generated runs, caches, local dependency folders, and temporary work products
are ignored. Experiment commands recreate their required output directories.
