# Arm A: Single-Lane Slow-Front Experiment

Arm A isolates longitudinal control. The ego vehicle cannot change lanes, so
increasing front spacing requires slowing down and sacrificing speed reward.
The endpoint is the first persistent slowdown-action onset extracted by the
shared ONSET pipeline.

## Current Configuration

- Training seeds: 3100-3119.
- Run prefix: `single_lane_slow_front_fixed_100k_seed`.
- Policies: FD, BAL, and SP Double DQN reward conditions.
- Training budget: 100,000 policy steps per policy.
- Training episode: 20 physical seconds.
- Evaluation episode: 120 physical seconds at 5 Hz.
- Development grid: 36 matched exposures.
- Selected model: latest checkpoint that passes the frozen development gate.

The current training array is:

```bash
qsub scripts/katana_armA_single_lane_fixed_20seed.pbs
```

## Counterfactual Evaluation

The standalone development-grid battery runs `original`, `no-front`,
`matched-speed-front`, and `far-front` without changing the selected models:

```bash
qsub scripts/katana_armA_single_lane_counterfactual.pbs
```

For one local completed run:

```bash
PYTHONPATH=src python3 experiments/single_lane_slow_front/run.py \
  rollout-counterfactual \
  --run-dir outputs/single_lane_slow_front_fixed_100k_seed3100 \
  --num-exposures 36 \
  --evaluation-duration 120 \
  --no-figures
```

## Held-Out V2

Arm A uses the sealed shared A/B grid. It is loaded only when
`--eval-grid heldout-v2` is explicit, and its checksum and physical support are
validated before any model inference.

```bash
qsub -J 0-19 scripts/katana_armA_heldout_v2.pbs
```

For a single-seed retry, pass the zero-based index without an array. For
example, seed 3119 is:

```bash
qsub -v ARM_A_INDEX=19 scripts/katana_armA_heldout_v2.pbs
```

Outputs are written beneath each run at
`rollout_counterfactual_heldout_v2/`. The final across-seed analysis is handled
by `scripts/aggregate_heldout_v2.py` after all three arms are available.
