# Arms B and C: Two-Lane Lane-Change Experiment

The multi-lane experiment measures the first physically completed lane change,
not merely the first lateral action. Arm B leaves the target lane empty, so a
lane change improves both speed and front spacing. Arm C adds a target-lane
vehicle and varies its merge gap, restoring a speed-spacing trade-off.

## Current Configuration

| Arm | Target lane | Seeds | Run prefix |
| --- | --- | --- | --- |
| B | empty | 4100-4119 | `multilane_twolane_open_fixed_100k_seed` |
| C | occupied | 4200-4219 | `multilane_twolane_occupied_fixed_100k_seed` |

Both arms use two lanes, ego lane 1, FD/BAL/SP Double DQN reward conditions,
100,000 policy steps per policy, 20-second training episodes, 120-second
evaluation episodes, and checkpoint-gated model selection.

Current training arrays:

```bash
qsub scripts/katana_armB_twolane_open_fixed_20seed.pbs
qsub scripts/katana_armC_twolane_occupied_fixed_20seed.pbs
```

## Counterfactual Evaluation

Both arms run `original`, `no-front`, `matched-speed-front`, and `far-front`.
Arm C additionally runs `open-target-lane`, which removes only the target-lane
vehicle while holding the trained policy fixed.

```bash
qsub scripts/katana_armB_twolane_counterfactual.pbs
qsub scripts/katana_armC_twolane_counterfactual.pbs
```

For one local completed Arm C run:

```bash
PYTHONPATH=src python3 \
  experiments/multilane_open_lane_change/rollout_counterfactual.py \
  --run-dir outputs/multilane_twolane_occupied_fixed_100k_seed4200 \
  --eval-grid development \
  --variants original no-front matched-speed-front far-front open-target-lane
```

## Held-Out V2

Arm B uses the shared A/B grid. Arm C uses the occupied grid with target-lane
gaps 25/55/85 m. The loader selects the correct sealed file from the training
metadata and rejects checksum, support, geometry, or exposure-count mismatches.

```bash
qsub scripts/katana_armB_heldout_v2.pbs
qsub scripts/katana_armC_heldout_v2.pbs
```

Each run writes `rollout_counterfactual_heldout_v2/`. Aggregate all A/B/C
results with `scripts/aggregate_heldout_v2.py` after synchronization.
