# Highway Transition Timing

This repository studies whether the onset time of an RL policy's behavioural
transition reveals how its reward trades speed against front spacing. The
maintained evidence consists of three controlled highway arms, twenty
independently trained seeds per arm, and a checksum-sealed held-out evaluation.

## Maintained Experiments

| Arm | Environment | Manipulation | Endpoint | Training seeds |
| --- | --- | --- | --- | --- |
| A | single lane | slowing buys spacing but costs speed | persistent slowdown onset | 3100-3119 |
| B | two lanes, empty target lane | lane change improves speed and spacing together | physical lane-index change | 4100-4119 |
| C | two lanes, occupied target lane | lane change restores the speed-spacing trade-off | physical lane-index change | 4200-4219 |

FD, BAL, and SP are reward conditions, not personality labels. Within each
training seed they share the same scenario sequence. Across-seed inference uses
the training seed as the unit of replication.

Current run prefixes:

```text
single_lane_slow_front_fixed_100k_seed<3100-3119>
multilane_twolane_open_fixed_100k_seed<4100-4119>
multilane_twolane_occupied_fixed_100k_seed<4200-4219>
```

## Repository Layout

```text
experiments/
  single_lane_slow_front/       Arm A implementation
  multilane_open_lane_change/   Arms B and C implementation
  heldout_v2/                   Frozen grids and confirmatory protocol
scripts/
  katana_arm*_fixed_20seed.pbs  Current training arrays
  katana_arm*_counterfactual.pbs
  katana_arm*_heldout_v2.pbs
  aggregate_heldout_v2.py
src/highway_transition_timing/  Shared onset, analysis, and reward code
tests/                          Unit and integration tests
docs/
  proposals/                    Canonical research proposal
  manuscripts/acra2026/         Current ACRA manuscript
  implementation.md             Current implementation contract
  dev_log.md                    Historical investigation record
outputs/                        Generated models and results; never cleaned here
```

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[pilot,plots,dev]'
```

Use `PYTHONPATH=src` when running directly from a source checkout.

## Current Training Arrays

The existing 20-seed outputs were produced by these scripts. Rerunning them is
not required for held-out v2.

```bash
qsub scripts/katana_armA_single_lane_fixed_20seed.pbs
qsub scripts/katana_armB_twolane_open_fixed_20seed.pbs
qsub scripts/katana_armC_twolane_occupied_fixed_20seed.pbs
```

## Independent Counterfactuals

These inference-only arrays evaluate the selected models on the development
grid. They do not retrain or replace checkpoints.

```bash
qsub scripts/katana_armA_single_lane_counterfactual.pbs
qsub scripts/katana_armB_twolane_counterfactual.pbs
qsub scripts/katana_armC_twolane_counterfactual.pbs
```

## Held-Out V2

The final confirmatory protocol is frozen in
`experiments/heldout_v2/protocol.md`. It reuses the existing 60 selected models
and runs inference on physical values that are disjoint from development while
remaining inside the maintained training and observation support.

Submit once on Katana:

```bash
qsub -J 0-19 scripts/katana_armA_heldout_v2.pbs
qsub scripts/katana_armB_heldout_v2.pbs
qsub scripts/katana_armC_heldout_v2.pbs
```

To retry one Arm A seed after an infrastructure-level failure without creating
an array, pass its zero-based index explicitly. For example, seed 3119 is:

```bash
qsub -v ARM_A_INDEX=19 scripts/katana_armA_heldout_v2.pbs
```

After all run directories are synchronized beneath local `outputs/`, validate
and aggregate them:

```bash
PYTHONPATH=src python3 scripts/aggregate_heldout_v2.py
```

The aggregate is written to `outputs/heldout_v2_20seed/analysis/`, including a
short `synchronisation_summary.md` for discussion with collaborators.

## Manuscript

Build the current ACRA draft from `docs/manuscripts/acra2026/`:

```bash
latexmk -pdf paper.tex
```

The manuscript currently distinguishes development evidence from held-out
evidence. A campus user study remains a separate claim-expansion decision.

## Verification

```bash
PYTHONPATH=src pytest -q
bash -n scripts/*.pbs
python3 -m py_compile scripts/aggregate_heldout_v2.py
```

Historical diagnostics and superseded experiments are described only in
`docs/dev_log.md`; obsolete execution scripts are intentionally not retained.
