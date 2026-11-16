# Equal-start reward weights

Development-only calibration requested after advisor review. It finds SP and FD
weights whose t=0 episode return exactly equals BAL's (0.70, 0.70), so that the
three training curves share one starting point and diverge only through learning.

## Method

Until `learning_starts`, SB3 samples actions uniformly, independent of the network,
so t=0 behaviour is one well-defined policy. Episode return is linear in the two
preference weights, `R = c + w_s*S + w_d*D`, and `c` (collision, risk, action
penalties) does not depend on them. Scoring one shared set of random-policy episodes
therefore makes equality exact algebra:

- SP (`w_s = 2 w_d`): `w_d = 0.7 (S + D) / (2S + D)`
- FD (`w_d = 2 w_s`): `w_s = 0.7 (S + D) / (S + 2D)`

The reference set is 2000 episodes per arm, run through the real training stack with
training settings parsed from the sweep command, on seeds not used for training
(9100, 9200).

## Weights

Use these per arm. Rounding to 2 decimals reintroduces a gap of up to 0.30.

| Arm | t=0 return (SE) | S | D | SP (speed, spacing) | FD (speed, spacing) |
|---|---:|---:|---:|---:|---:|
| A | 59.138 (1.04) | 33.099 | 73.789 | (1.068978, 0.534489) | (0.414118, 0.828237) |
| C | 53.829 (0.67) | 31.297 | 73.698 | (1.078515, 0.539258) | (0.411300, 0.822601) |

Total weight is no longer fixed: about 1.60 SP, 1.40 BAL, 1.24 FD. The fixed collision
and risk penalties are therefore relatively lighter for SP and heavier for FD.

## What is and is not guaranteed

- Exact: the x=0 point, which is this computed reference value, not a training-log
  entry. State that in any figure caption. Rounded to 6 decimals the gap is at most
  about 2e-5.
- Not exact: logged points from about step 1000 onward. They come from each seed's
  few real episodes, so small between-configuration gaps and wide early IQRs remain.
- Curve heights after t=0 still reflect different reward functions and do not rank
  driving performance.

## Training

20 new models: A/C x SP/FD x 5 paired seeds (A 3100-3104, C 4200-4204), 100K steps each,
with frozen jobs in `jobs.json`. BAL is reused from the existing F5/BAL records.
Submission details are in `katana_submission.json`; submission is not evidence of
completed training. Run from a source snapshot on Katana:

```bash
qsub -v VENV_BUNDLE=/srv/scratch/$USER/venv_archives/highway-env.2DBNKs \
  scripts/katana_reward_equal_start.pbs
```

Each task trains, selects a checkpoint and evaluates original scenes. A model with no
checkpoint passing the eligibility gate exits non-zero, but its training logs remain.

## Results

All 20 new models completed 100K steps; with the 10 reused BAL runs, all 30 selected the
100K checkpoint, have 36 of 36 onsets and 0 collisions in development evaluation.

Median of 5 seed-level conditional median onsets (Arm C: actual lane change), with range:

| Arm | FD | BAL | SP |
|---|---:|---:|---:|
| A | 1.2 s (0.0-2.7) | 2.7 s (1.6-3.6) | 2.6 s (2.3-3.1) |
| C | 8.5 s (8.0-9.4) | 6.5 s (5.6-8.7) | 5.8 s (5.8-7.9) |

Median logged return at 99K steps:

| Arm | FD | BAL | SP |
|---|---:|---:|---:|
| A | 96.5 | 94.9 | 100.4 |
| C | 92.8 | 100.1 | 111.7 |

Descriptive only: five seeds, development scenes, no inference. Returns are not comparable
across configurations because the reward functions differ.

- The first logged point sits above x=0 in Arm A (median about 71-75 at 1K steps, against
  59.14). The value at 1K steps rests on about 10 episodes per seed, whose mean has a
  standard error near 15 in Arm A, so this is mostly sampling noise rather than learning.
- Arm A FD seed 3100 has a median onset of 0.0 s because 61% of its episodes slow on the
  first step. This is real policy behaviour, not a data error.

```bash
.venv/bin/python experiments/reward_equal_start/collect_results.py
Rscript experiments/reward_equal_start/plot_training_curves.R
```

## Reproduce

```bash
# Check the decomposition against a real run's monitor.csv.
.venv/bin/python experiments/reward_equal_start/calibrate.py replay \
  --arm C --seed 4200 --variant F9 --speed 1.12 --spacing 0.28
# Build the reference set (about 11 minutes); writes to outputs/reward_equal_start/.
.venv/bin/python experiments/reward_equal_start/calibrate.py reference --episodes 2000
```

Checked by replay on 9 run/weight combinations across both arms (episode length and
return match monitor.csv over the first 8 episodes), and on scenario labels against
`training_scenarios.csv`. Arm A's training code resets once before building the model
and Arm C's does not; the script reproduces that difference.
