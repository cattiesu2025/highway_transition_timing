# 20-seed equal-start rerun: results (2026-11-14)

Runs: 400 jobs (`jobs_20seed.json`, Katana arrays 9760987[] and 9761720[]), all trained to 100k steps.
Copied to `outputs/`. A size-only dry-run against Katana shows no differences.
Pipeline: `collect_20seed.py` → `plot_training_curves_20seed.R {es,fl}`, `plot_onset_20seed.R {es,fl}`.
Figures and source data: `outputs/reward_equal_start/figures_20seed/`.

## Model selection
237 of 240 runs per design have a selected, eligible checkpoint. The 3 failures in each design are all Arm D.
In every one, the no-front gate failed: the policy changes lanes with no front vehicle. They match the earlier 5-seed failures.

- **ES:** D SP 4204, 4207, 4209
- **FL:** D SP 4204, 4207; D FD 4216

The failed runs are not replaced. Their training curves are still included.

## Start levels (first SB3 log, mean of 20 seeds)
| Arm | ES: SP / BAL / FD | FL: SP / BAL / FD |
|---|---|---|
| A | 69.5 / 67.8 / 66.5 | 67.8 for all three |
| B | 61.8 / 61.7 / 61.6 | 61.7 for all three |
| C | 55.6 / 56.4 / 57.0 | 56.4 for all three |
| D | 53.2 / 54.1 / 54.8 | 54.1 for all three |

- Under t=0 weights, the first-log gap is at most 3 reward units, well inside the seed CI (about ±6 to ±13).
- Under FL weights, the starts are equal by construction.

## Onset (median of seed medians, 95% bootstrap CI over seeds)
| Arm | Design | FD | BAL | SP |
|---|---|---|---|---|
| A (slowdown) | ES | 2.0 [1.3, 2.4] | 2.5 [2.35, 2.6] | 2.75 [2.55, 2.95] |
| A | FL | 1.8 [1.65, 2.25] | 2.5 [2.35, 2.6] | 2.7 [2.55, 2.8] |
| B (lane change) | ES | 2.3 [2.05, 2.5] | 2.5 [1.75, 3.1] | 3.15 [2.4, 3.4] |
| B | FL | 2.4 [2.2, 2.75] | 2.5 [1.75, 3.1] | 3.4 [2.55, 3.85] |
| C (lane change) | ES | 8.85 [8.4, 9.4] | 7.9 [6.55, 9.4] | 6.45 [6.1, 7.05] |
| C | FL | 9.15 [7.55, 9.4] | 7.9 [6.55, 9.4] | 6.5 [6.0, 7.4] |
| D (lane change) | ES | 1.1 [0.75, 1.3] | 0.75 [0.6, 1.0] | 0.6 [0.6, 1.0], n=17 |
| D | FL | 0.6 [0.6, 1.6], n=19 | 0.75 [0.6, 1.0] | 1.05 [0.6, 1.65], n=18 |

Every shown episode has an onset.
- **Collisions:** 0 under ES; 1 under FL (Arm C SP, one episode).

## Reading
- **Both designs give the same ordering in A, B and C.** Changing how SP/FD are equalised (t=0 vs first log) moves the medians by at most 0.3 s.
  - **Arm A:** FD < BAL < SP. FD vs SP CIs are separated in both designs.
  - **Arm B:** SP is latest. FD and BAL overlap because BAL's CI is wide (1.75–3.1).
  - **Arm C:** reversed, FD > BAL > SP. SP vs FD CIs are separated. Many seeds sit at 9.4 s, which looks like a scene-determined ceiling, so FD/BAL upper bounds are censored by it.
- **Arm D has no stable ordering.** Onsets sit at the 0.6 s floor (first possible lane change), and the FD/SP order flips between ES and FL. This arm is also where all selection failures occur.
- **Training curves:** in B/C/D the plateaus are ordered SP > BAL > FD; in A, BAL is lowest. The rewards differ, so plateau height does not rank driving.
