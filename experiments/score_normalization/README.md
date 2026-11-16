# Score-normalisation pilot (SN1)

Exploratory pilot requested after advisor review: the speed and front-distance scores
were normalised by hand-set constants, and the distance score sat at 1 most of the time.
This is a separate pilot, not a replacement of maintained experiments.

| Term | Maintained | SN1 | Basis for SN1 |
|---|---|---|---|
| speed_score | clip((v − 20) / 15, 0, 1) | clip((v − 10) / 25, 0, 1) | action target-speed range 10–35 m/s |
| front_distance_score | clip(d / 120, 0, 1), 1 if no front | clip(d / 200, 0, 1), 1 if no front | highway-env `PERCEPTION_DISTANCE` = 200 m |

A front vehicle beyond 200 m is absent from the Kinematics observation, so "no front" and
"front beyond 200 m" both score 1 and the reward never separates states the policy cannot.
highway-env has no default front-distance reward; the 200 m horizon is this project's choice.

## What changes and what does not

- Both slopes shrink by exactly 0.6 (15/25 and 120/200). The speed-for-distance exchange
  rate is unchanged: 1 m/s is worth 8 m under both definitions.
- Preference weights are multiplied by 1/0.6, so BAL is (1.166667, 1.166667). Collision (2),
  risk (3) and the arm's action costs keep their maintained relative strength.
- What differs is where the scores saturate: distance now varies over 120–200 m, and speed
  varies below 20 m/s.

## Calibration (random policy, before learning starts)

Per-step means on 2000 reference episodes per arm (seeds 9100 for A, 9200 for B–D):

| Arm | S maintained | D maintained | D:S | S SN1 | D SN1 | D:S | D:S, front present |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.385 | 0.858 | 2.23 | 0.578 | 0.706 | 1.22 | 1.07 |
| B | 0.344 | 0.923 | 2.69 | 0.606 | 0.839 | 1.38 | 0.98 |
| C | 0.344 | 0.810 | 2.35 | 0.606 | 0.632 | 1.04 | 0.86 |
| D | 0.342 | 0.873 | 2.55 | 0.605 | 0.746 | 1.23 | 1.08 |

Share of steps at a clip bound:

| Arm | S = 0 maintained | S = 0 SN1 | D = 1 maintained | D = 1 SN1 | of which no front |
|---|---:|---:|---:|---:|---:|
| A | 27% | 0% | 64% | 34% | 23% |
| B | 0% | 0% | 81% | 64% | 60% |
| C | 0% | 0% | 51% | 27% | 23% |
| D | 0% | 0% | 70% | 41% | 27% |

Equal weights are therefore close to, but not exactly, equal score contributions. The
residual comes from steps with no front vehicle, which differ by arm. No data-driven
rescaling is applied to force 1:1.

Known limitation: multilane arms (B–D) only reach 20–30 m/s, so their SN1 speed score
stays within about 0.4–0.8 (per-step SD 0.13, against 0.26–0.30 for distance).

## Weights

SP and FD keep the 2:1 weight ratio and are solved so the seed-mean first logged training
reward equals BAL's, on the first 4 episodes of all 20 training seeds per arm
(`calibrate_first_log.py`; S and D under maintained scores match
`outputs/reward_equal_start/first_log_episodes_20seed.csv` on all 320 episodes).

| Arm | S per episode | D per episode | SP (speed, spacing) | FD (speed, spacing) |
|---|---:|---:|---|---|
| A | 54.390 | 65.519 | (1.605220, 0.802610) | (0.754436, 1.508872) |
| B | 54.948 | 76.209 | (1.644409, 0.822205) | (0.737906, 1.475812) |
| C | 55.741 | 59.547 | (1.572866, 0.786433) | (0.769311, 1.538622) |
| D | 50.198 | 65.052 | (1.625388, 0.812694) | (0.745738, 1.491477) |

t=0 weights from the reference sets differ from these by under 1%; both are written to
`outputs/score_normalization/calibration/weights.json` by `solve_weights.py`.

## Design

24 runs: Arms A–D × BAL/SP/FD × 2 seeds (A 3100–3101, B 4100–4101, C/D 4200–4201),
100K steps, frozen in `jobs.json`. Training, checkpoint selection and evaluation are the
maintained commands (reward-sweep command for A–C, distance-variant runner for D); only
the two scores and the preference weights differ. Paired maintained-score comparisons
already exist for the same seeds: `reward_equal20_arm{A,B,C}_{BAL,FL-SP,FL-FD}_100k_seed*`
and `arm_c_distance-affine-v1_R20-{BAL,FL-SP,FL-FD}_100k_seed*`.

Two seeds: feasibility and direction only, not a confirmatory claim. Report onset
incidence, collisions and censoring with full denominators, as in the maintained protocol.
A model with no checkpoint passing the eligibility gate exits non-zero; keep its logs.

## How the scores are swapped

`run.py install()` wraps `highway_adapter.extract_diagnostics` in memory, inside the
training process only. Shared source and maintained defaults are unchanged. Arms A–C run
the maintained script through `run.py entry <script> ...`; Arm D installs in-process.
Each run directory records the rule in `score_normalization.json`.

Do not re-evaluate SN1 models with the generic evaluation or re-scoring scripts: they do
not install the SN1 scores, so logged `speed_score`, `front_distance_score` and
`reward_total` would follow the maintained formulas. Behaviour is unaffected, because the
policy does not read the reward.

## Run

```bash
python3 experiments/score_normalization/run.py --index 0          # print only
qsub -v VENV_BUNDLE scripts/katana_score_normalization.pbs        # all 24
```

Index layout: 0–5 Arm A, 6–11 Arm B, 12–17 Arm C, 18–23 Arm D; within an arm BAL, SP, FD,
two seeds each. `--smoke` trains 1200 steps for a local check (A–C usually end at the
eligibility gate, as an untrained model should).

## Checks done

- `tests/test_score_normalization.py`: formulas, slope ratio, weight equalisation, frozen
  manifest, command construction, reversible install, rewards inside a live Arm A
  training environment.
- Local smoke runs for A (index 0), C (12) and D (18): the first training episode's logged
  return equals the replay prediction under SN1 scores (A 124.5092, C 104.4679, D 20.6863)
  and differs from the maintained-score prediction. Arm B shares the Arm C path and was
  not smoke-run separately.
- No 100K run has been started.
