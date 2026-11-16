# Arm C distance variant

Independent extension of Arm C using F1/F4/F7. Existing A/B/C source files,
models and results are unchanged. This is a development implementation, not a
completed experiment or a newly sealed held-out protocol.

## Distance-only mapping (`distance-affine-v1`)

Generate the original C scene first, including its accepted speeds and target
vehicle distance. Then transform only its two distances:

- `D_o_new = 40 + (D_o_old - 90) * 110 / 105`.
- `u = (D_t_old - 10) / 80`.
- `extra = 30 + u * (min(80, 195 - D_o_new) - 30)`.
- `D_t_new = D_o_new + extra`.

Distances are rounded to millimetres. This maps the original white-car support
90–195 m into 40–150 m and preserves its rank. It is **not uniform sampling** of
40–150 m and is not the broad-v2 pilot sampler. Target distance is always at
least 30 m greater than white-car distance and no more than 195 m.

For each seed/reset index, ego speed, white-car speed, vehicle presence, source
family, environment seed and the 20-reset family quotas remain identical.
No-front training contains neither white nor yellow traffic vehicles. The
source boundary family maps from 180–195 m to approximately 134.286–150 m;
its name is a source-family label, not a claim of current observation-boundary
placement. Actual TTC/deceleration bins are recomputed and source bins are
logged separately. Different episode lengths can still change the number of
resets visited; pairing is by reset index, not training step.

The 36 development scenes use exactly the same transform on the original C
grid. White-car distances become 102.857/118.571/134.286/150 m. Ego is 28 m/s,
white car is 10/14/18 m/s, yellow car remains 29 m/s. This grid is paired with
the old C grid; it is not the pilot's 45–150 m grid. No claim is made that old
and new geometry will give stable FD/BAL/SP timing differences.

## Inherited settings

FD/F1=(0.28,1.12), BAL/F4=(0.56,0.84), SP/F7=(0.98,0.42), in
(speed, front spacing) order. The original C `ExperimentConfig`, environment,
training algorithm and gate implementation are reused in an isolated module:
100k training steps, 20-second training episodes, 120-second evaluation,
5/15 Hz decision/simulation frequencies, checkpoints every 5k steps, latest
eligible checkpoint selection. No fallback to the final model if gates fail.

`source_scenario_type` and `source_*_bin` document the original stratification;
new `ttc_bin` and `required_deceleration_bin` describe the transformed scene.
`geometry_profile` identifies the new sampling distribution in every reset
record, even though the inherited sampler is still labelled `stratified`.

The inherited no-front evaluation removes only the white vehicle and retains
the yellow vehicle; far-front moves the white car to 360 m. These controls
need not satisfy the main-scene ordering and are not new training families.
Original, no-front, matched-speed-front and far-front control rollouts run after
selection, using the selected checkpoint with no reselection. `lg_*` and
`post_lg_*` fields retain fixed-vehicle gaps, speeds and visibility around each
step, including the first command and actual lane change.

## Local checks

All commands are dry runs without `--execute`. Existing output directories are
refused, including incomplete runs. Use a fresh `--out-root` when repeating a
smoke; do not erase failed runs or present smoke output as scientific evidence.

```bash
.venv/bin/python experiments/arm_c_distance_variant/run.py --agent BAL --seed 4200
.venv/bin/python experiments/arm_c_distance_variant/run.py --mode audit --execute
.venv/bin/python experiments/arm_c_distance_variant/run.py --mode smoke --execute
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_arm_c_distance_variant.py
```

Audit verifies 2,000 paired resets, the 36-scene manifest, actual initial
observations, and 57 one-second scripted left-lane changes (including a tight
corner). This is a geometry/implementation check, not a safety guarantee for
learned policies. Smoke uses 1,200 training steps (the inherited trainer scales learning starts to
120 for this smoke; full 100k training uses 1,000), model reload and one full-horizon evaluation. Smoke
skips scientific model selection and is explicitly marked non-scientific.

## One independent training job

```bash
python3 experiments/arm_c_distance_variant/run.py \
  --mode train --agent BAL --seed 4200 --execute
```

On Katana, the single-job launcher accepts `DISTANCE_AGENT` and `DISTANCE_SEED`:

```bash
qsub -v VENV_BUNDLE=/srv/scratch/$USER/venv_archives/highway-env.2DBNKs,DISTANCE_AGENT=BAL,DISTANCE_SEED=4200 scripts/katana_arm_c_distance_variant.pbs
```

The user authorized Katana execution on 2026-10-26. The dedicated
`katana_arm_c_distance_pilot.pbs` array maps indices 0–2 to FD, 3–5 to BAL,
6–8 to SP, each with seeds 4200–4202. Submission status is recorded in
`katana_submission.json` and `dev_log.md`. This array contains only new
C-distance models; it does not submit A/B/C baseline jobs.
The user requested the full 20-seed experiment on 2026-10-26. The original
9 jobs are retained in array `9615815[]`; the remaining 51 jobs (seeds
4203–4219 × FD/BAL/SP) were submitted in array `9615837[]` with
`katana_arm_c_distance_remaining.pbs`. Together they cover all 60 new
C-distance models exactly once, using the same original source snapshot and
environment bundle. Submission does not mean training has completed.

`protocol.json` stores configuration, code hashes, requested mode and baseline
metadata status. `status.json` records complete/failed and whether this is a
scientific run. A provisional `*_main.zip` in a failed run is not an eligible
model. Baseline lookup uses matching F1/F4/F7 C selected-validation runs (first
five seeds from the fixed-sum sweep). `metadata_verified` means configuration
and selected-checkpoint metadata matched; it is not a complete re-audit of old
model binaries. Missing/incompatible baselines are reported, not retrained.

The primary comparison with old C changes its training and evaluation geometry
together. Separating training-distribution effects from evaluation-scene effects
requires a separate crossed evaluation; this runner does not claim to do that.
