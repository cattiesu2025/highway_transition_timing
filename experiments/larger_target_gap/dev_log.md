# Pilot development log

## Implementation

- Added isolated runner: linked geometry, 60/20/20 slow/no-front/matched reset blocks, seeds 5200–5202, final 100k checkpoint, 27 development scenes and four paired evaluation variants.
- Added fixed-identity original/target vehicle diagnostics before and after every evaluation step, and geometry/visibility/scripted-action preflight.
- Validation pending; no pilot training has been submitted.

## Running

Use `run.py --index 0 --execute --preflight-only` for geometry checks and `--smoke` for 200-step integration only. Real 100k pilot tasks run on Katana. Output directories refuse overwrite.

## Validation update

- Initial preflight: all 28 initial scenes observed both front vehicles; left-once scripted policies changed lanes without collision in 28/28, while IDLE collided in 28/28 within 20 seconds.
- Initial tests: 12 passed (new pilot + original occupied-lane regression).
- FD/5200 smoke: 200 training steps and 12 evaluation episodes completed; not a scientific pilot result.
- Added collector with explicit missing-job status and PBS 0–8 array. PBS syntax check passed.
- Review correction: absent or passed front vehicles are excluded from the two-front-vehicle space comparison; raw signed geometry remains logged. Added position-clipping check and actual perception-distance metadata. Final validation pending.
- Katana submission is pending connection details; no SSH configuration was found in this environment. User has been asked for login address/alias while local work continues.

## Final local validation

- Final tests after review: 12 passed in 1.69 s; Python compilation passed.
- BAL/5200 final-source smoke: 200 steps, 12 episodes across all four variants, complete marker present. Removed-vehicle comparisons correctly remain blank.
- Final preflight: all position observations unclipped; runtime perception distance 200 m. All 28 scripted left-once cases passed.
- Collector smoke check correctly reports 0/9 real pilot jobs complete and does not count the two short smoke runs.
- Final status: implementation and local validation complete; 9-model Katana pilot not submitted because login connection details are unavailable. No model-quality or ranking conclusion is supported yet.

## 2026-10-24 — Downloaded pilot audit

User completed Katana runs and downloaded all nine real outputs. Verified final checkpoint timesteps, provenance hashes, 972 full evaluation episodes and 97,200 steps. Collector reports 9/9 complete. Main scenes: all 243 episodes command LANE_LEFT at t=0 and physically change lane at 0.6 s, with more target-lane space and zero collisions. No-front controls never change lane; matched-speed controls all change lane. Results and limitations documented in `PILOT_RESULTS.md`; no additional training initiated.

## 2026-10-24 — Fixed-action timing audit completed

Implemented `audit_timing.py` and ran the agreed 24-scene distance grid locally, without training: 120 trajectories, 360 reward evaluations. Verified identical paired initial states, visible/unclipped vehicles, per-step reward reconstruction and independent discounted-total reconstruction. White-car distance 60–120 m favours immediate changes; 150 m favours 1–2 s waits, with small FD/BAL versus SP differences at 14 m/s. Waiting benefit at 1 s is solely discounted action-cost deferral. Full protocol, limitations and results in `TIMING_AUDIT.md`; raw artifacts in `outputs/larger_target_gap_timing_audit_v1`. No training or reward settings changed.

## 2026-10-24 — Broad v2 retraining preparation

User authorized retraining. Added explicit broad-v2 profile with uniform white-car distance 40–150 m and conditional extra gap 30–min(80,195−D_o), preserving initial yellow-car visibility. Updated collector for 36-scene grid and protocol checks, added dedicated PBS and frozen `BROAD_V2.md`. Existing default v1 geometry retained. Validation pending. SSH host reachable outside network sandbox, but noninteractive authentication failed; no jobs submitted at this point.

### Broad v2 validation and submission handoff

- 14 tests passed, including v1 occupied-lane regression, broad distribution and 195 m visibility checks. Python compile and PBS syntax passed.
- FD/5200 broad-v2 200-step smoke completed with 12 control evaluation episodes. Preflight covered 36 grid scenes + tight training corner, two scripted policies each (74 trajectories); all 37 immediate lane changes were collision-free, initial positions unclipped.
- Collector now writes `by_distance.csv`; regression collection of all nine downloaded v1 jobs exactly reproduced the previous `summary.csv`. Broad-v2 collector correctly reports 0/9 real jobs so far, excluding smoke.
- User elected to submit from an already logged-in Katana terminal. No real retraining submitted by this assistant; dedicated command is `qsub scripts/katana_larger_target_gap_broad_v2.pbs` after updating code.
