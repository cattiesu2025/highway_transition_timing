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
