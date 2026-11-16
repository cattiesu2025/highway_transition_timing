# Development log

## 2026-10-26 — Independent distance-only implementation

Implemented `run.py` with an isolated original C module and deterministic
`distance-affine-v1` geometry transform. Original A/B/C implementation files
and outputs were not edited. Source speeds, scene presence, seed/reset pairing
and family quotas are preserved; original boundary/difficulty labels are
retained as source metadata and actual difficulty is recalculated.

Added dry-run/audit/smoke/train modes, output overwrite protection, protocol and
source hashes, failed-run status, actual vehicle gap logging, inherited latest
eligible checkpoint selection, and post-selection control rollouts. Added a
single-job Katana launcher; no job was submitted and no full training performed.

Validation:

- 38 tests passed, including original multi-lane and counterfactual regressions,
  scene invariance, inherited control semantics, failure-gate behavior and
  output overwrite protection.
- 2,000 paired resets checked in the final smoke audit.
- 57 scenes passed initial observation checks and one-second scripted lane
  changes without collision. This is not a learned-policy safety result.
- Final BAL/F4 seed 4200 smoke completed 1,200 steps, model reload, one 120-second
  horizon evaluation and analysis. Scientific-result flag remains false. The
  inherited trainer uses 120 learning-start steps in this short smoke and
  1,000 in a full run.
- Final verification artifact: `/tmp/arm_c_distance_final_check/arm_c_distance-affine-v1_F4_100k_seed4200_smoke`.
  Its recorded source hashes matched the final implementation.
- An earlier smoke failed on a status dictionary update after preflight. Fixed
  it, added orchestration coverage, and retained the failed output separately.
- All 60 matching C/F1,F4,F7 baseline directories passed configuration and
  eligible-selection metadata checks locally. This check does not re-audit
  old model binaries or create new independent baseline seeds.
- Katana script passed `bash -n`.

Geometry is a transparent development choice: rank-preserving distance
transformation, not the broad-v2 uniform sampler. New held-out scenes and a
formal 20-seed experiment have not been run or claimed. The inherited gate may
reject all checkpoints in the new geometry; this remains a reported failure,
not permission to select a final checkpoint or retune the gate.

## 2026-10-26 — Authorized Katana pilot submitted

User authorized remote execution using configured SSH. Submitted array
`9615815[].kman.restech.unsw.edu.au`: seeds 4200–4202 × FD/F1, BAL/F4, SP/F7,
9 new models at 100k steps each. Latest observed state: all nine queued in
`cse12`, no running subjobs yet. Submission is not evidence of completed training.

Uploaded a standalone source snapshot to
`/srv/scratch/z5535967/highway_transition_timing/run_snapshots/arm_c_distance_pilot_20261026`;
verified all 26 manifest file hashes and nine unused output paths before qsub.
Original A/B/C code and outputs were not overwritten. Uses the existing
`highway-env.2DBNKs` environment bundle through the established node-local
staging helper. Launcher logs will be written to
`outputs/arm_c_distance_launch_logs/` on Katana when tasks start.

Submission mapping, source/archive hashes and paths are recorded in
`katana_submission.json`. PBS launcher syntax checked. No 20-seed expansion
was submitted. Check with `ssh katana 'qstat -t -u z5535967'`.

## 2026-10-26 — Remaining 51 models submitted at user request

The user clarified that the intended scope was 20 seeds per preference, and
explicitly requested submission of the remaining 51 models. Submitted array
`9615837[].kman.restech.unsw.edu.au`, indices 0–16 FD, 17–33 BAL, 34–50 SP;
each group maps to seeds 4203–4219. Retained original array `9615815[]`, which
was observed with all nine subjobs running at the time of the additional submission.
The two arrays cover all 60 (agent, seed) pairs without overlap.

Reused the original remote source snapshot and environment bundle. All original
26 source manifest hashes were verified unchanged, and all 51 additional output
paths were unused before submission. Only a new array launcher was added;
training configuration and original A/B/C remain unchanged. Script syntax and
complete/non-overlapping seed mapping were checked. Exact mapping is in
`katana_submission.json`. No claim of completed training or results yet.

Post-submission verification: first array Running:9 with gradient-update logs
for every model; additional array Queued:51 Running:0.

## Final execution status check

All jobs exited; 60 models reached 100k training steps. 58 runs completed selection and evaluation (FD 19, BAL 19, SP 20); completed metadata and all five sets of 36 episode outcomes passed presence/count checks. FD/4209 and BAL/4205 failed selection after all 20 checkpoints were checked. FD failed no-front specificity at every checkpoint (20/20), and matched-speed specificity at 8/20. BAL failed no-front specificity at 13/20 and matched-speed specificity at 14/20. These are protocol eligibility failures, not incomplete training. No fallback, gate change, or rerun performed.
