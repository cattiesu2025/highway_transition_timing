# A–D clip/onset coverage extension

Requested 2026-10-27: supplement DISAGREEMENTS coverage and include Arm D.
The user confirmed that D means the independent Arm C distance-affine-v1
experiment, and explicitly authorized uploading the implementation and running
the inference-only comparison on Katana.

This is an extension of the historical P0 **per-seed diagnostic**, not the
pooled-corpus human-study protocol in `preregistration_v2.md`. Nothing is trained,
the sealed held-out evaluation is not used, and no human-study claim is scored.

## Corpus and accounting

- A: original single-lane models, seeds 3100–3119.
- B: original empty-target-lane models, seeds 4100–4119.
- C: original occupied-target-lane models, seeds 4200–4219.
- D: new distance-affine-v1 models, seeds 4200–4219, FD/F1, BAL/F4, SP/F7.
  FD/4209 and BAL/4205 failed model selection and remain explicitly ineligible.
  A provisional model file from a failed run is never used.

Each eligible model supplies the same 36 development exposures as its original
evaluation. D uses its transformed development grid. D also has its own reward
weights, so differences against these historical A/B/C models cannot be
attributed solely to geometry.

The endpoint is exactly the stored `target_onset_t` used by P0. In A this is
the persistent slowdown onset. In B/C/D it is the earliest direction-matching
lane-change command in the window preceding a realised lane-index change
(or the confirmation step if no such command is recorded). It is **confirmed
by** physical lane crossing, but is not the crossing/completion timestamp.
Consequently an onset at step zero is possible and is retained. Earlier
informal descriptions calling this simply "physical lane-change onset" should
be read with this distinction.

The expected model inventory is 60 each for A/B/C and 58/60 for D. HIGHLIGHTS
has up to 300 selected clips per arm/variant (D: 290). DISAGREEMENTS has six
directed policy pairs per seed: up to 600 clips per arm (D: 560, from 112/120
eligible directed pairs). Counts are actual selected clips; fewer candidates
are not padded with invented clips. No-onset episodes stay in the denominator.

## Selection and coverage

HIGHLIGHTS retains both `max_min` and `max_second`, budget 5 per seed/leader,
the existing stable tie rule and spacing constraint. These settings reproduce
the P0 A/B/C coverage counts: 14/28, 129/94, 73/86, respectively, each out of
300 selected clips (max-min/max-second).

DISAGREEMENTS compares policies within a training seed, in both leader
directions. At every leader state with different greedy actions it forks the
simulator and runs the follower for up to ten steps, stopping on termination.
Its native context is `[max(t-9,0), t+10]`, shortened by termination. Its
importance is the reference `last_state`: absolute difference between the
two terminal states' summed policy values, with per-policy, per-trace min-max
normalization and agent ratio 1. It retains the upstream image-equality zero
guard, overlap limit 5 against the union of previously selected indices,
stable ties, and discarded-candidate fallback. The leader's terminal frame is
retained for scoring, as in the reference implementation. If one trajectory
ends sooner, endpoint indices are aligned. Selection is over the whole
36-exposure corpus for each seed/leader/follower, budget 5.

The **primary coverage metric** uses the exact historical P0 common window:
`range(max(t-10,0), min(t+10,L-1))`, where L is the number of stored policy
steps. This preserves its right-open terminal clamp rather than silently
changing the published diagnostic. Native DISAGREEMENTS window coverage is
exported separately as a sensitivity result. At interior states the two
20-frame windows differ by one frame; they must not be conflated.

Coverage tests only the onset of the real **leader episode**. It does not assign
that onset to the synthetic follower branch, or claim to measure both sides of
the paired video. The result is a selection/alignment diagnostic, not IoU or a
ranking of explanation quality. ONSET containment is true by construction and
does not constitute evidence that ONSET is superior.

## Replay and audit

Replay must match stored ego position, speed, lane and episode length. Stored
action labels are also checked; only an argmax flip with Q difference at most
1e-5 is admitted, logged, and replayed using the stored action. Other deviations
fail the run. The common reward wrapper can be omitted during simulation
because it only recomputes unused rewards; the actual simulator is cloned.

The runner saves input/model/source SHA256 manifests and refuses to resume if
they change. Per-episode candidate caches allow interruption recovery. Final
aggregation accepts only 36-exposure completed, audited runs; pending,
partial, and ineligible groups remain visible. HIGHLIGHTS can be reported from
stored tables before counterfactual processing finishes, labelled provisional.

`paired_by_seed.csv` repeats each HIGHLIGHTS selection for each eligible
directed pair, providing matched seed/leader support for comparison with
DISAGREEMENTS, especially for D's missing models. Native and common coverage,
no-onset counts, candidate shortages and replay ties are retained in outputs.

## Execution

```bash
PYTHONPATH=src .venv/bin/python scripts/clip_onset_coverage.py \
  --arm D --seed 4200 --leader FD --outputs-root outputs --out outputs/clip_coverage
PYTHONPATH=src .venv/bin/python scripts/aggregate_clip_onset_coverage.py outputs/clip_coverage
```

The Katana launcher maps 80 array indices to A–D × 20 seeds, running the three
leaders as three CPU processes per job. Each process evaluates only the
available policies. Smoke runs use a separate output directory and cannot be
aggregated as full results.
