# Fixed-weight validation of onset gap and hazard controls

Frozen selection: FD=F1 (0.28,1.12), BAL=F4 (0.56,0.84), SP=F7 (0.98,0.42).
Use exactly the same three pairs in A/B/C; do not reselect based on validation
results. This checks sensitivity to replacing the original preference weights.
Reward values are reward-function-specific, not common driving-quality scores.

## Design

- A: seeds 3100–3119, single-lane slowdown.
- B: seeds 4100–4119, two lanes, empty target lane.
- C: seeds 4200–4219, two lanes, occupied target lane.
- Each arm/configuration has 20 training seeds. Match the original fixed_100k
  protocol: 100K steps, 5K checkpoint interval, 20-second training episodes,
  120-second evaluation, 36 development exposures, existing checkpoint gates,
  reward costs, stratified reset schedule and learning algorithm.
- Reuse the first five fixed-sum models per arm/configuration (45 total), after
  checking frozen manifests, actual weights/protocol and eligible selected models.
- Train 135 remaining models in separate `reward_selected_validation_arm*`
  folders. Existing folders are refused; no original model is overwritten.
- Run four standalone full-rollout controls on all 180 selected models:
  original, no-front, matched-speed-front, far-front. No checkpoint reselection.
  A uses slowdown onset; B/C use actual physical lane crossing, not action onset.
- This is development sensitivity analysis, not independent held-out confirmation.
  The first five seeds informed weight selection. Report reused and additional
  seeds transparently. The existing checkpoint gate also uses hazard controls;
  standalone rollouts do not make those controls unseen validation data.

## Task mapping

| Arm | F1 (FD) | F4 (BAL) | F7 (SP) |
|---|---|---|---|
| A | 0–19 | 20–39 | 40–59 |
| B | 60–79 | 80–99 | 100–119 |
| C | 120–139 | 140–159 | 160–179 |

Each block uses ascending seeds. Its first five tasks reuse models and only
run controls; the remaining 15 train then run controls. The full 180-task
manifest is `jobs_selected_validation.json`.

## Katana execution

Sync this runner/manifest/collector, both new PBS files, existing experiment
code, and the updated `scripts/katana_local_env.py` (including bounded space
waiting), plus headless helpers. Reuse the unchanged environment snapshot.
The PBS files request 4 CPUs / 16 GB / six hours per task, and use node-local
extraction, a checksum-keyed shared environment, preparation-only node locking,
30-minute space waiting and logged checks. Each job keeps private runtime caches;
training and evaluation do not hold the preparation lock.

```bash
export VENV_BUNDLE=/srv/scratch/$USER/venv_archives/highway-env.2DBNKs
qsub -v VENV_BUNDLE scripts/katana_selected_validation.pbs
```

If training completed but standalone controls failed, rerun evaluation only
using that validation task index, without deleting or retraining the model:

```bash
qsub -v VENV_BUNDLE,VALIDATION_INDEX=5,VALIDATION_PHASE=evaluate \
  scripts/katana_selected_validation_retry.pbs
```

For failure before a new run folder exists, omit `VALIDATION_PHASE=evaluate`
to train and evaluate. If an incomplete training folder exists, inspect/archive
it first; never remove completed model folders. An existing reused task defaults
to evaluation only even in `complete` phase.

## Download and collect

New models are in `reward_selected_validation_arm*`; the first 45 models and
their new controls remain in their original fixed-sum directories. Download
both directory families, not just the new one. Existing baseline folders for
all 20 seeds are needed to verify matched training reset schedules.

```bash
.venv/bin/python experiments/reward_sensitivity/collect_selected_validation.py
```

Expected: 180/180 original-scene records and 180/180 standalone control sets.
The collector checks manifests, selection, metadata, reset prefixes and matched
exposure identifiers. It exports `onset_states.csv`, retaining absent onset,
onset at t=0, collisions and finite/nonfinite gaps, alongside standard per-seed
and counterfactual summaries. Without `--allow-incomplete`, missing runs/controls
fail collection. `figures_ready` is true only at full coverage.

## Two figure outputs after completion

1. Original-scene onset gap: one seed median per agent; omit left-censored
   (t=0) and unobserved onsets from conditional gap estimates, retain counts.
   Compute agent contrasts within matched exposure first, then the seed median;
   bootstrap whole seeds for 95% intervals, sign test excluding ties. Pair by
   exposure ID/seed/rollout metadata, not agent-prefixed episode IDs.
2. Hazard controls: per-seed median onset time under original/far-front/
   matched-speed-front, bootstrap intervals over seeds. Display no-front onset
   counts and all denominators/censoring explicitly. Never copy prior figure
   conclusions or substitute 120 seconds for missing onset.

Validation setup: 45 existing models passed reuse verification; unit tests and
local incomplete collection check are software validation, not new training or
behavioral results. No cluster job has been submitted from this workstation.


## 2026-10-18 — Shared environment and 53-task recovery

Audit of array 9543211 (180 logs) found 127 successes, 52 three-hour node-lock
wait timeouts, and one 30-minute space wait timeout (index 42 on k137, 0.81 GiB
available versus 1.65 GiB needed). All 53 failures occurred before experiment
startup: 39 new training tasks and 14 reused-model evaluation tasks. Frozen
indices and failure types are recorded in `recovery_9543211.json`.

The previous full-job lock prevented simultaneous imports/training even when
the node had enough CPU and memory. The selected-validation PBS files now pass
`--shared-env`. Under the preparation lock, the helper validates the archive
checksum, extracts once, and atomically publishes a per-user, per-node cache
keyed by the snapshot SHA256. Cache hits use the published ready marker and
skip re-extraction. The preparation lock is released before headless preflight,
training and evaluation. Thus the scheduler's allocated tasks run concurrently.

Each task still gets its own TMPDIR, fontconfig and runtime caches; Python
bytecode writes are disabled for shared-environment children. Child exit or
failure cleans only that task's private directory. The verified environment is
retained for later jobs, not automatically deleted while another job may use
it. Do not modify packages in the shared cache. It remains until node cleanup
or deliberate removal when no jobs use it. Different snapshot hashes use
separate caches; capacity remains bounded by the node filesystem. Normal
staging failures remove their unpublished temporary directory; hard kill/node
failure may leave an unpublished staging directory, which is never reused.
Do not manually remove cache directories while jobs are active.

Only this validation array and its retry/recovery entry points opt into shared
mode. Other existing PBS scripts retain legacy behavior. Avoid running legacy
and shared environment jobs on the same node concurrently because they use
the same preparation lock. Space checking/waiting and package-origin preflight
remain in place. This fixes training serialization, not unrelated consumption
of node disk space: k137-like shortages can still time out and will be logged.

After syncing the updated helper, both validation PBS files, and the new
recovery PBS, reuse the unchanged snapshot and submit only:

```bash
export VENV_BUNDLE=/srv/scratch/$USER/venv_archives/highway-env.2DBNKs
qsub -v VENV_BUNDLE scripts/katana_selected_validation_recover.pbs
```

Recovery slots 0–52 map to original validation indices. The log prints both.
The reused-model jobs automatically evaluate only; new jobs train then evaluate.
The runner's existing-directory protection remains, so a completed model is
never silently retrained. Do not rerun the full 180-task array.

Validation: 36 tests passed, including two real concurrent child processes
sharing a single extraction while keeping separate runtime directories,
cleanup after child/extraction failure, recovery mapping and existing guards.
PBS syntax and diff whitespace checks passed. No cluster submission performed.
