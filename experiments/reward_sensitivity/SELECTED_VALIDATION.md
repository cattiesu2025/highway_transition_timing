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
extraction, same-user node locking, 30-minute space waiting and logged checks.

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
