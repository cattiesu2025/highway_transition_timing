# C/R6 risk-penalty pilot

Two new reward ablations with a freshly trained paired baseline. This is a
separate exploratory pilot, not a replacement of maintained experiments.

| Mode | Closing-risk component (closing speed c > 0.1 m/s) |
|---|---|
| baseline | max(0, 1 - TTC/6), using original implementation exactly |
| early30 | max(0, 1 - TTC/30) |
| smooth6 | 6*c/(distance + 6*c) = 6/(TTC+6) |

All modes keep the original close-distance component clip((24-distance)/24,0,1).
The risk score is the maximum of closing-risk and close-distance components.
No front vehicle gives zero risk. Risk coefficient remains 3. Smooth6 removes
the TTC cutoff, but retains the original c > 0.1 tolerance and close-distance
protection; it does not penalize safe, nonclosing traffic at large distances.
Thirty seconds and the smooth scale of six seconds are fixed pilot choices,
not optimized thresholds. Changing the shape also changes penalty magnitude;
this is not a matched-penalty-budget comparison.

C/R6 speed/spacing weights 1.03/0.48; seeds 4201 and 4202; 100K training steps;
20 s episodes; gamma .95; same scene generator, optimizer, architecture,
collision penalty, action costs and 5 Hz decisions. Reusing seeds pairs random
initialization and reset sequence prefixes, not necessarily every transition or
full scene counts: different actions can change episode lengths.

Primary evaluation is the fixed final 100K checkpoint for every mode. This
holds checkpoint age constant. The original eligibility gate is audited and
reported, but does not replace the checkpoint or discard failed runs. These
results must not be presented as the maintained selected-policy protocol.
Four development controls (original/no-front/matched-speed-front/far-front),
36 exposures each, 120 s. No held-out data is opened. Physical lane-change
onset, collision, and absent-onset results must all be retained. Only two
seeds: feasibility/diagnosis, not a confirmatory significance claim.

The isolated runner adapts a privately loaded experiment module in its process;
shared source defaults are unchanged. Each run has its own exclusive output
directory and refuses overwrite. Use this runner for the pilot, not the generic
reward_sensitivity collector or generic re-evaluation scripts, which do not
know these risk rules. protocol.json/config.json describe the full rule;
training_runs.csv alone is not sufficient to identify it.

## Run

From the project root (the portable bundle is also a project root):

```bash
python3 experiments/risk_penalty_ablation/run.py --index 0
qsub scripts/katana_risk_penalty_ablation.pbs
```

Indices 0/1 baseline seeds4201/4202; 2/3 early30; 4/5 smooth6.
PBS runs both training and control evaluation. Outputs are
`/srv/scratch/$USER/highway_transition_timing/outputs/risk_penalty_armC_R6_*`.
The existing `/srv/scratch/$USER/venvs/highway-transition` environment is used.

After local synchronization:

```bash
python3 experiments/risk_penalty_ablation/collect.py
```

Each run stores compressed per-step training risk logs with actual reward
contribution, original-rule risk on the same visited state, scenario identity,
collision and termination flags. This measures historical training activity
for these new runs; it is not a reconstruction of the previous training.
control_summary.csv and per-control steps/episode summaries retain evaluations.
Source hashes and package versions are recorded. complete.json is written last.

## Local wiring verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_risk_penalty_ablation.py tests/test_highway_adapter.py
.venv/bin/python experiments/risk_penalty_ablation/run.py --index 0 --smoke --execute --out-root /tmp/risk-pilot-smoke
```

Smoke mode uses 200 steps and three four-second evaluations per control. It
checks save/reload, actual reward wiring, logging, and evaluation only. It is
not evidence of learned safety or onset separation.
