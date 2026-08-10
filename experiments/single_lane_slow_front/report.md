# Single-Lane Slow-Front Sanity Experiment Report

Date: 2026-07-25

## Summary

This experiment isolates longitudinal slowdown timing in a minimal Highway
setting. The goal is to check whether reward-preference agents show interpretable
differences in when they slow down for a front vehicle, without lane-change
confounds.

The current 20K mixed-training run gives a clean sanity result:

- All agents produce valid slowdown onsets in 36/36 slow-front evaluation
  scenarios.
- The median slowdown latency follows the expected ordering:
  `FD = 0.0`, `BAL = 0.5`, `SP = 5.5`.
- The main timing gaps are positive:
  `BAL - FD = 0.5`, `SP - BAL = 4.0`, `SP - FD = 5.5`.
- Full-episode no-front counterfactuals produce no stable slowdown for any
  agent, including SP: `FD = 0/36`, `BAL = 0/36`, `SP = 0/36`.
- The semantic layer used here is transparent action-level grounding:
  `FASTER` / `IDLE` / `SLOWER`, with stable slowdown accepted only after
  persistent slowdown evidence.

This supports the interpretation that the learned policies are not simply
unconditional slowdown policies. In particular, the no-front training negative
control fixed the earlier SP issue where SP slowed immediately even without a
front vehicle.

## Experimental Setup

Environment:

- `highway-v0`
- Single lane
- Longitudinal action space only: `SLOWER`, `IDLE`, `FASTER`
- No lateral actions, so lane changes cannot explain the result
- Kinematics observation: `presence`, `x`, `y`, `vx`, `vy`
- Normalized relative observations by default

Semantic grounding:

- The rollout table records the executed low-level discrete action at each
  policy step: `FASTER`, `IDLE`, or `SLOWER`.
- For this single-lane experiment, the main target is action-level
  `slowdown_onset`, not a learned hidden concept.
- A timestep counts as slowdown evidence if either:
  - the action is `SLOWER`, or
  - ego speed drops by at least `0.35 m/s` from the previous policy step.
- The onset is accepted only when slowdown evidence persists for
  `persistence_k = 3` consecutive steps.
- Therefore, `valid slowdown onset` means the analysis found the first stable
  slowdown segment in a rollout.

Both maintained experiments define onsets this way, from observable action
semantics: `slowdown_onset` here, and `lane_change_onset` (confirmed by a
realised lane-index change) in the multi-lane setting.

Training used Double DQN with 20,000 timesteps per agent. The three agents
shared the same environment, action space, and training distribution; they
differed only in reward weights:

| Agent | Reward preference | Intended behavior |
|---|---|---|
| `FD` | High front-distance reward, lower speed reward | Maintain larger distance from the front vehicle; expected to slow earliest. |
| `BAL` | Balanced speed and front-distance rewards | Trade off speed preservation and spacing; expected to be intermediate. |
| `SP` | High speed reward, lower front-distance reward | Preserve speed longer; expected to delay slowdown. |

To avoid learning an unconditional slowdown policy, the training distribution
mixed slow-front hazard episodes with no-front cruise episodes:

| Training scene | Fraction | Purpose |
|---|---:|---|
| Slow-front episode | 80% | Learn when to respond to a slower front vehicle. |
| No-front cruise episode | 20% | Negative control: learn that slowing is unnecessary without a front vehicle. |

Observed training reset counts:

```text
FD:  train_no_front=14, train_slow_front=70
BAL: train_no_front=14, train_slow_front=70
SP:  train_no_front=16, train_slow_front=75
```

## Main Slow-Front Evaluation

The main evaluation uses 36 held-out deterministic slow-front scenarios. Each
agent is evaluated on the same scenario set.

Outcome summary:

| Agent | Valid slowdown onset | Censored | Terminal failure |
|---|---:|---:|---:|
| FD | 36/36 | 0/36 | 0/36 |
| BAL | 36/36 | 0/36 | 0/36 |
| SP | 36/36 | 0/36 | 0/36 |

Median slowdown latency:

| Agent | Median latency |
|---|---:|
| FD | 0.0 |
| BAL | 0.5 |
| SP | 5.5 |

Pairwise timing gaps:

| Gap | Median | IQR | Valid pairs |
|---|---:|---:|---:|
| BAL - FD | 0.5 | [0.0, 7.0] | 36/36 |
| SP - BAL | 4.0 | [0.0, 6.0] | 36/36 |
| SP - FD | 5.5 | [0.0, 16.0] | 36/36 |

Interpretation: FD slows earliest, BAL is close behind, and SP delays slowdown
the most. This matches the intended reward-preference direction: the more
speed-preserving agent waits longer before entering a stable slowdown response.

## Counterfactual Checks

Two counterfactual checks were run.

Initial-action counterfactual checks only the first action at `t=0`.
Rollout-level counterfactual runs the full episode and checks whether a stable
slowdown onset occurs anywhere in the episode.

### Initial-Action Counterfactual

At `t=0`, the no-front and far-front controls produce zero initial `SLOWER`
actions for all agents:

| Variant | FD SLOWER | BAL SLOWER | SP SLOWER |
|---|---:|---:|---:|
| original | 36/36 | 18/36 | 13/36 |
| no-front | 0/36 | 0/36 | 0/36 |
| far-front | 0/36 | 0/36 | 0/36 |
| matched-speed-front | 36/36 | 0/36 | 24/36 |

This removes the earlier concern that SP has an unconditional opening slowdown
bias.

### Rollout-Level Counterfactual

The full-episode no-front control is the strongest negative control.

| Variant | Agent | Valid slowdown onset | Median latency | Action summary |
|---|---|---:|---:|---|
| no-front | FD | 0/36 | NA | `FASTER=3780`, `IDLE=540` |
| no-front | BAL | 0/36 | NA | `FASTER=4320` |
| no-front | SP | 0/36 | NA | `FASTER=3768`, `IDLE=552` |

Since `4320 = 36 episodes x 120 steps`, this is not a `t=0`-only result. It
shows that across the complete no-front rollouts, none of the agents produced a
stable slowdown onset.

Far-front controls still produce slowdown, but with delayed timing:

| Variant | FD median | BAL median | SP median |
|---|---:|---:|---:|
| far-front | 20.0 | 37.0 | 45.0 |

This is consistent with a front-vehicle-driven response: when the front vehicle
starts much farther away, slowdown occurs later. SP remains the latest agent in
this control as well.

Matched-speed-front is less clean as a negative control because a front vehicle
is still present and occupies the lane. It should be treated as a diagnostic,
not the main causal control.

## Interpretation

The mixed single-lane experiment supports a cautious causal interpretation:

- In slow-front scenarios, FD/BAL/SP differ in slowdown timing.
- In no-front scenarios, the learned policies do not produce stable slowdown.
- In far-front scenarios, slowdown is delayed rather than immediate.

Therefore, the observed slowdown timing is consistent with front-vehicle
pressure interacting with reward preferences, rather than a trivial default
slowdown policy.

The strongest current statement is:

> In a controlled single-lane Highway setting, mixed no-front training produces
> policies whose slowdown timing differs by reward preference, while
> full-episode no-front counterfactuals rule out unconditional slowdown as the
> explanation.

## Caveats

This is a sanity experiment, not the final Highway result.

- The run is only 20K timesteps. A 100K or longer mixed-training run is needed
  before using this as the main mentor-facing result.
- `terminal_failure=0` in the timing table means no failure before the detected
  slowdown onset. It does not mean the full episode is collision-free.
- In the original slow-front rollouts, many episodes still collide after the
  slowdown onset:
  - FD: 21/36 collision flags
  - BAL: 24/36 collision flags
  - SP: 24/36 collision flags
- Thus, the current result supports timing and counterfactual sanity, but not
  final safe-driving performance.

## Relation to Multi-Lane Highway Progress

The main project target is still the multi-lane Highway setting. The current
multi-lane experiment uses a controlled open-lane scene containing the ego
vehicle, one slower front vehicle, empty adjacent lanes, and no random traffic.
As in the single-lane sanity experiment, Double DQN training used 20,000
timesteps per agent and mixed `80%` slow-front episodes with `20%` no-front
cruise episodes.

In the 24 matched slow-front evaluation scenes, all three agents completed a
physical lane change in `24/24` episodes and no collisions were observed.
Median first actual lane-change latency was:

- `SP`: 1.0 policy step
- `BAL`: 4.0 policy steps
- `FD`: 9.5 policy steps

The stable lane-change-onset analysis produced valid paired gaps in all `24/24`
scenes:

- `SP - BAL`: median -3.0 policy steps
- `BAL - FD`: median -6.5 policy steps
- `SP - FD`: median -9.5 policy steps

Thus, the learned policies have a robust behavioural ordering: `SP` changes
lanes earliest, `BAL` is intermediate, and `FD` changes lanes latest. However,
the counterfactual figure shows that the agents differ in why this ordering
appears:

- `BAL`: initial lane change in `12/24` original and `12/24` matched-speed
  states, but `0/24` no-front and `0/24` far-front states.
- `FD`: initial lane change in `6/24` original and `4/24` matched-speed states,
  but `0/24` no-front and `0/24` far-front states.
- `SP`: initial lane change in `24/24` states under every counterfactual,
  including no-front and far-front conditions.

For BAL and FD, removing or moving the front vehicle changes the initial action,
which supports context-sensitive behaviour. SP instead retains a strong general
opening lane-change bias even after no-front examples were added to training.
Therefore the timing result establishes behavioural differentiation among the
policies, but SP's early lane changes cannot yet be attributed specifically to
the slower front vehicle. The 20K mixed-training intervention was not sufficient
to remove this bias; a longer SP-only convergence check or a targeted reward
ablation is required before making a stronger causal claim.

This single-lane experiment is therefore useful as a controlled diagnostic:

- It removes the lane-change action entirely.
- It verifies that the reward-preference ordering can appear in longitudinal
  slowdown timing.
- It shows that adding no-front negative controls can remove unconditional
  slowdown behavior.
- It provides a cleaner template for the next multi-lane iteration: keep the
  open-lane design, add no-front/far-front controls, and validate whether early
  `SP` lane changes are caused by the slow front vehicle rather than by an
  unconditional opening lane-change bias.

## Reproduction Commands

Train mixed 20K:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py \
  --out outputs/single_lane_slow_front_mixed_20k_smoke \
  --timesteps 20000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --no-front-train-fraction 0.2 \
  --bootstrap-samples 500
```

Run initial-action counterfactual:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py counterfactual \
  --run-dir outputs/single_lane_slow_front_mixed_20k_smoke \
  --num-exposures 36
```

Run rollout-level counterfactual:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py rollout-counterfactual \
  --run-dir outputs/single_lane_slow_front_mixed_20k_smoke \
  --num-exposures 36 \
  --bootstrap-samples 500
```

## Artifacts

Main output directory:

```text
outputs/single_lane_slow_front_mixed_20k_smoke
```

Key files:

- `outputs/single_lane_slow_front_mixed_20k_smoke/training_runs.csv`
- `outputs/single_lane_slow_front_mixed_20k_smoke/analysis/outcome_summary.csv`
- `outputs/single_lane_slow_front_mixed_20k_smoke/analysis/gap_summary.csv`
- `outputs/single_lane_slow_front_mixed_20k_smoke/analysis/report_slowdown_latency_by_agent.png`
- `outputs/single_lane_slow_front_mixed_20k_smoke/analysis/report_counterfactual_valid_onsets.png`
- `outputs/single_lane_slow_front_mixed_20k_smoke/analysis/report_rollout_counterfactual_slowdown_response.png`
- `outputs/single_lane_slow_front_mixed_20k_smoke/analysis/report_initial_action_counterfactual_rates.png`
- `outputs/single_lane_slow_front_mixed_20k_smoke/counterfactual_front_vehicle/counterfactual_action_summary.csv`
- `outputs/single_lane_slow_front_mixed_20k_smoke/rollout_counterfactual_front_vehicle/counterfactual_rollout_summary.csv`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/analysis/actual_lane_change_summary.csv`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/analysis/gap_summary.csv`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/analysis/report_lane_change_latency_by_agent.png`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/analysis/report_initial_action_counterfactual_rates.png`
- `outputs/multilane_open_lane_change_mixed_20k_smoke/counterfactual_open_lane/counterfactual_action_summary.csv`

## Next Step

Run the same mixed setup at 100K timesteps, then repeat both counterfactual
checks. If the 100K run preserves the timing ordering and reduces post-onset
collisions, it can become the stronger version for mentor-facing slides.
