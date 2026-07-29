# Highway Matched-Exposure Transition Timing Plan

> Scope: Highway-first plan for comparing reward-preference agents through matched-exposure mode-transition response timing.
> Source proposal: `02_proposal/10Jul_transition_timing.md`.
> Current decision: use custom configured Highway agents rather than trying to exactly reproduce CL/SD/FR.
> Important change: semantic event labels are optional contextual annotations, not the core timing anchor.
> Current mode vocabulary: use three primary modes, with collision/failure handled as episode outcomes rather than modes.
> Current evaluation decision: use matched-exposure response timing only.

## 1. Core Idea

The project should compare **the timing of persistence-confirmed behavioural-mode transitions across reward-preference agents after the same controlled Highway traffic exposure**.

The main claim is:

```text
The project examines whether reward-preference agents exhibit different mode-transition timing under the same controlled Highway traffic exposure.
```

Example:

```text
Exposure E12:
ego_speed = 28 m/s
nearest_front_distance = 20 m
front_vehicle_speed = 22 m/s

FrontDistance agent: lane_keeping_cruise -> traffic_spacing_adjustment after 3 timesteps
Balanced agent:      lane_keeping_cruise -> traffic_spacing_adjustment after 7 timesteps
SpeedPriority agent: high_speed_cruise continues; no stable transition within horizon
```

This avoids overclaiming causality between an event label and a mode transition, and it avoids confusing "when an agent encounters a front vehicle" with "how quickly the policy responds after the same exposure." Event/context labels can still describe the exposure state, but the primary measurement is response latency from the matched exposure point.

Primary timing comparison does not require agents to enter the same semantic destination mode. The main timing target can be the first stable mode transition after exposure:

```text
first_stable_transition_after_exposure
```

The semantic labels of the transition, such as `lane_keeping_cruise -> traffic_spacing_adjustment` or `lane_keeping_cruise -> high_speed_cruise`, are recorded to explain what each agent changed into. They are not required to match across agents for the primary timing-gap calculation.

## 2. Research Questions

### RQ1. Mode Detection

Can step-level Highway behaviour be grouped into stable, interpretable modes?

Evidence:

- mode labels are not one-step action noise;
- detected modes correspond to visible driving behaviour;
- mode segmentation is robust to persistence-window choices.

Definition:

> A mode is treated as stable only when the same smoothed mode label persists for at least `k` consecutive timesteps. The default setting is `k = 3`, with robustness checks for `k in {2, 3, 5}`.

### RQ2. Matched-Exposure Response Timing

When agents are placed in the same traffic exposure state, do their first stable behavioural-mode transitions occur at different response latencies?

Evidence:

- ego speed, ego lane, front-vehicle distance and relative speed are controlled;
- response latency from the matched exposure point to the next stable transition onset differs across agents;
- timing gaps are computed as paired within-exposure differences before being aggregated across exposure IDs;
- response-latency features classify the reward-preference condition better than simple action-frequency baselines.

### RQ3. Summary Compactness

Can matched-exposure response-latency summaries characterize policy differences more clearly than clip-only summaries?

Evidence:

- a small number of exposure-aligned response-latency summaries explains the main agent differences;
- timing summaries reveal differences that immediate action-disagreement clips do not organize explicitly;
- representative clips match the detected mode-onset patterns.

## 3. Highway Environment

Use a maintained Highway simulator, preferably `highway-env`, a Farama-hosted autonomous-driving environment package used through the Gymnasium API, unless the original DISAGREEMENTS Highway code is found and easy to run.

Relevant terminology and `highway-env` facts:

- Farama is the foundation/maintainer, not the environment API itself.
- Gymnasium is the maintained fork/API successor of OpenAI Gym.
- `highway-env` is a collection of Gymnasium-compatible autonomous-driving environments hosted under Farama.
- `highway-v0` is a multi-lane highway driving environment with neighbouring vehicles.
- Default discrete meta-actions include `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`.
- Default reward configuration includes speed, right-lane, lane-change and collision terms.
- Reward functions and environment configuration can be customised.
- In `highway-v0`, ego collision should be treated as a terminal failure outcome, not as a recovery mode.

The implementation should report whether it uses:

- original DISAGREEMENTS code;
- `highway-env` through the Gymnasium API;
- another Highway simulator.

### 3.1 Matched-Exposure Configuration

Use the same controlled exposure states for all agents.

Key traffic controls:

| Parameter | Role | Main setting |
|---|---|---|
| `ego_lane` | ego lane at exposure start | identical across agents within an exposure |
| `ego_speed` | ego speed at exposure start | identical across agents within an exposure |
| `nearest_front_distance` | distance to the nearest front vehicle at exposure start | identical across agents within an exposure |
| `front_vehicle_speed` | lead vehicle speed at exposure start | identical across agents within an exposure |
| `vehicles_density` | background traffic density | identical across agents within an exposure; optionally varied across exposure sets |
| `duration` | response horizon after exposure | fixed across agents |

Suggested scenario difficulty sweep:

```text
low traffic:    vehicles_density = 0.5
medium traffic: vehicles_density = 1.0
high traffic:   vehicles_density = 1.5 or 2.0
```

Do not give different agents different traffic density. Density is a scenario condition, not an agent condition.

## 4. Agent Conditions

Use custom reward-preference agents. These are inspired by the reward-preference design in Amitai and Amir (2021), but they are our own experimental conditions.

| Agent ID | Descriptive name | Reward pressure | Intended aggregate behaviour |
|---|---|---|---|
| FD | FrontDistance / Conservative | high front-vehicle distance + collision avoidance + moderate speed | enters traffic-spacing adjustment mode early |
| SP | SpeedPriority | high speed + collision avoidance, weak distance preference | stays in high-speed cruise longer or has more no-onset cases |
| BAL | Balanced | intermediate weights over speed, front distance and collision avoidance | switches between speed and spacing modes at intermediate times |

Important naming rule:

> In analysis tables, prefer `FD`, `SP`, and `BAL` or reward-based names. Avoid treating "conservative" or "aggressive" as ground-truth behavioural conclusions.

### 4.1 Reward Components

Use a shared reward-component vocabulary for all agents:

| Component | Meaning | FD weight | SP weight | BAL weight |
|---|---|---:|---:|---:|
| speed_score | normalized ego speed | medium | high | medium |
| front_distance_score | normalized distance to nearest front vehicle in ego lane | high | low | medium |
| collision_penalty | collision/crash penalty | high | high | high |
| collision_risk_penalty | optional common anticipatory penalty for low time-to-collision / very close front vehicle risk | 0 in main experiment | 0 in main experiment | 0 in main experiment |
| lane_change_penalty | optional penalty for excessive lane-change oscillation | low/medium | low/medium | low/medium |
| slow_down_penalty | optional common penalty for the `SLOWER` action in brake-cost lane-change timing pilots | 0 in main experiment | 0 in main experiment | 0 in main experiment |
| right_lane_score | right-lane preference | 0 in main experiment | 0 in main experiment | 0 in main experiment |

Design rule:

> No reward term should directly optimise early or late transition timing.

The timing pattern must emerge from reward preferences, not from a reward term that explicitly rewards early switching.

Reward-setting rule:

> Set `right_lane_reward = 0.0` in the main experiment.

The right-lane reward can make lane changes reflect a lane-position preference rather than speed or spacing behaviour. It should only appear in a robustness or ablation condition, such as `right_lane_reward = 0.1`.

Secondary brake-cost condition:

> For lane-change timing comparisons, optionally train a separate `brake-cost`
> agent set with the same positive `slow_down_penalty` applied to FD, BAL and SP.
> If policies accelerate into a slow front vehicle before any crash penalty is
> observed, add the same positive `collision_risk_penalty` to all three agents
> as an anticipatory safety cost.

This does not replace the baseline reward-preference experiment. It creates a
controlled condition where all agents face the same braking cost and same
close-front risk cost, making `lane_change_onset` a cleaner focal target.
Results should be reported as `FD/BAL/SP under common braking and collision-risk
costs`, not as the baseline agent behaviour.

For the clean lane-change timing variant, use a gradual closing scenario:

```text
gradual-slow-front
```

In this condition, a slow front vehicle is present from the initial state but is
placed far ahead, such as 60-120m. The ego vehicle naturally closes the distance
over time; no vehicle is inserted mid-episode and the ego state is not reset.
The `lane_change_onset` target then measures when each policy first begins a
stable lane-change response while approaching the slower front vehicle.

The `free-cruise-then-slow-front` delayed-insertion mode can be retained as a
stress test, but it should not be used as the main timing scenario because it
creates an abrupt state change.

Training/evaluation alignment rule:

> If the gradual slow-front scenario is used as the main lane-change timing
> evaluation, train the agents with `--training-exposure-mode
> gradual-slow-front` rather than the default random `highway-v0` reset
> distribution.

The training reset should sample the same scenario family: slow front vehicle
present from the initial state, initial front distance sampled continuously from
a far range such as 55-150m, front speed sampled continuously around 14-22m/s,
ego speed sampled around 24-31m/s, and adjacent lane open or constrained
depending on the episode. Evaluation can still use a fixed grid of matched
exposure states for reproducible paired comparisons. No mid-episode insertion
should be used for this training condition.

### 4.2 Diagnostic Scores

During evaluation, compute the same diagnostic scores for all agents, regardless of their training reward:

```text
speed_score
front_distance_score
closest_k_distance_score
collision_risk_score
lane_position_score
lane_change_count
```

These scores are used for analysis and mode grounding. They should not be confused with the training reward unless explicitly stated.

## 5. Rollout Dataset

Run every trained agent on the same held-out matched-exposure states.

Primary clean rollout unit:

```text
agent_condition x exposure_id x rollout
```

Each rollout starts at `exposure_t = 0` from a controlled exposure state. The result measures how the trained policy responds after that exposure.

Use one fixed trained policy per condition in the first clean experiment:

```text
FD_main
BAL_main
SP_main
```

`policy_id` records which trained model is being evaluated.

Evaluation repeat rule:

- Main experiment: use deterministic evaluation when possible. Fix the exposure state, background vehicle state and environment seed; evaluate the policy with argmax/mean action rather than action sampling. Under this setting, repeating the same `agent_condition x exposure_id` should reproduce the same rollout, so one rollout is enough.
- Robustness only: if the policy or environment is stochastic during evaluation, run repeated rollouts with explicit `rollout_id` and `rollout_seed`. Use the same set of rollout seeds for FD, BAL and SP under the same `exposure_id` so stochastic variation is also paired.
- Do not mix training variation and rollout variation. `policy_id` identifies the trained model being evaluated. `rollout_seed` means a repeated evaluation of the same trained model from the same exposure.

Aggregation hierarchy for stochastic evaluation:

1. raw rollout level:

```text
agent_condition x policy_id x exposure_id x rollout_seed
```

2. paired rollout-gap level:

```text
agent_pair x policy_pair x exposure_id x rollout_seed
```

Compute timing gaps only between agents evaluated with the same `exposure_id` and `rollout_seed`. If using multiple independently trained policies, define matched policy pairs clearly, such as `FD_seed0` versus `SP_seed0`, or summarize within each policy before aggregating.

3. exposure-policy summary level:

```text
agent_pair x policy_pair x exposure_id
```

Summarize repeated rollout gaps within this cell using median gap, interquartile range, onset probability, no-onset rate and terminal-failure rate.

4. final comparison level:

```text
agent_pair x exposure_id
or
agent_pair x exposure_difficulty_bin
```

Aggregate the exposure-policy summaries across exposure IDs. If optional multiple training seeds are used, report them as a robustness layer rather than the main comparison.

### 5.1 Step Table

One row per timestep:

```text
episode_id
agent_condition
policy_id
exposure_seed
exposure_id
rollout_id
rollout_seed
evaluation_policy_mode
t
ego_lane
ego_speed
ego_position
action
q_values_or_action_scores
policy_hidden_activations_or_probe_features
nearest_front_vehicle_id
nearest_front_distance
closest_k_distances
speed_score
front_distance_score
closest_k_distance_score
collision_risk_score
lane_position_score
reward_total
reward_components
collision_flag
done
termination_reason
episode_outcome
exposure_t
```

### 5.2 Mode Table

One row per timestep after mode grounding:

```text
episode_id
agent_condition
policy_id
exposure_seed
exposure_id
rollout_id
rollout_seed
t
mode_label
mode_confidence
raw_intent_label
raw_intent_confidence
mode_features
smoothing_status
```

Primary mode labels should be assigned only for timesteps before terminal failure. Collision itself is not a mode label.

### 5.3 Transition Table

One row per accepted mode transition:

```text
transition_id
episode_id
agent_condition
policy_id
exposure_seed
exposure_id
rollout_id
rollout_seed
mode_before
mode_after
onset_t
confirmation_t
transition_interval
persistence_length
transition_confidence
local_context_before
local_context_after
episode_outcome
response_latency
```

The `onset_t` is the first timestep of the new persistence-confirmed mode. The `confirmation_t` is the timestep at which the persistence rule confirms that the new mode has lasted for at least `k` timesteps.

Example:

```text
t=10: lane_keeping_cruise
t=11: traffic_spacing_adjustment
t=12: traffic_spacing_adjustment
t=13: traffic_spacing_adjustment

If persistence k = 3:
onset_t = 11
confirmation_t = 13
transition_interval = [11, 13]
```

### 5.4 Episode Outcome Table

One row per rollout:

```text
episode_id
agent_condition
policy_id
exposure_seed
exposure_id
rollout_id
rollout_seed
episode_outcome
termination_t
termination_reason
collision_flag
valid_transition_count
no_onset_modes
first_collision_before_spacing_adjustment
```

Use the following outcome categories for transition-timing analysis:

| Outcome category | Meaning | Timing treatment |
|---|---|---|
| `valid_onset` | non-collision episode where the analysis target appears | include response latency |
| `no_onset_censored` | non-collision episode ends by duration before the analysis target appears | record as censored at `duration`, not as missing |
| `terminal_failure` | collision occurs before the analysis target appears | report separately as failure, not as a mode transition |

Do not silently drop failed or censored episodes. The primary timing plots can focus on valid non-failure onsets, but result tables must also report collision rate, `terminal_failure` count and `no_onset_censored` count.

### 5.5 Matched-Exposure Table

One row per controlled or matched exposure condition:

```text
exposure_id
exposure_seed
ego_lane
ego_speed_at_exposure
nearest_front_distance_at_exposure
front_vehicle_speed_at_exposure
relative_closing_speed_at_exposure
vehicles_density
background_vehicle_state
exposure_source
```

`exposure_source` can be:

```text
controlled_initial_state
cloned_rollout_state
```

Preferred fair-comparison design:

1. create controlled exposure states where ego lane, ego speed, front-vehicle distance and front-vehicle speed are identical for all agents;
2. start FD, BAL and SP from each exposure state;
3. run each agent for a short horizon;
4. measure `response_latency = onset_t - exposure_t` for the next accepted transition.

This directly tests whether agents switch modes at different times after the same traffic exposure.

Each exposure is one paired scenario fragment, not the whole experiment. The result should be a distribution of paired timing gaps across many held-out exposure IDs.

Exposure grouping has two levels:

1. Exact matched exposure block:

```text
exposure_id = one controlled traffic state
```

Within one `exposure_id`, FD, BAL and SP start from the same ego lane, ego speed, front-vehicle distance, front-vehicle speed and traffic density. This is the unit for paired timing-gap comparison.

2. Exposure difficulty bin:

```text
exposure_difficulty_bin = a coarse group of similar exposure states
```

Use difficulty bins only for stratified reporting, not for replacing the paired comparison. Suggested bins:

| Bin dimension | Example groups | Meaning |
|---|---|---|
| front distance | close / medium / far | how near the lead vehicle is at exposure start |
| relative closing speed | low / medium / high | whether ego is catching the lead vehicle slowly or quickly |
| traffic density | sparse / medium / dense | how constrained lane-change options may be |
| combined urgency | low / medium / high | optional combined bin from front distance and closing speed |

Example:

```text
E12:
front_distance = 20m
relative_closing_speed = 6m/s
traffic_density = medium
exposure_difficulty_bin = close_front_high_closing_medium_density
```

### 5.6 Timing Gap Table

One row per agent pair, exposure, policy pair and analysis target:

```text
gap_id
policy_id_a
policy_id_b
exposure_seed
exposure_id
rollout_id
rollout_seed
analysis_target
agent_a
agent_b
latency_a
latency_b
gap_b_minus_a
outcome_a
outcome_b
gap_status
exposure_difficulty_bin
```

Definition:

```text
gap_b_minus_a = response_latency_b - response_latency_a
```

For example, if `agent_a = FD` and `agent_b = SP`, a positive gap means SP reached the analysis target later than FD under the same exposure. For the primary target, this means SP made its first stable mode transition later than FD, regardless of what semantic mode it transitioned into. Use `gap_status` to distinguish `valid_pair`, `censored_pair` and `terminal_failure_pair`.

If deterministic evaluation is used, `rollout_id` can be a constant such as `r0`. If stochastic evaluation is used, compute gaps only between rollouts with the same `exposure_id` and `rollout_seed`, or summarize repeated rollout latencies within each agent before computing the exposure-level paired gap.

## 6. Mode Grounding

Mode grounding labels each timestep with an interpretable behavioural mode. A mode should describe a sustained driving behaviour or strategy, not simply rename one action. For example, `LANE_LEFT` and `SLOWER` are action-level signals; both can be evidence for `traffic_spacing_adjustment` when they are part of sustained behaviour that increases distance from front or nearby vehicles.

Primary mode vocabulary:

| Mode                       | Meaning                                                             | Possible evidence                                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| lane_keeping_cruise        | straight/lane-stable cruising                                       | `IDLE`, stable lane, stable moderate speed                                                                                                                     |
| high_speed_cruise          | sustained high-speed driving                                        | high speed score, `FASTER` or high-speed Q preference, low braking                                                                                             |
| traffic_spacing_adjustment | sustained behaviour to increase space from front or nearby vehicles | front distance or closest-k distance low/decreasing, `SLOWER` and/or `LANE_LEFT`/`LANE_RIGHT`, front-distance or closest-k score improves after the adjustment |

The labels are behavioural summaries, not claims about hidden intention. Raw actions such as `LANE_LEFT`, `LANE_RIGHT`, `FASTER` and `SLOWER` are evidence for modes, not modes by themselves. For `traffic_spacing_adjustment`, record an auxiliary `adjustment_method` field with values such as `brake`, `lane_change`, or `mixed`; this keeps the mode vocabulary compact while preserving how the adjustment was performed.

Collision and post-collision behaviour are not primary modes. In `highway-v0`, collision is a terminal failure outcome, so it should be reported in the episode outcome table and aggregate failure metrics.

### 6.1 Mode Features

Use observable behaviour and policy outputs:

```text
action
recent action history
ego speed and speed delta
ego lane and lane delta
nearest_front_distance and its delta
closest_k_distances
q_values_or_action_scores
policy_hidden_activations_or_probe_features
tcav_concept_scores
concept_score_margin
diagnostic score vector
mode-context features such as whether front distance or closest-k distance improves after the action
```

### 6.2 TCAV-Based Intent Labelling

Preferred method:

1. define concept sets for the three primary mode labels:
   - `lane_keeping_cruise`;
   - `high_speed_cruise`;
   - `traffic_spacing_adjustment`;
2. collect positive and negative examples for each concept from rollouts, using observable state/action traces and visual audit;
3. train CAVs over policy hidden activations or a stable state-policy embedding;
4. for each timestep, compute concept-aligned scores for the selected action, selected-action Q-value, action Q-gap, or policy-output score;
5. assign a raw intent label to the highest-scoring concept if its margin is large enough; otherwise label the timestep as `ambiguous`.

Wording rule:

> TCAV labels should be described as concept-aligned behavioural intent or mode evidence, not as proof of the agent's true hidden intention.

Fallback method:

- if policy activations are not available, use diagnostic/rule-based labels over action, state, Q-values and score deltas;
- keep the same three-mode vocabulary so that TCAV-based and rule-based results can be compared.

Optional supporting methods:

- change-point detection over action distributions or Q-value rankings;
- clustering over state-policy features;
- compare TCAV-based, rule-based and clustering-based segmentations.

### 6.3 Temporal Aggregation and Gap Bridging

The raw intent label is timestep-level evidence. The mode is a temporally aggregated segment.

Operational definition of stable mode:

```text
stable_mode(label, onset_t, k) = label appears at every timestep from onset_t to onset_t + k - 1
```

after the predefined short-gap bridging rule has been applied.

Procedure:

1. assign a raw intent label at every timestep;
2. bridge tiny interruptions: if a short segment `B` of length `r` or less is surrounded by the same label `A`, merge `A B A` into `A`;
3. use `r in {1, 2}` as the main robustness setting;
4. run-length encode the smoothed labels into mode segments;
5. accept a new mode only if it persists for `k` timesteps;
6. report robustness for `k in {2, 3, 5}`.

Example:

```text
raw:      A A A B A A A
if r = 1:
smoothed: A A A A A A A
```

Persistence example:

```text
t=10: lane_keeping_cruise
t=11: traffic_spacing_adjustment
t=12: traffic_spacing_adjustment
t=13: traffic_spacing_adjustment

if k = 3:
onset_t = 11
confirmation_t = 13
response_latency = onset_t - exposure_t
```

Do not bridge across terminal collision. Store both raw and smoothed labels so that the smoothing decision can be audited.

Acceptance criteria:

- mode labels are visually checkable in clips;
- TCAV concept labels pass held-out concept sanity checks;
- one-step action changes do not dominate the transition table;
- short interruption bridging reduces flicker without hiding meaningful safety responses;
- transition counts are plausible, not near-zero or exploding.

## 7. Matched-Exposure Response Timing

The primary timing object is:

```text
first_stable_transition_after_exposure at response_latency
```

Compare response latency across agents under the same exposure ID. The destination mode does not need to be the same across agents for the primary timing-gap calculation.

Important distinction:

> The detector should not be built to find only one preselected mode. First extract all stable mode segments and all accepted transitions. The primary paired timing gap can use the first accepted transition after exposure, regardless of its semantic destination. Semantic focal comparisons can then be reported as secondary slices after the mode vocabulary is fixed.

### 7.1 Timing Metrics

| Metric | Purpose |
|---|---|
| first_transition_latency | time from `exposure_t` to the first accepted stable mode transition |
| response_latency(mode) | secondary metric: time from `exposure_t` to the first accepted onset of a specific semantic mode |
| lane_change_onset_latency | secondary action-level metric: time from `exposure_t` to the first persistence-confirmed `LANE_LEFT`/`LANE_RIGHT` onset |
| response_latency_distribution | distribution of exposure-aligned response latencies |
| latency_rank_by_exposure | which agent changes stable mode first under the same exposure |
| mode_duration | how long the agent remains in a mode |
| transition_frequency | how often the agent changes modes within the response horizon |
| transition_matrix | which mode transitions are common for each agent |
| context_at_exposure | controlled traffic context at `exposure_t` |

### 7.2 Full Transition Summary and Focal Comparisons

Always report the full transition summary first:

- aligned mode timelines for representative matched exposures;
- transition matrix for each agent condition;
- mode-duration distributions;
- transition-count distributions.

Then use semantic focal comparisons as secondary hypothesis tests. These focal comparisons are not used to force the detector; they are selected after the mode vocabulary is fixed so the statistical analysis does not become an unconstrained search over every possible transition.

Primary timing comparison:

| Analysis target | Expected pattern |
|---|---|
| first stable transition after exposure | FD may change stable mode earlier than BAL/SP; SP may continue without a stable transition within the horizon |

This primary target does not require the same `mode_after` across agents. It asks when the policy first leaves its current stable behaviour after the same exposure.

Secondary semantic focal comparisons:

| Focal mode/transition | Expected pattern |
|---|---|
| lane_keeping_cruise -> traffic_spacing_adjustment | FD shorter latency than BAL, BAL shorter latency than SP |
| high_speed_cruise -> traffic_spacing_adjustment | FD more frequent or shorter latency than SP in close-front exposures |
| traffic_spacing_adjustment -> lane_keeping_cruise | SP may return to cruise earlier |
| any mode -> high_speed_cruise | SP earlier and longer duration than FD/BAL |
| lane_change_onset | in the brake-cost condition, compare which agent begins a stable lane-change response earlier |

These are hypotheses about timing patterns, not causal claims about a particular future event.

### 7.3 Handling Multiple-Step Transitions

A transition is an interval, not just a single frame:

```text
transition_interval = [onset_t, confirmation_t]
response_latency = onset_t - exposure_t
```

Use `response_latency` for timing comparison because every agent starts from the same exposure state. Use `confirmation_t` as evidence that the onset was not one-step noise.

Report both:

- response-latency results;
- robustness across persistence windows.

### 7.4 Timing Gap Aggregation

An individual matched exposure is an illustrative unit and one statistical block. It can show the method clearly, but it is not sufficient evidence by itself.

For each `exposure_id`, compute paired timing gaps between agents for the same analysis target. The default primary target is:

```text
first_stable_transition_after_exposure
```

This target does not require the two agents to have the same `mode_before` or `mode_after`.

```text
gap(FD, SP | exposure_id) = response_latency_SP - response_latency_FD
gap(FD, BAL | exposure_id) = response_latency_BAL - response_latency_FD
gap(BAL, SP | exposure_id) = response_latency_SP - response_latency_BAL
```

Then summarize the gap distribution across exposure IDs. Report median, interquartile range, bootstrap confidence intervals and latency-rank counts. If optional multiple policy seeds are used, report seed-level robustness separately. Do not rely only on a naive mean, because different exposure states can naturally produce different timing gaps.

The primary paired timing-gap plot should be a zero-centered box/strip plot:

- x-axis: agent pair, such as `SP - FD`, `BAL - FD` and `SP - BAL`;
- y-axis: `gap_b_minus_a = response_latency_b - response_latency_a`;
- each jittered point: one matched `exposure_id`;
- summary overlay: boxplot or median plus IQR/bootstrap CI;
- reference line: horizontal `y = 0`, meaning both agents transition at the same response latency.

Stratify or control for exposure difficulty using variables such as front distance, relative closing speed and traffic density. Treat `no_onset_censored` as censored and `terminal_failure` as failure; do not convert them into ordinary finite latencies.

### 7.5 Collision and Censoring Handling

Primary transition-timing analysis should be run on non-collision mode sequences. However, the denominator must include all attempted rollouts.

For each analysis target, classify each rollout as:

```text
valid_onset: target transition observed before the response horizon ends
no_onset_censored: no collision, but target transition not observed before the response horizon ends
terminal_failure: collision before the target transition
```

This avoids selection bias. For example, if SP has many collisions or many `no_onset_censored` episodes, those are part of the policy-style result rather than rows to delete silently.

When plotting paired timing gaps, do not encode `no_onset_censored` or `terminal_failure` as arbitrary large numeric gaps. Report these outcomes beside the gap distribution using a stacked bar chart over outcome categories:

```text
valid_onset / no_onset_censored / terminal_failure
```

### 7.6 Exposure Alignment Requirement

The main fair response-timing metric is:

```text
response_latency = transition_onset_t - exposure_t
```

Fairness requirement:

> In matched-exposure analyses, ego speed, ego lane, nearest-front distance, front-vehicle speed and traffic density should be identical or tightly matched across agents at `exposure_t`.

Construct controlled initial states with the same ego speed, front distance and lead-vehicle speed. If exact environment cloning is available, use cloned exposure states.

## 8. Optional Context Annotation

Semantic event labels are no longer the core analysis. They can be used only as contextual annotations around matched exposures or transition onsets.

Examples:

```text
At onset_t, nearest_front_distance = 12.4m
At onset_t, closest_k_distance_score = 0.31
At onset_t, ego speed = 28.7m/s
At onset_t, collision_risk_score = high
```

Optional labels may include:

| Context label | Derived from | Use |
|---|---|---|
| close_front_vehicle | nearest front distance or quantile | describe transition context |
| high_collision_risk | risk diagnostic score | describe transition context |
| right_lane_available | lane topology/state | describe opportunity context |
| dense_neighbourhood | closest-k distance score | describe traffic density |

Rule:

> Do not claim that a context label caused the transition unless a separate intervention/counterfactual test is performed.

## 9. Baselines

Compare matched-exposure response timing against policy-summary baselines on the same exposure states.

| Baseline | Source | What it tests |
|---|---|---|
| Random clips | simple control | whether summaries beat unstructured examples |
| HIGHLIGHTS-style clips | Amir and Amir (2018) | whether response timing adds beyond high-value-gap clips |
| Critical-state clips | Huang et al. (2018) | whether response timing adds beyond high-confidence states |
| DISAGREEMENTS clips | Amitai and Amir (2021) | whether response timing adds beyond immediate action disagreement |
| Aggregate behaviour metrics | simple baseline | whether response timing adds beyond mean speed, collision rate and distance statistics |

DISAGREEMENTS remains important, but now it is a baseline method applied to our custom agents under the same matched exposure states:

```text
FD vs SP
FD vs BAL
BAL vs SP
```

Main comparison question:

> Do matched-exposure response-latency summaries reveal policy-style differences that are not clear from immediate action-disagreement clips or aggregate behaviour metrics alone?

## 10. Experiments

### E1. Agent Sanity Experiment

Question:

Do FD, SP and BAL express different reward preferences?

Measurements:

- mean speed;
- collision rate;
- nearest-front distance;
- closest-k distance;
- braking frequency;
- lane-change frequency;
- mode-duration statistics.

Expected:

- FD has larger front-distance scores and earlier/more frequent traffic-spacing adjustment modes.
- SP has higher speed and longer high-speed cruise modes.
- BAL is intermediate.

### E2. Mode Detection Stability Experiment

Question:

Are detected modes and transitions stable?

Measurements:

- mean mode duration;
- one-step mode rate before/after smoothing;
- transition count per episode;
- transition confidence;
- visual audit of representative clips.

### E3. Main Matched-Exposure Response-Timing Experiment

Question:

Given the same traffic exposure, do FD, SP and BAL switch modes at different response latencies?

Controlled exposure variables:

- ego lane;
- ego speed;
- nearest-front distance;
- front-vehicle speed;
- relative closing speed;
- vehicles density/background traffic state.

Measurements:

- full transition matrix by agent condition;
- `response_latency = transition_onset_t - exposure_t`;
- paired timing gaps within the same `exposure_id`;
- valid-onset, no-onset-censored and terminal-failure counts within the response horizon;
- first stable transition after exposure and its semantic `mode_before -> mode_after` label;
- latency rank by exposure;
- timing-gap distributions by exposure difficulty bin;
- mode-duration distribution within the response horizon;
- adjustment method for `traffic_spacing_adjustment`.

Expected:

- FD should have shorter response latency into `traffic_spacing_adjustment`.
- SP should have longer response latency, more high-speed continuation, or more no-onset/terminal-failure cases.
- BAL should be intermediate.

### E4. Baseline Comparison Experiment

Question:

What does matched-exposure response timing explain beyond clip-only or aggregate baselines?

Compare:

- random clips;
- HIGHLIGHTS-style clips;
- critical-state clips;
- DISAGREEMENTS clips;
- aggregate behaviour metrics;
- matched-exposure mode-transition timing summaries.

Measurements:

- reward-preference classification from timing features;
- summary compactness;
- overlap with action-disagreement states;
- cases where response latency differs but immediate action disagreement is sparse or hard to summarize.

### E5. Robustness Experiment

Robustness dimensions:

- persistence window `k in {2, 3, 5}`;
- mode vocabulary and focal-comparison definitions;
- matched-exposure construction method;
- exposure matching tolerance, if approximate matching is used;
- optional policy/training seeds;
- optional context thresholds.

Success condition:

- major timing patterns hold across reasonable mode-persistence settings.

## 11. Statistical Plan

Primary unit:

```text
transition instance within agent_condition x policy_id x exposure_id
```

For response-latency analyses:

```text
observed_or_focal_mode x agent_condition x policy_id x exposure_id
```

Report:

- median response latency;
- interquartile range;
- bootstrap confidence intervals;
- paired timing-gap comparisons within exposure IDs;
- latency-rank counts;
- valid-onset, censored and terminal-failure counts.

Use non-parametric or permutation tests where possible, because response latencies may not be normally distributed. If many non-collision episodes are `no_onset_censored`, report a survival-style summary or a cumulative-onset curve rather than only a histogram of observed latencies.

The main cross-agent timing comparison should treat `exposure_id` as a paired block. A simple option is paired bootstrap or permutation testing over exposure IDs. A stronger optional model is:

```text
response_latency ~ agent_condition + exposure_features + (1 | exposure_id)
```

This keeps agent timing differences separate from the fact that some exposure states are intrinsically easier, closer, denser or more urgent than others. If optional multiple policy seeds are used, add a policy-level random effect or report seed-wise robustness separately.

If stochastic evaluation uses repeated rollouts, do not treat repeated rollouts from the same trained model and same exposure as independent exposure states. Either pair by the same `rollout_seed` across agents, or first summarize repeated rollouts within each `agent_condition x policy_id x exposure_id` cell before computing cross-agent timing gaps.

### 11.1 Interpretation of Timing-Gap Outputs

The main reported object is the paired timing gap:

```text
gap(agent_b - agent_a) = response_latency_b - response_latency_a
```

For example:

```text
gap(SP - FD) = response_latency_SP - response_latency_FD
```

A positive `gap(SP - FD)` means SP reaches the analysis target later than FD under the same matched exposure. For the primary target, this means SP makes its first stable mode transition later than FD. A negative value means SP transitions earlier.

Interpret the output metrics as follows:

| Output | What it represents | Example interpretation |
|---|---|---|
| median gap | central or typical paired timing difference across matched exposures | `median gap(SP - FD) = +5` means that in the median exposure, SP reaches the analysis target 5 timesteps later than FD |
| mean gap | average paired timing difference under the sampled exposure distribution | useful as a secondary summary, but can be affected by extreme exposure cases |
| IQR | spread of paired gaps across exposures | large IQR means the timing difference depends strongly on exposure context |
| bootstrap CI | uncertainty of the estimated median or mean gap | checks whether the observed central gap is stable across the sampled exposure set |
| gap by exposure difficulty bin | where the timing difference appears | close-front/high-closing exposures may show larger SP-FD gaps than easy exposures |
| latency-rank counts | how often each agent transitions first, second or last | shows whether the ordering FD/BAL/SP is consistent across exposures |
| no-onset rate | proportion of rollouts where the analysis target is not observed within the response horizon | indicates absent or very late switching, not an ordinary finite latency |
| terminal-failure rate | proportion of rollouts where collision occurs before the analysis target | indicates failure before transition and should be reported separately from timing gaps |

Use median gap as the main "typical timing" summary because paired timing gaps may contain extreme values from unusually difficult exposures. Mean gap can be reported as a secondary quantity, but the main claim should be based on the paired gap distribution, not raw latency averages.

For valid-valid pairs, compute numeric gaps directly. For valid-no-onset or failure pairs, keep the outcome category instead of converting it into an arbitrary large latency.

## 12. Figures and Tables

Required figures:

| Figure | Purpose |
|---|---|
| pipeline diagram | matched exposure -> mode grounding -> response latency -> timing summary |
| aligned mode timeline | compare FD/BAL/SP mode sequences for one exposure |
| exposure-aligned latency plot | compare response latency after matched traffic exposure |
| paired timing-gap distribution | use zero-centered box/strip plot with jittered exposure points and `y = 0` reference line |
| episode outcome stacked bar | show valid-onset, no-onset-censored and terminal-failure proportions beside the timing-gap plot |
| transition matrix heatmap | compare common mode transitions by agent |
| baseline comparison panel | show DISAGREEMENTS clips vs matched-exposure timing summary |

Required tables:

| Table | Purpose |
|---|---|
| reward configuration | define FD/SP/BAL weights |
| mode vocabulary | define behavioural modes and evidence |
| matched-exposure latency results | report response latency under controlled traffic exposure |
| paired timing-gap table | report within-exposure timing gaps, gap status and exposure difficulty bins |
| episode outcome table | report valid-onset, no-onset-censored and terminal-failure counts |
| aggregate sanity metrics | show agents actually differ |
| robustness table | show results across persistence windows |

## 13. Risks and Mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Agent labels become circular | "conservative" may look like assumed conclusion | use FD/SP/BAL reward-condition names in analysis |
| Mode definitions are hand-crafted | may look subjective | predefine concept examples/rules before analysis; validate TCAV labels and run robustness over definitions |
| Transition is one-step noise | timing result unreliable | use onset plus confirmation window |
| Failed episodes create selection bias | dropping collisions can make risky policies look normal | report terminal failures separately and keep them in denominators |
| No-onset episodes create selection bias | deleting no-onset cases hides late or absent switching | record `no_onset_censored` at duration |
| Right-lane reward confounds lane changes | lane changes may reflect right-lane preference rather than spacing | set `right_lane_reward = 0.0` in the main experiment |
| Agents are not exposed to the same state | timing comparison becomes unfair | use controlled matched exposures and compare within exposure IDs |
| Timing does not beat aggregate metrics | weak novelty | include aggregate baseline explicitly |
| DISAGREEMENTS already explains differences | weak novelty | report as baseline result; focus on mode-level timing where disagreement clips are fragmented |

## 14. Immediate Next Actions

- [ ] Create a primary analysis notebook (`ipynb`) for the full research workflow: load rollouts, build mode/transition/outcome tables, compute response latency, compute paired gaps, run robustness checks and export static figures/tables.
- [ ] Implement or choose Highway simulator.
- [ ] Define exact FD/SP/BAL reward weights.
- [ ] Set main reward config with `right_lane_reward = 0.0`.
- [ ] Train short pilot agents and check aggregate behaviour.
- [ ] Define mode vocabulary and TCAV/probe-assisted mode detector, with diagnostic rules as fallback.
- [ ] Implement transition onset/confirmation extraction.
- [ ] Implement episode outcome classification: `valid_onset`, `no_onset_censored`, `terminal_failure`.
- [ ] Implement matched-exposure evaluation with fixed ego speed, front distance and front-vehicle speed.
- [ ] Run one matched exposure across all three agents and plot aligned mode timelines.
- [ ] Plot the paired timing-gap distribution as boxplot plus jittered exposure points with a zero reference line.
- [ ] Add a stacked bar outcome panel for valid-onset, no-onset-censored and terminal-failure rates.
- [ ] Optionally build a Streamlit dashboard after the notebook results are stable, for interactive filtering by agent pair, exposure difficulty bin and persistence window `k`.
- [ ] Compare against DISAGREEMENTS on the same three agents and exposure states.

## 15. Positioning

Do not claim:

- the agent changes mode because of a future event;
- event labels causally explain transition timing;
- "conservative" and "speed-priority" are psychological ground truth;
- collision or recovery is a normal behavioural mode after terminal failure;
- transition timing replaces DISAGREEMENTS.

Claim instead:

- reward preferences induce different matched-exposure response-latency patterns;
- response latency can be measured robustly using persistence-confirmed mode changes;
- collisions and no-onset episodes are separate outcomes that must be reported with timing results;
- Highway provides a clear visual domain for comparing these timing patterns;
- DISAGREEMENTS is a strong baseline for immediate action differences, while matched-exposure response timing summarizes trajectory-level behavioural shifts.
