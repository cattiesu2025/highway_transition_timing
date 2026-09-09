# Stage-5 Summary Comparison: Pre-Registration v2 (DRAFT, NOT FROZEN)

> Status: **draft**. Nothing here is binding until the freeze procedure in
> section 10 is executed and its SHA-256 recorded. No summary has been shown to
> any participant, and no summary has been scored against any FD/BAL/SP claim,
> at the time of writing.
>
> This design supersedes the discarded computational-benchmark draft. Section
> 0 records what changed and why; the obsolete draft was removed during the
> repository cleanup on 2026-09-09.

## 0. What changed from v1, and why

v1 made a computational benchmark the primary evidence: six methods scored on
nine cells across fidelity, stability, compactness, and claim strength. Three
findings during its preparation made that design unsound, and one made it
unnecessary.

**The ground truth was ONSET's own estimator.** v1 §4.1 defined the correct
answer as the sign of the across-seed paired gap; v1 §5 classified a cell null
when the bootstrap interval contained zero; v1 §6 froze ONSET's readout as
"report the sign when the interval excludes zero, otherwise abstain". These are
one procedure written in three places. On the nine cells the classification and
the interval criterion agree in every row, so ONSET scored 4/4 on
identification and 5/5 on over-claiming by construction rather than by
evidence. v1 §4.2 described ONSET as structurally exposed on over-claiming; it
was structurally immune.

**The equivalence bound rewarded noise.** v1 §5 set each cell's bound to the
larger absolute end of its bootstrap interval, so a noisier cell got a wider
bound. The same claim of a two-second difference was a false claim in arm A's
`SP - BAL` (bound 0.40 s) and free in arm C's `BAL - FD` (bound 2.60 s), which
is the cell where the data are least informative.

**The oracle diagnostic collapsed onto an excluded metric.** v1 §6 promised an
upper bound computed "from exactly the states that method selected". Read
faithfully, that is a bound derived from what the clip windows show, which is
close to a restatement of onset coverage — the quantity v1 §8 declares
unscoreable because ONSET is defined by onsets.

**The field does not require the computational benchmark.** The comparison this
work answers to (Amitai and Amir 2022) supports its claim against HIGHLIGHTS
entirely with human experiments. It reports no fidelity, stability, or
compactness table. Repairing an instrument with the defects above is not worth
doing when the standard instrument is available.

v2 therefore moves the comparison to a human tier and keeps the computational
work as **descriptive mechanism** rather than as a scoreboard — the role v1 §8
already assigned to onset coverage. The diagnostics in section 7 are already
computed and none depends on the ground truth that failed.

### 0.1 Why the summaries are drawn across training seeds

The comparison work trains **one network per condition**: "All highway agents
were trained for 2000 episodes"; summaries "were generated for fully trained
agents, thus reflecting their final policies". Its `numSim = 10` is ten
evaluation episodes of that one pair, not ten training runs. Its ground truth
is therefore a property of one trained pair.

This project has twenty independently trained networks per condition per arm,
and they do not agree. In arm B's `BAL - FD` the per-seed median gap runs from
-2.60 s to +2.30 s, eight seeds positive and twelve negative, while *within*
almost every seed the difference is large and consistent — seed 4100 has all 36
exposures pointing the same way at +1.30 s. A summary of one trained pair
therefore supports a conclusion that reverses when the pair is retrained.

Summaries here are consequently drawn from the **pooled corpus of all twenty
seeds** for each condition, so that the object being summarised is the reward
condition rather than one network, and the across-seed quantities of section 5
are the matching ground truth. Section 9 records that this applies the clip
baselines beyond the scope their authors defined, and section 7's D8 and D9
record what that costs them.

> **This changes a recorded commitment.** `docs/user_requirements.md` has been
> updated: fidelity becomes identification accuracy, claim strength becomes the
> false-claim rate on null cells, and stability across seeds and compactness
> over an evidence budget are recorded as claims this experiment cannot
> support. The canonical proposal `docs/proposals/10Jul_transition_timing.docx`
> must be edited to match before this document is frozen.

## 1. Purpose

Experiment 1 established that transition timing separates reward conditions and
that the direction of the separation reverses between tasks. It did not
establish that a transition-timing summary is a better instrument than the clip
summaries already in the literature.

This experiment asks whether a person, shown a transition-timing summary, can
answer two questions about a pair of reward conditions more reliably than a
person shown a clip summary of the same rollouts under the same display budget:

1. **identification** — which of these two policies begins its transition
   earlier?
2. **over-claiming** — is there a difference at all?

The second question is the contribution relative to the comparison work, and it
is not a harder version of the first. Their participants chose between two
agents in a forced choice, with no way to answer "I cannot tell": their random
baseline sits at 0.5 because a response had to be given. Over-claiming is
therefore unmeasurable in that format, not overlooked. A summary that names a
winner when there is nothing to name fails in a way their design cannot see.

## 2. Data and scope

No training. No new rollouts. Rendering clips requires a deterministic replay
pass with `rgb_array` capture, described in section 4.3; it re-executes stored
policies and produces no new experimental data. The sealed held-out grid is not
opened; this experiment runs on development grid rollouts only.

| Arm | Run prefix | Seeds | Endpoint |
| --- | --- | --- | --- |
| A | `single_lane_slow_front_fixed_100k_seed` | 3100-3119 | `slowdown_onset` |
| B | `multilane_twolane_open_fixed_100k_seed` | 4100-4119 | `lane_change_onset` |
| C | `multilane_twolane_occupied_fixed_100k_seed` | 4200-4219 | `lane_change_onset` |

Each arm: 20 seeds x 3 policies x 36 exposures x 600 policy steps at 5 Hz. All
sixty run directories exist with their models and `evaluation/steps.csv`.

**The corpus for every method is the pooled twenty seeds of a condition within
one arm**, 720 episodes per condition. No seed is privileged and none is
excluded. Every method draws its clips from that corpus under the same budget.

## 3. Summary methods

| Method | Selection criterion | Source | Human tier |
| --- | --- | --- | --- |
| RANDOM | uniform over candidate states | lower bound | no |
| CRITICAL-STATE | `max_a Q - mean_a Q` above threshold, top-k | Huang et al. 2018 | no |
| HIGHLIGHTS | `max_a Q - min_a Q`, greedy with spacing | Amir and Amir 2018 | no |
| HIGHLIGHTS (Huber) | `max_a Q - second_a Q`, greedy with spacing | Huber et al. 2020 | **yes** |
| DISAGREEMENTS | disagreement states ranked by last-state importance | Amitai and Amir 2022 | **yes** |
| ABSTRACT | per-policy transition matrix over (TTC band x speed band x lane) | Topin and Veloso 2019 style | no |
| ONSET | clips centred on transition onsets, stratified across seeds and difficulty | this project | **yes** |

Three methods reach the human tier: **ONSET, HIGHLIGHTS (Huber), and
DISAGREEMENTS**. The variant carried forward is `max_second`, which is the
variant the comparison work's own code runs (`get_highlights.py:146`, recorded
in `docs/reference_impl/PROVENANCE.md` section 3.1).

**RANDOM is not run with participants.** Its role is a floor, and the floor is
already established in the literature the comparison work itself cites for it:
HIGHLIGHTS "was shown to be better than random (Huber et al. 2020)". Spending a
group of participants to re-establish a known floor is not defensible when the
same money buys statistical power on the contrast actually in question. RANDOM
remains in the descriptive tier of section 7.

CRITICAL-STATE, base HIGHLIGHTS (`max_min`), and ABSTRACT are likewise
descriptive-tier only. Each additional method is another block of items in a
within-subject design, and the comparison work states the constraint for the
same situation: "to reduce cognitive overload". Their exclusion is recorded
here, in advance, so that it cannot later be read as a selection among methods
that were tried and dropped.

`max_min` and `max_second` are reported as separate methods, not as one method
with a parameter. Their selections overlap by a per-arm mean of 0.6 to 3.1
states out of 5, with individual seeds ranging from 0 to 5, so the choice
determines the summary. Labelling either of them "HIGHLIGHTS" alone would
misattribute: the name is Amir and Amir's, and the `max_second` rule is Huber
et al.'s.

Parameters come from Table 1 of the DISAGREEMENTS paper, Highway column, with
the code cross-check recorded in `docs/reference_impl/PROVENANCE.md`:
`k = 5`, trajectory length 20 (so `context_length = 10`), `h = 10`,
`overlapLim = 5`, importance method `last_state`. Q values are min-max
normalised per agent over the trace before any cross-agent comparison, with
`agent_ratio = 1`; without this the condition with the largest Q scale
dominates every ranking. `numSim` is not taken from the paper's value of 10:
selection runs offline over the whole pooled corpus, the same corpus HIGHLIGHTS
selects from, because an unmatched corpus would confound the comparison with a
sampling difference. HIGHLIGHTS-DIV is excluded, for the reasons in PROVENANCE
section 3.3. DISAGREEMENTS pairs policies **within a training seed** — seed
*i*'s FD against seed *i*'s BAL — because a disagreement between networks that
never trained together is not a disagreement about anything.

## 4. Clip-based comparison

All human-study methods are rendered as clips. This keeps the display modality
fixed: methods differ only in which moments they select from the same rollout
corpus.

Each item shows five clips per policy, ten clips total, with the same clip
length, frame rate, playback controls, labels, and timing display for every
method. The numeric panel uses the same table layout for every method and is
restricted to facts about the clips on screen. The stimulus contains no verdict
and no gap estimate.

ONSET selects clips centred on transition onsets. HIGHLIGHTS selects clips
centred on high-importance states. DISAGREEMENTS selects clips centred on
disagreement states.

ABSTRACT is retained only as a descriptive baseline because it does not define
a native clip-selection rule.

The participant supplies the verdict, a magnitude in seconds, and a confidence
rating. A summary that stated its own verdict would leave the participant
nothing to do but copy it, which measures agreement with an assertion rather
than what the summary conveys. The comparison work's participants likewise
chose from clips rather than being shown a conclusion.

`n_steps` = 200 policy steps, which is 40 seconds of driving at 5 Hz, or ten
clips of 20 states. The budget is on what the summary **shows**, not on what
the selection algorithm reads: every method here scans the full corpus to
select from it, which is what `highlights()` and Algorithm 1 of the
DISAGREEMENTS paper both do. Simulated steps produced by DISAGREEMENTS
branching count against the shown budget when they are shown.

Policy 1 and policy 2 are the first and second condition named in the cell's
contrast label, so in `BAL - FD` policy 1 is BAL and policy 2 is FD. The
wording stays neutral rather than naming FD/BAL/SP, so that a participant
cannot infer the answer from the condition names.

### 4.1 What each method may show

With the modality equalised, the remaining risk moves into the numeric panel: a
panel that states policy 1's median onset and policy 2's median onset has
handed over the answer in two numbers, while a panel of importance scores
answers nothing.

**The numeric panel is restricted to facts about the clips on screen**, in the
same fields for every method:

| Field | Content |
| --- | --- |
| per clip (10 rows) | exposure label, clip start time in seconds, clip end time in seconds |
| per policy (2 rows) | number of distinct episodes represented, number of shown clips |

No aggregate over episodes that are not shown. No paired gap. No interval. No
selection score. No seed identifier — a participant who could see that one
method's five clips came from one training run and another's from five would be
answering a different question from the one asked.

Under this restriction, what distinguishes the three methods is only where
their clips are centred: ONSET on transition onsets, HIGHLIGHTS on states of
high `max_a Q - second_a Q`, DISAGREEMENTS on states where the two policies'
actions differ. D1 records that clip methods' windows contain the onset in 4.7%
to 43.0% of cases; ONSET's contain it by construction. Whether that lets a
person answer a timing question is what the study measures, and it is not
settled by construction: a participant must still compare two sets of five
clips from different episodes and judge which policy acted earlier.

Where a method's selected exposures for the two policies do not intersect, the
participant is comparing unmatched scenes. Nothing is done about this: it is a
real property of the selection rule, D3 records how often it happens, and
repairing it for the clip methods would turn a baseline into something its
authors did not propose.

### 4.2 ONSET's clip selection

ONSET is the only method whose selection rule is written here rather than taken
from a published algorithm, so it is specified in full and in advance.

**Stratified across seeds, then across difficulty.** Draw five seeds from the
twenty by taking every fourth in ascending order — 3100, 3104, 3108, 3112, 3116
for arm A and the corresponding seeds for B and C — then one exposure from each
selected seed. Seed spacing is fixed by position in the seed range, never by
any property of the results. This is the mechanism by which ONSET can represent
a condition rather than a network, and D9 records that the clip methods have no
equivalent.

**Which exposure within a seed.** Order that arm's 36 exposures by
time-to-contact at exposure, `nearest_front_distance / relative_closing_speed`,
and assign the five selected seeds to the exposures nearest the 10th, 30th,
50th, 70th, and 90th percentiles of that ordering, breaking ties by ascending
`exposure_id`. This spreads the sample across the difficulty range including
both ends, is deterministic, and uses the one axis all three arms share: the
frozen TTC cells apply to arm A directly, while arms B and C carry a single
scenario bin and would otherwise have no stratum to sample from. TTC is a
property of the scene, fixed before any policy runs, so the sample cannot be
steered by an outcome.

**Exposures are matched within a clip pair.** Both policies of an item are
shown the same (seed, exposure) pairs. Matched exposures are what the method is
defined on, in the same way that ranking each policy's states independently is
what HIGHLIGHTS is defined on.

**Where each clip is centred.** On that policy's onset in that exposure, using
the same windowing function as the clip methods: `[t - 10, t + 10)`, twenty
frames, clamped at the episode boundaries exactly as
`highlights_state_selection.py:57` clamps them. ONSET's window rule is
identical to HIGHLIGHTS'; only the centre differs.

**Exposures without an onset.** If a selected (seed, exposure) pair has no
onset for either policy — censored, terminal, or a collision — it is replaced
by the next exposure in that seed's TTC ordering, moving away from the median,
and the substitution is recorded. On the seeds checked so far this rule does not
fire: every one of the 36 exposures yields `valid_onset` for all three policies,
108 of 108 episodes per arm, with no censored, terminal, or collision outcomes.
The check is repeated over all twenty seeds of each arm before rendering, and
the result recorded here.

### 4.3 Cross-policy replay, and how it is verified

DISAGREEMENTS needs each policy's action values on the other policies' states,
and every method needs rendered frames. The stored step tables carry only the
acting policy's Q vector and no observations, so one deterministic replay pass
is required, implemented in `scripts/p1_replay_check.py`. Branching uses
`deepcopy` of the environment, as the upstream implementation does; no
state-reload utility is needed.

Verification is in two parts, because v1's single "match step for step"
requirement is already falsified by a numerical tie that has no bearing on
whether the environment was reconstructed correctly:

- **dynamics, strict.** Replayed `ego_position`, `ego_speed`, `ego_lane`, and
  episode length must match `evaluation/steps.csv` exactly. Status: 108/108
  episodes on each of arms A, B, and C for the seeds checked.
- **action labels, reported not tolerated.** Label mismatches are counted and
  each is recorded with the Q gap between the stored and replayed action. A gap
  below **1e-5** is recorded as a numerical tie. The threshold is set from the
  measured margin distribution in D7 and from float32 reproducibility, not from
  any FD/BAL/SP outcome. Labels are not corrected: they perturb the
  disagreement set and that perturbation is part of what D7 reports.

## 5. Cells, classification, and the equivalence bound

Primary measure: **onset timing in seconds**, the project's endpoint. The
onset-gap in metres is a separate corroborating measure and is not mixed in.

Produced by `scripts/preregistration_cell_bounds.py`, output in
`outputs/summary_comparison/preregistration_cell_bounds.csv`. A cell is
`effect` only if the seed-level sign test reaches p < 0.05 **and** the
across-seed bootstrap interval excludes zero; otherwise `null`.

| Arm | Contrast | Median (s) | 95% CI | +/-/tie | p | Class |
| --- | --- | ---: | --- | ---: | ---: | --- |
| A | BAL - FD | +1.20 | [+0.95, +1.40] | 18/1/1 | 7.6e-05 | effect |
| A | SP - BAL | +0.00 | [-0.40, +0.40] | 8/9/3 | 1.00 | null |
| A | SP - FD | +1.15 | [+1.00, +1.40] | 20/0/0 | 2.0e-06 | effect |
| B | BAL - FD | -0.65 | [-1.30, +1.05] | 8/12/0 | 0.50 | null |
| B | SP - BAL | +0.20 | [-0.25, +0.70] | 10/7/3 | 0.63 | null |
| B | SP - FD | +0.20 | [-1.40, +0.95] | 11/9/0 | 0.82 | null |
| C | BAL - FD | -1.05 | [-2.60, +0.00] | 3/13/4 | 0.021 | **null, borderline** |
| C | SP - BAL | -2.00 | [-3.85, -1.00] | 0/20/0 | 2.0e-06 | effect |
| C | SP - FD | -3.80 | [-4.20, -3.15] | 0/20/0 | 2.0e-06 | effect |

Four effect cells, five null cells.

**Equivalence bound.** One bound applies to every cell: **0.4 s, two policy
steps at 5 Hz.** A directional claim whose magnitude exceeds it, made in a cell
classified null, is a false claim. The bound is fixed to the measurement
resolution of the endpoint, not derived from the data it judges. v1 derived it
per cell from the bootstrap interval, which made noisier cells more permissive:
the same claim was a false claim under a 0.40 s bound in arm A and free under a
2.60 s bound in arm C.

**Disclosed borderline: arm C, BAL - FD.** Its sign test reaches p = 0.021 but
its interval touches zero, so the conjunctive rule classifies it null. The two
criteria disagree, and so do the two measures: in metres the same contrast is
+3.9 m at 12/8 seeds, clearly null, while in seconds it is 13/3 in the same
direction. It is not used as an item.

**Prior justification, and one disclosure.** Front-distance weights are FD
1.00, BAL 0.70, SP 0.25 and speed weights are FD 0.45, BAL 0.70, SP 1.00.
FD/BAL is the closest pair on both axes, a reason for arm B's and arm C's
`BAL - FD` nulls. Arm B is the arm in which a lane change maximises the speed
and front-distance terms at once, so no condition has a reason to prefer a
different action there, a reason for all three of its nulls.

Arm A's `SP - BAL` has **no such prior reason**: those two conditions are
separated by 0.45 on the front-distance axis and 0.30 on the speed axis, more
than FD/BAL on both, so the weight argument predicts an effect and does not get
one. It is disclosed here rather than given a justification constructed after
the fact. It is used as an item because a null cell with no prior explanation
is the honest test of over-claiming, and any statement depending on it is
reported with and without it.

Whether a pair is behaviourally similar is task-dependent, not readable from
the weights: `BAL - FD` is an effect in arm A (+1.20 s) and null in arms B and
C. v1 cited the comparison work's exclusion of `<SD, CL>` as a precedent for
treating FD/BAL as the similar pair. That citation is withdrawn: their
exclusion was a display-budget decision in a within-subject human study, which
is the constraint section 3 applies to method count here, and their criterion
referenced the reward specification rather than a measured outcome.

## 6. The human experiment

**Method is within subjects**, following the comparison work's Experiment 2:
"A within subject setup was chosen in order to allow participants to provide a
direct comparison between the methods". Each participant sees all three
methods. Their n = 45 was a total, not a per-group figure.

**Participants.** Recruited through Prolific. **n = 72**, which divides evenly
into the nine counterbalancing combinations below at eight participants each.
Attention filters follow the comparison work's precedent: completion time below
a pre-set threshold, and a comprehension quiz on each summary method that must
be passed before its block begins. Exclusion criteria are fixed before
recruitment opens.

**Items.** Six per participant: one identification item and one over-claiming
item per method.

| Task | Cell | Across-seed median | Role |
| --- | --- | ---: | --- |
| identification | E1 = A `BAL - FD` | +1.20 s | hardest effect; where clip methods are expected to approach chance |
| identification | E2 = C `SP - BAL` | -2.00 s | intermediate |
| identification | E3 = C `SP - FD` | -3.80 s | easiest; ceiling check |
| over-claiming | N1 = A `SP - BAL` | +0.00 s | null with no prior reason, disclosed in section 5 |
| over-claiming | N2 = B `BAL - FD` | -0.65 s | large within-seed differences whose direction reverses across seeds (8 positive, 12 negative) |
| over-claiming | N3 = B `SP - FD` | +0.20 s | mixed within and across seeds |

The effect items are graded rather than uniformly easy. The comparison
literature's own user study shows clip summaries falling to 0.63 correctness on
its hardest agent pair against a 0.5 forced-choice floor, so an item set of only
large differences would put all three methods near ceiling and measure nothing.
E1 is the analogue of that hardest pair.

N2 is included deliberately. It is the cleanest instance of the phenomenon in
section 0.1: within nearly every seed the two conditions differ by a large and
internally consistent margin, and the direction reverses between seeds. Drawn
from the pooled corpus, its evidence is genuinely mixed, so *not
distinguishable* is an answer a participant can read off the screen.

**Counterbalancing.** Two 3x3 Latin squares, crossed, giving nine combinations
of eight participants each.

Method order, so that practice and fatigue are shared equally by the three
methods rather than accruing to whichever is shown first:

| Order group | first | second | third |
| --- | --- | --- | --- |
| S1 | ONSET | HIGHLIGHTS | DISAGREEMENTS |
| S2 | HIGHLIGHTS | DISAGREEMENTS | ONSET |
| S3 | DISAGREEMENTS | ONSET | HIGHLIGHTS |

Method-to-cell assignment, so that no method is systematically paired with the
easiest cell:

| Pairing group | ONSET | HIGHLIGHTS | DISAGREEMENTS |
| --- | --- | --- | --- |
| P1 | E1, N1 | E2, N2 | E3, N3 |
| P2 | E2, N2 | E3, N3 | E1, N1 |
| P3 | E3, N3 | E1, N1 | E2, N2 |

**Procedure.** Participants are introduced to the driving task and to the
premise that two teams each trained twenty driving agents under different
objectives, so that a set of clips drawn from several different networks of one
condition is intelligible rather than confusing. Each method block opens with
an explanation of that method and a comprehension check. For each item they see
the summary in the section 4 template and answer three things in a fixed order:
one verdict from the set *policy 1 earlier* / *policy 2 earlier* / *not
distinguishable*; a magnitude in seconds, with "cannot say" available; and a
confidence rating on a 7-point scale. A free-text reason is collected last, so
that composing it cannot change the answer above it. Clips can be paused,
replayed, and rewatched without limit. Participants are not told whether the
same condition appears in more than one item, and make each decision
independently.

**Ground truth.** Section 5, the across-seed quantities, which match the pooled
corpus the summaries are drawn from. Identification is correct when the verdict
matches the sign of the across-seed paired gap. Over-claiming is correct when
the verdict is *not distinguishable*; it is a **false claim** when a direction
is named and the stated magnitude exceeds 0.4 s.

**Hypotheses**, registered in advance:

- **H1** Participants identify the earlier policy more often from ONSET
  summaries than from HIGHLIGHTS or DISAGREEMENTS summaries.
- **H2** The gap in H1 is larger on E1 than on E3, because E3 is near ceiling
  for every method.
- **H3** Participants make fewer false claims on the null items from ONSET
  summaries than from HIGHLIGHTS or DISAGREEMENTS summaries.
- **H4** No prediction is registered for preference or satisfaction. The
  comparison work found no clear preference between its two methods and this
  study is not powered to detect one.

H3 is registered as a directional prediction but is genuinely open. ONSET's
stratified draw exposes the disagreement between seeds, which should support
abstention; its clips are also the most legible, which may instead invite a
confident reading of sampling noise. **An outcome in which ONSET makes the most
false claims is a reportable result, and is the reason the null items are in the
design**: they are the only items on which the proposed method can lose, and a
comparison in which the proposed method cannot lose is not a comparison.

**Analysis.** Within-subject comparisons across methods use Wilcoxon
signed-rank tests, with rank-biserial correlation as the effect size and
bootstrapped 95% intervals at 2000 samples and seed 12345, following the
non-parametric approach of the comparison work. Identification and
over-claiming are analysed separately and both are primary; a method that wins
one and loses the other has not won. Order group and pairing group enter as
checks that counterbalancing worked, not as hypotheses. Confidence and
free-text reasons are reported descriptively.

**Power.** With n = 72 in a within-subject design the study can detect a
moderate difference between methods and cannot resolve a small one. This is
stated in advance: the comparison work at n = 45 obtained p = 0.014 on one item
and p = 0.187 on another. An outcome in which the three methods are
indistinguishable is a reportable result under section 9.

## 7. Computational diagnostics (descriptive, not scored)

These are properties of the selection algorithms, computed without generating
any verdict and without reference to any FD/BAL/SP contrast. They are reported
as mechanism — the role section 8 assigns to onset coverage — and **never as a
ranking**.

**D1 Onset coverage.** Share of clip context windows containing the episode's
onset, over all 20 seeds: 4.7% and 9.3% in arm A for `max_min` and
`max_second`, 43.0% and 31.3% in arm B, 24.3% and 28.7% in arm C. Per condition
the range is 0.0% to 52.0%.

**D2 Importance-peak alignment.** Hit rate within 10 steps of the onset against
the chance rate for the same episodes: lift ranges from 0.0x to 17.0x and is
bimodal, not a single multiplier. SP is at or below chance in arms A and C
(0.0x to 0.3x) while every other condition in arms B and C is 9.5x to 17.0x.
The SP condition, where alignment is worst, is the condition carrying the
largest timing effects. The chance denominator currently runs over 600-step
episodes that are largely post-event cruising; section 11 resolves the analysis
window before this number is reported.

**D3 Paired-design breakage.** Exposures selected for the two policies of a
pair intersect in a per-seed mean of 0.7 to 3.1 out of 5, and not at all in up
to 13 of 20 seeds (arm C). HIGHLIGHTS ranks importance independently per
policy, so nothing makes the two selections land on shared scenes.

**D4 Effective sample.** The five selected states come from a mean of 4.3
distinct episodes in arm A, and in the worst case from a single episode.

**D5 Disagreement density.** The disagreement filter of Algorithm 1 retains 92%
to 99% of states for every pair except arm B's `BAL - SP`, which retains 7%.
The filter therefore does not restrict the candidate pool the way the method
presumes, and DISAGREEMENTS here differs from HIGHLIGHTS mainly in its
importance function rather than in its selection mechanism. Recorded before the
study so that any similarity between the two methods' results is read as a
property of this setting rather than as an explanation found afterwards.

**D6 Leader asymmetry.** Disagreement counts depend strongly on which policy
drives: arm A `SP` as leader against `FD` gives 5.5 disagreements per episode
while `FD` as leader against `SP` gives 598.5, a factor of 100. Both directions
are generated, as the upstream implementation does.

**D7 Action-label ties.** Replay reproduces the stored rollouts exactly on
dynamics but not always on the action label: one step in 64,800 flipped between
`SLOWER` and `IDLE` where the two Q values differed by 1e-6, with no effect on
the trajectory. Across the stored tables, the share of steps whose top two Q
values differ by less than 1e-5 is 0.00% to 0.05%, and by less than 0.01 is up
to 99% for arm B's `BAL` and `SP`. The disagreement criterion is a hard argmax
comparison and is therefore unstable exactly where the policies are
indifferent. The criterion is **not** modified: it is kept faithful to the
upstream implementation and the tie rate is reported alongside.

**D8 Seed reversal.** Per-seed median gaps for the null cells span both signs
with large magnitudes on each side: arm B `BAL - FD` runs from -2.60 s to
+2.30 s at 8 seeds positive and 12 negative; arm C `BAL - FD` from -4.20 s to
+3.20 s at 3/13/4; arm A `SP - BAL` from -1.20 s to +1.60 s at 8/9/3. Within
almost every individual seed the difference is large and internally consistent.
A conclusion drawn from one trained pair therefore reverses on retraining. This
is the finding that motivates the pooled corpus of section 0.1, and it is a
statement about evaluation practice in this literature rather than about any
one method's fidelity.

**D9 Pooled selection concentrates on one network.** Applied to a condition's
pooled twenty-seed corpus, HIGHLIGHTS' global top five clips come from a single
seed in 9 of 18 arm-condition-variant cases, from two seeds in 8, and from
three in 1; never from more than three of the twenty. The mechanism is that
`max_a Q - second_a Q` is not normalised across traces, so whichever network
has the widest Q spread dominates the global ranking. A summary that a reader
takes to characterise a training recipe is in practice a summary of one
checkpoint, and nothing in the output says so. DISAGREEMENTS normalises Q per
agent (PROVENANCE 3.6); its concentration is measured separately before the
study, and section 11 records that the wording here depends on the result.

## 8. Excluded by construction

**Whether a summary contains the onset is not a scored outcome.** ONSET is
defined by onsets, so scoring methods on onset coverage and finding that the
onset summariser wins is circular. D1 measures it and it is reported as
descriptive mechanism.

**No oracle upper bound is computed.** v1 proposed one to separate selection
failure from readout failure. With a human tier the readout is measured
directly, and the oracle as v1 defined it reduces to a restatement of D1, which
this section excludes from scoring.

Also excluded: any metric that references the FD/BAL/SP contrast in its
definition, mirroring the model-selection gate's rule from experiment 1.

## 9. Claims the experiment cannot support

Recorded in advance so they are not quietly attempted later.

- **That the clip baselines were designed for this task.** They were not.
  HIGHLIGHTS and DISAGREEMENTS summarise a single trained agent; here they are
  given a condition's pooled twenty-network corpus, which is beyond the scope
  their authors defined. The extension is applied identically to all three
  methods, is disclosed here, and its cost to the baselines is measured in D9
  rather than asserted. A single-pair replication under the comparison work's
  own setup is a follow-up study, not part of this one.
- That ONSET conveys timing better than clip summaries **in general**. The
  question posed is a timing question, chosen because it is the question this
  project's instrument addresses. Every method in this literature is evaluated
  on the question it was designed for; the defensible claim is scoped to timing
  comparisons.
- That clip summaries are blind to transitions. D2 measures alignment above
  chance for most conditions, so the defensible statement is that the selection
  is imprecise and its offset is task-dependent, not that it is uninformative.
- That any method conveys the answer under a smaller evidence budget.
  `n_steps` is fixed at 200 and not swept; the compactness claim of the
  canonical proposal is not tested here.
- That any method is more stable across training seeds. Stability is a property
  of the underlying policies here, measured in D8, not of the summaries.
- That the results generalise beyond the development grid. The held-out grid is
  not opened by this experiment.
- That participants prefer any method. No prediction is registered (H4).

An outcome in which all methods perform comparably is a reportable result. The
comparison work's own conveying-differences experiment found DISAGREEMENTS "at
least equivalently useful" rather than superior, and no clear participant
preference. The contribution is the map of where each method fails, which
section 7 already populates, not a ranking.

## 10. Freeze procedure

1. Resolve the open items in section 11.
2. Edit the canonical proposal `docs/proposals/10Jul_transition_timing.docx` to
   match section 0 and the already-updated `docs/user_requirements.md`, so the
   change of instrument is recorded where the commitment lives.
3. Obtain UNSW HREC approval. Record the approval number here.
4. Record the SHA-256 of this file, of
   `outputs/summary_comparison/preregistration_cell_bounds.csv`, and of the
   scripts that produced it: `scripts/preregistration_cell_bounds.py` and
   `scripts/aggregate_selected_20seed.py`.
5. Commit and **push**, so the freeze carries a timestamp the author does not
   control. A local commit date does not. This directory is tracked; `docs/` is
   not, which is why the pre-registration lives here.
6. Only then generate summaries and open recruitment.

## 11. Open items before freezing

- [ ] Common analysis window for the chance baseline in D2. Selection is
      unaffected, but hit-rate denominators currently run over 600-step
      episodes that are largely post-event cruising, which inflates the
      reported lift.
- [ ] Rendering specification: exact panel layout, clip pixel size, frame rate,
      and the rendering of the strict-parity numeric table of section 4.1. The
      template is fixed in section 4; the instantiation is not, and it must be
      frozen before any summary is rendered.
- [ ] Pilot with 6 to 9 participants to measure completion time, which sets the
      per-participant payment. Six items of ten clips each may exceed the
      15-minute task length a 3 USD payment corresponds to on Prolific; the
      levers are clip length, whether replay is capped, and whether the
      free-text reason is optional. The pilot contributes no data to the
      analysis.
- [ ] DISAGREEMENTS' seed concentration under the pooled corpus, the D9
      measurement for the method that does normalise Q per agent. If it also
      concentrates on one or two seeds, D9 is a property of importance-ranked
      selection in general rather than of HIGHLIGHTS' missing normalisation,
      and the wording of D9 changes accordingly.
- [ ] Replay verification and the onset-availability check of section 4.2
      extended from three seeds to all sixty run directories.
- [ ] Tie handling in the sign test underlying section 5: the frozen primary
      follows experiment 1 and drops ties, with a tie-inclusive sensitivity
      version reported alongside.
