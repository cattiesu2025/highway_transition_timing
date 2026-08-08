# Single-Lane Slow-Front Experiment

This isolated experiment keeps the single-lane longitudinal sanity check out of
the main Highway adapter.

It trains FD/BAL/SP in `highway-v0` with:

- one lane;
- a deterministic, difficulty-stratified 20-reset training block;
- separate no-front, non-closing, near-closing, visible slow-front, and
  delayed-visible scenarios;
- no lateral actions in the DQN action space;
- longitudinal target speeds at `10,15,20,25,30,35 m/s`;
- normalized relative kinematics observations by default;
- `slowdown_onset` as the timing target.

The action space is:

```text
SLOWER, IDLE, FASTER
```

`duration` and `evaluation-duration` are physical seconds. With the default
5Hz policy frequency, a 120-second evaluation contains at most 600 policy
steps.

Audit 2,000 generated resets and run the immediate-`SLOWER` feasibility oracle
before training:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py audit \
  --out outputs/single_lane_slow_front_stratified_audit_seed0 \
  --num-resets 2000 \
  --seed 0
```

Run a short local FD-only smoke:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py \
  --out outputs/single_lane_slow_front_stratified_smoke_seed0 \
  --agents FD \
  --timesteps 1000 \
  --num-exposures 4 \
  --duration 20 \
  --evaluation-duration 20 \
  --bootstrap-samples 20 \
  --no-figures \
  --verbose 0
```

Run one frozen 100K seed:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py \
  --out outputs/single_lane_slow_front_stratified_100k_seed0 \
  --timesteps 100000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --seed 0 \
  --bootstrap-samples 500
```

## 20-Second Training-Horizon Diagnostic

This isolated diagnostic keeps the total training budget at 100,000 policy
steps but shortens each fixed training episode from 120 s to 20 s. Evaluation
remains 120 s. At 5 Hz this changes the nominal episode cap from 600 to 100
policy steps. The training wrapper enforces this integer step cap explicitly,
avoiding an extra step from floating-point time accumulation. This produces
roughly 1,000 reset exposures per non-terminal agent
run instead of roughly 167 full-duration exposures. Reward weights, the
stratified block, Double DQN hyperparameters, target speeds, and the evaluation
grid remain unchanged.

After the local audit and smoke checks pass, submit the isolated 20-seed Katana
array:

```bash
qsub scripts/katana_single_lane_duration20_20seed.pbs
```

Array indices `0-19` map directly to run seeds `3000-3019`. Each run
initializes its training RNG once and samples one continuous scenario stream;
FD/BAL/SP replay the same within-run stream prefix for paired comparison. Each
task audits 2,000 specifications, trains the 1x FD/BAL/SP policies for 100,000
total steps with `--duration 20`, evaluates 36 original slow-front scenes with
`--evaluation-duration 120`, and verifies the three saved model archives.
Outputs are written to:

```text
/srv/scratch/$USER/highway_transition_timing/outputs/
  single_lane_slow_front_duration20_100k_seed<seed>/
```

Do not rename these directories to the historical `stratified_100k` names.
Run no-front, matched-speed, and far-front counterfactuals only after the
training-integrity and original development checks pass.

## Reward-Strength Validation

Use `--reward-strength-multiplier` only for the approved `{1,2,4}` validation.
FD scales its front-distance term, SP scales its speed term, and BAL scales its
complete effective reward as a numerical-scale negative control. The scenario
block and all training/evaluation settings remain unchanged.

Run the two new seed-0 pilot levels without overwriting the existing 1x run:

```bash
for MULTIPLIER in 2 4; do
  PYTHONPATH=src python experiments/single_lane_slow_front/run.py \
    --out "outputs/single_lane_slow_front_reward_strength_${MULTIPLIER}x_100k_pilot_seed0" \
    --reward-strength-multiplier "${MULTIPLIER}" \
    --timesteps 100000 \
    --num-exposures 36 \
    --duration 120 \
    --evaluation-duration 120 \
    --seed 0 \
    --bootstrap-samples 500
done
```

Pass the same multiplier to `counterfactual` and `rollout-counterfactual` when
evaluating each new run so reward traces and metadata use its trained weights.

Generate the seed-0 training, validation, and requested two-panel
counterfactual figures with R:

The plotting scripts are maintained locally under the ignored `figures/`
workspace and are intentionally not distributed through Git.

```bash
R_LIBS_USER=tmp/r-lib Rscript \
  figures/single_lane/plot_reward_strength_validation.R
```

The completed seed-0 pilot failed the predeclared directional and
matched-speed opening gates: FD T50 was `1.0/1.2/0.0 s` at `1x/2x/4x`, and FD
opened with `SLOWER` in `9/36` matched-speed scenes at both 2x and 4x. SP moved
monotonically later (`1.0/1.4/2.4 s`) and every no-front cell remained `0/36`,
but these partial passes did not authorize the five-seed expansion. Seed 0
figure outputs and source-data CSVs are under
`outputs/single_lane_slow_front_reward_strength_seed0/figures/`.

The approved seed-1 diagnostic then repeated the FD specificity failure:
matched-speed opening `SLOWER` counts were `9/36` at 2x and `15/36` at 4x.
Although seed-1 FD T50 changed monotonically earlier (`1.4/0.0/0.0 s`), the
4x SP policy had `0/36` valid original-scene onsets and `36/36` collision
failures. The same 4x SP failure occurred for every matched-speed and far-front
rollout, while no-front remained `0/36` onset with no failure. This confirms a
design-level reward/safety-tradeoff problem rather than a seed-0-only anomaly.
Do not train seeds 2--4 with this intervention; revise the reward-strength
design and repeat audit, smoke, and paired seed-0/seed-1 pilots first.

Only after a revised local validation gate passes should paired seeds be
submitted on Katana.
Install `requirements-pilot.txt` in the active Katana environment before
submission; each array task then audits and trains one seed, while all three
reward conditions in a seed share the same scenario schedule.

```bash
module load python/3.11.3
source "/srv/scratch/${USER}/venvs/highway-transition/bin/activate"
qsub scripts/katana_single_lane_stratified.pbs
```

The array uses seeds `0-4` and writes
`/srv/scratch/$USER/highway_transition_timing/outputs/single_lane_slow_front_stratified_100k_seed<seed>`.
Adjust the OpenPBS walltime, memory, project, or queue directives if required
by the Katana allocation, but keep the experiment arguments frozen.

The default `stratified` block contains 4 no-front, 4 non-closing-front,
2 near-closing-front, 8 visible slow-front, and 2 delayed-visible-front
resets. The eight visible positive cases contain two resets from each feasible
TTC/required-deceleration cell: `gradual/gentle`, `easy/gentle`,
`medium/moderate`, and `hard/strong`. TTC and required deceleration control
scenario coverage and analysis strata; Double DQN still learns only the Q
values implied by observation, action, reward, and next observation.

Use `--training-scenario-profile legacy-random` to reproduce the previous
probabilistic mixture. The legacy `--no-front-train-fraction` and
`--near-matched-speed-train-fraction` flags only affect that profile.

Run front-vehicle counterfactuals on a completed run:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py counterfactual \
  --run-dir outputs/single_lane_slow_front_controls_100k \
  --num-exposures 36
```

Run full-episode front-vehicle counterfactuals on a completed run:

```bash
PYTHONPATH=src python experiments/single_lane_slow_front/run.py rollout-counterfactual \
  --run-dir outputs/single_lane_slow_front_controls_100k \
  --num-exposures 36 \
  --bootstrap-samples 500
```

Counterfactual variants:

- `original`: keep the slow front vehicle;
- `no-front`: remove the front vehicle;
- `matched-speed-front`: keep the front vehicle but set its speed to ego speed;
- `far-front`: keep the slow front vehicle but place it at least 260m ahead.

The expected sanity pattern is that `SLOWER` should be most likely in
`original`, and should drop under `no-front`, `matched-speed-front`, and
`far-front`.

Outputs:

- `models/`: one Double DQN model per reward condition;
- `training_logs/`: Stable-Baselines3 monitor and progress CSVs;
- `training_scenarios.csv`: every reset's family, physical state, TTC,
  required deceleration, visibility, seed, and difficulty labels;
- `training_runs.csv`: frozen training configuration and observed family
  counts;
- `evaluation/steps.csv` and `evaluation/exposures.csv`;
- `analysis/episode_outcomes.csv`;
- `analysis/gap_summary.csv`;
- `analysis/paired_timing_gaps.png`;
- `analysis/training_convergence.png`.
- `analysis/report_rollout_counterfactual_slowdown_response.{png,svg,pdf}`;
- `counterfactual_front_vehicle/counterfactual_initial_actions.csv`;
- `counterfactual_front_vehicle/counterfactual_action_summary.csv`.
- `rollout_counterfactual_front_vehicle/<variant>/evaluation/steps.csv`;
- `rollout_counterfactual_front_vehicle/<variant>/analysis/episode_outcomes.csv`;
- `rollout_counterfactual_front_vehicle/<variant>/analysis/gap_summary.csv`;
- `rollout_counterfactual_front_vehicle/counterfactual_rollout_summary.csv`.

## Five-seed R figures

After the five `single_lane_slow_front_stratified_100k_seed{0..4}` directories
are available locally, generate the cross-seed training and counterfactual
figures with the local, Git-ignored plotting script:

```bash
R_LIBS_USER=tmp/r-lib Rscript \
  figures/single_lane/plot_multiseed_results.R
```

The script requires `ggplot2`, `patchwork`, `dplyr`, `tidyr`, `readr`,
`svglite`, `ragg`, and `scales`. It writes editable SVG/PDF, 600-dpi TIFF,
300-dpi PNG, and source-data CSV files under
`outputs/single_lane_slow_front_stratified_100k_multiseed/figures/`.
