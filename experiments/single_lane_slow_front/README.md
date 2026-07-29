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

After the local validation gate passes, submit five paired seeds on Katana.
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
