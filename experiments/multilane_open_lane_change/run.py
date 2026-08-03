"""Isolated open-lane multi-lane lane-change timing experiment.

Training mixes slow-front episodes with no-front cruise episodes in a
controlled multi-lane scene. Matched evaluation episodes contain the ego
vehicle and one slower front vehicle, with empty adjacent lanes and no random
traffic. The goal is to measure whether FD/BAL/SP policies produce different
lane-change timing when lane change is available.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.constants import AGENTS, LANE_CHANGE_ONSET_TARGET, SLOW_ACTIONS
from highway_transition_timing.highway_adapter import (
    action_name_from_env,
    configure_headless_runtime,
    dqn_class_for_variant,
    extract_diagnostics,
    finite_or_blank,
    predict_action_and_scores,
    require_highway_deps,
    reward_components,
    step_row_from_diagnostics,
    weighted_reward,
)
from highway_transition_timing.io import write_csv_rows
from highway_transition_timing.pipeline import PipelineResult, run_pipeline
from highway_transition_timing.plotting import (
    write_optional_figures,
    write_training_convergence_figures,
)
from highway_transition_timing.rewards import (
    MAIN_REWARD_WEIGHTS,
    RewardWeights,
    reward_config_table_with_slow_down_penalty,
    with_common_slow_down_penalty,
)
from highway_transition_timing.utils import json_dumps

try:
    import gymnasium as gym

    _GymWrapper = gym.Wrapper
except ImportError:
    _GymWrapper = object


LANE_CHANGE_ACTIONS = {"LANE_LEFT", "LANE_RIGHT"}


@dataclass(frozen=True)
class ExperimentConfig:
    duration: int = 120
    evaluation_duration: int = 120
    seed: int = 0
    lanes_count: int = 4
    ego_lane: int = 1
    policy_frequency: int = 5
    simulation_frequency: int = 15
    observation_normalize: bool = True
    observation_absolute: bool = False
    slow_down_penalty: float = 0.2
    collision_risk_penalty: float = 3.0
    collision_penalty: float | None = None
    dqn_variant: str = "double-dqn"
    learning_rate: float = 5e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 32
    gamma: float = 0.95
    train_freq: int = 1
    gradient_steps: int = 1
    n_steps: int = 1
    target_update_interval: int = 250
    exploration_initial_eps: float = 1.0
    exploration_fraction: float = 0.25
    exploration_final_eps: float = 0.05
    no_front_train_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.duration <= 0 or self.evaluation_duration <= 0:
            raise ValueError("duration values must be positive seconds")
        if self.policy_frequency <= 0 or self.simulation_frequency <= 0:
            raise ValueError("simulation frequencies must be positive")
        if self.lanes_count < 2:
            raise ValueError("lanes_count must be >= 2 for a multi-lane experiment")
        if not 0 <= self.ego_lane < self.lanes_count:
            raise ValueError("ego_lane must be a valid lane index")
        if not 0.0 <= self.no_front_train_fraction <= 1.0:
            raise ValueError("no_front_train_fraction must be in [0, 1]")

    @property
    def policy_step_seconds(self) -> float:
        return 1.0 / self.policy_frequency

    @property
    def evaluation_max_policy_steps(self) -> int:
        return int(round(self.evaluation_duration * self.policy_frequency))


@dataclass(frozen=True)
class OpenLaneSpec:
    exposure_id: str
    exposure_seed: int
    ego_speed: float
    front_distance: float
    front_speed: float
    ego_lane: int = 1
    ego_longitudinal: float = 100.0
    exposure_t: int = 0
    scenario_type: str = "open_lane_slow_front"
    include_front_vehicle: bool = True


class OpenLaneTrainingResetWrapper(_GymWrapper):
    """Replace training resets with sampled slow-front or no-front scenes."""

    def __init__(self, env, config: ExperimentConfig):
        if _GymWrapper is object:
            self.env = env
        else:
            super().__init__(env)
        self.config = config
        self.reset_count = 0
        self.last_training_exposure: OpenLaneSpec | None = None
        self.training_variant_counts: Counter[str] = Counter()

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        spec = make_training_spec(self.reset_count, self.config)
        self.reset_count += 1
        self.last_training_exposure = spec
        self.training_variant_counts[spec.scenario_type] += 1
        obs = apply_open_lane_scene(
            self.env,
            spec,
            include_front_vehicle=spec.include_front_vehicle,
        )
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


class OpenLaneRewardWrapper(_GymWrapper):
    """Local reward wrapper using the project FD/BAL/SP reward weights."""

    def __init__(self, env, agent_condition: str, weights: RewardWeights):
        if _GymWrapper is object:
            self.env = env
        else:
            super().__init__(env)
        self.agent_condition = agent_condition
        self.weights = weights
        self.last_reward_components: dict[str, float] = {}

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        diagnostics = extract_diagnostics(self.env)
        components = reward_components(diagnostics)
        action_name = action_name_from_env(self.env, int(action))
        if action_name in LANE_CHANGE_ACTIONS:
            components["lane_change_penalty"] = -1.0
        if action_name in SLOW_ACTIONS:
            components["slow_down_penalty"] = -1.0
        reward = weighted_reward(components, self.weights)
        self.last_reward_components = components
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--num-exposures", type=int, default=36)
    parser.add_argument("--agents", nargs="+", default=list(AGENTS))
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Training episode duration in physical seconds.",
    )
    parser.add_argument(
        "--evaluation-duration",
        type=int,
        default=120,
        help="Evaluation horizon in physical seconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lanes-count", type=int, default=4)
    parser.add_argument("--ego-lane", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--collision-risk-penalty", type=float, default=3.0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--slow-down-penalty", type=float, default=0.2)
    parser.add_argument("--absolute-observation", action="store_true")
    parser.add_argument("--no-normalize-observation", action="store_true")
    parser.add_argument(
        "--no-front-train-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of training resets with no front vehicle. "
            "Use 0.0 to reproduce the original always-front baseline."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        duration=args.duration,
        evaluation_duration=args.evaluation_duration,
        seed=args.seed,
        lanes_count=args.lanes_count,
        ego_lane=args.ego_lane,
        observation_normalize=not args.no_normalize_observation,
        observation_absolute=args.absolute_observation,
        slow_down_penalty=args.slow_down_penalty,
        collision_risk_penalty=args.collision_risk_penalty,
        collision_penalty=args.collision_penalty,
        learning_rate=args.learning_rate,
        no_front_train_fraction=args.no_front_train_fraction,
    )


def make_env(agent_condition: str, config: ExperimentConfig, training: bool = False):
    require_highway_deps(include_training=False)
    import gymnasium as gym
    import highway_env  # noqa: F401

    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(
        {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 4,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": config.observation_absolute,
                "normalize": config.observation_normalize,
                "order": "sorted",
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
            "lanes_count": config.lanes_count,
            "vehicles_count": 1,
            "vehicles_density": 1.0,
            "duration": config.duration if training else config.evaluation_duration,
            "policy_frequency": config.policy_frequency,
            "simulation_frequency": config.simulation_frequency,
            "collision_reward": 0.0,
            "right_lane_reward": 0.0,
            "lane_change_reward": 0.0,
            "high_speed_reward": 0.0,
            "normalize_reward": False,
            "offroad_terminal": True,
        }
    )
    if training:
        env = OpenLaneTrainingResetWrapper(env, config)
    weights = with_common_slow_down_penalty(
        MAIN_REWARD_WEIGHTS[agent_condition],
        config.slow_down_penalty,
        config.collision_risk_penalty,
        config.collision_penalty,
    )
    return OpenLaneRewardWrapper(env, agent_condition, weights)


def make_training_spec(reset_index: int, config: ExperimentConfig) -> OpenLaneSpec:
    rng = random.Random(config.seed + 1009 * reset_index)
    include_front_vehicle = rng.random() >= config.no_front_train_fraction
    return OpenLaneSpec(
        exposure_id=f"train_reset_{reset_index:06d}",
        exposure_seed=config.seed + reset_index,
        ego_lane=config.ego_lane,
        ego_speed=round(rng.uniform(24.0, 31.0), 3),
        front_distance=round(rng.uniform(120.0, 280.0), 3),
        front_speed=round(rng.uniform(10.0, 20.0), 3),
        scenario_type="train_slow_front"
        if include_front_vehicle
        else "train_no_front",
        include_front_vehicle=include_front_vehicle,
    )


def make_eval_specs(num_exposures: int, config: ExperimentConfig) -> list[OpenLaneSpec]:
    front_distances = [140.0, 180.0, 220.0, 260.0]
    front_speeds = [10.0, 14.0, 18.0]
    ego_speeds = [26.0, 28.0, 30.0]
    specs: list[OpenLaneSpec] = []
    for idx in range(num_exposures):
        specs.append(
            OpenLaneSpec(
                exposure_id=f"M{idx:04d}",
                exposure_seed=config.seed + idx,
                ego_lane=config.ego_lane,
                ego_speed=ego_speeds[
                    (idx // (len(front_distances) * len(front_speeds)))
                    % len(ego_speeds)
                ],
                front_distance=front_distances[idx % len(front_distances)],
                front_speed=front_speeds[(idx // len(front_distances)) % len(front_speeds)],
            )
        )
    return specs


def apply_open_lane_scene(
    env,
    spec: OpenLaneSpec,
    include_front_vehicle: bool = True,
    front_distance: float | None = None,
    front_speed: float | None = None,
    reset_time: bool = True,
):
    from highway_env.vehicle.behavior import IDMVehicle

    unwrapped = env.unwrapped
    road = unwrapped.road
    lane_index = lane_index_for(spec.ego_lane)
    lane = road.network.get_lane(lane_index)
    ego = unwrapped.action_type.vehicle_class(
        road,
        lane.position(spec.ego_longitudinal, 0),
        lane.heading_at(spec.ego_longitudinal),
        spec.ego_speed,
        target_lane_index=lane_index,
        target_speed=spec.ego_speed,
    )
    vehicles = [ego]
    if include_front_vehicle:
        lead_distance = spec.front_distance if front_distance is None else front_distance
        lead_speed = spec.front_speed if front_speed is None else front_speed
        lead_longitudinal = spec.ego_longitudinal + float(lead_distance)
        lead = IDMVehicle(
            road,
            lane.position(lead_longitudinal, 0),
            lane.heading_at(lead_longitudinal),
            float(lead_speed),
            target_lane_index=lane_index,
            target_speed=float(lead_speed),
            enable_lane_change=False,
        )
        vehicles.append(lead)
    road.vehicles = vehicles
    unwrapped.vehicle = ego
    unwrapped.controlled_vehicles = [ego]
    if reset_time:
        unwrapped.time = 0
    return unwrapped.observation_type.observe()


def lane_index_for(lane_id: int) -> tuple[str, str, int]:
    return ("0", "1", int(lane_id))


def train_agents(
    output_dir: Path,
    agents: Sequence[str],
    total_timesteps: int,
    config: ExperimentConfig,
    verbose: int,
) -> None:
    require_highway_deps(include_training=True)
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    dqn_class = dqn_class_for_variant(config.dqn_variant)
    models_dir = output_dir / "models"
    logs_dir = output_dir / "training_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for agent in agents:
        training_env = make_env(agent, config, training=True)
        agent_logs_dir = logs_dir / agent
        agent_logs_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(training_env, filename=str(agent_logs_dir / "monitor.csv"))
        env.reset(seed=config.seed)
        model = dqn_class(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=min(config.learning_starts, max(1, total_timesteps // 10)),
            batch_size=config.batch_size,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            n_steps=config.n_steps,
            target_update_interval=config.target_update_interval,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps,
            seed=config.seed,
            verbose=verbose,
        )
        model.set_logger(configure(str(agent_logs_dir), ["stdout", "csv"]))
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model_path = models_dir / f"{agent}_main.zip"
        model.save(model_path)
        env.close()
        training_counts = dict(
            sorted(getattr(training_env, "training_variant_counts", Counter()).items())
        )
        rows.append(
            {
                "agent_condition": agent,
                "policy_id": f"{agent}_main",
                "model_path": str(model_path),
                "monitor_path": str(agent_logs_dir / "monitor.csv"),
                "progress_path": str(agent_logs_dir / "progress.csv"),
                "total_timesteps": total_timesteps,
                "seed": config.seed,
                **{
                    f"reward_{key}": value
                    for key, value in asdict(MAIN_REWARD_WEIGHTS[agent]).items()
                },
                "reward_slow_down_penalty": config.slow_down_penalty,
                "reward_collision_risk_penalty": config.collision_risk_penalty,
                "reward_common_collision_penalty": config.collision_penalty
                if config.collision_penalty is not None
                else "",
                "experiment": "multilane_open_lane_change",
                "training_duration_seconds": config.duration,
                "evaluation_duration_seconds": config.evaluation_duration,
                "policy_frequency_hz": config.policy_frequency,
                "policy_step_seconds": round(config.policy_step_seconds, 6),
                "evaluation_max_policy_steps": config.evaluation_max_policy_steps,
                "training_no_front_fraction": config.no_front_train_fraction,
                "training_scene_counts": json_dumps(training_counts),
                "action_space": "LANE_LEFT,IDLE,LANE_RIGHT,FASTER,SLOWER",
                "blockers": "none",
                "background_vehicles": 0,
                "observation_normalize": config.observation_normalize,
                "observation_absolute": config.observation_absolute,
                "dqn_variant": config.dqn_variant,
            }
        )
    write_csv_rows(output_dir / "training_runs.csv", rows)


def evaluate_agents(
    output_dir: Path,
    agents: Sequence[str],
    num_exposures: int,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require_highway_deps(include_training=True)
    dqn_class = dqn_class_for_variant(config.dqn_variant)
    models = {
        agent: dqn_class.load(output_dir / "models" / f"{agent}_main.zip")
        for agent in agents
    }
    specs = make_eval_specs(num_exposures, config)
    exposure_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for spec in specs:
        exposure_recorded = False
        for agent in agents:
            env = make_env(agent, config, training=False)
            obs, _info = env.reset(seed=spec.exposure_seed)
            obs = apply_open_lane_scene(env, spec)
            if not exposure_recorded:
                exposure_rows.append(exposure_row_from_spec(env, spec, config))
                exposure_recorded = True

            model = models[agent]
            policy_id = f"{agent}_main"
            episode_id = f"{agent}_{spec.exposure_id}_r0"
            done = False
            t = 0
            lane_change_action_count = 0
            while not done and t < config.evaluation_max_policy_steps:
                action, q_values = predict_action_and_scores(model, obs, deterministic=True)
                action_name = action_name_from_env(env, int(action))
                if action_name in LANE_CHANGE_ACTIONS:
                    lane_change_action_count += 1

                pre_step_diagnostics = extract_diagnostics(env)
                next_obs, reward, terminated, truncated, _info = env.step(int(action))
                post_step_diagnostics = extract_diagnostics(env)
                environment_done = bool(terminated or truncated)
                reached_horizon = t + 1 >= config.evaluation_max_policy_steps
                done = environment_done or reached_horizon
                collision = bool(post_step_diagnostics["collision_flag"])
                row = step_row_from_diagnostics(
                    diagnostics=pre_step_diagnostics,
                    post_step_diagnostics=post_step_diagnostics,
                    agent_condition=agent,
                    policy_id=policy_id,
                    episode_id=episode_id,
                    exposure_id=spec.exposure_id,
                    exposure_seed=spec.exposure_seed,
                    rollout_id="r0",
                    rollout_seed=0,
                    t=t,
                    action_name=action_name,
                    action_scores=q_values,
                    reward_total=reward,
                    reward_components=getattr(env, "last_reward_components", {}),
                    collision_flag=collision,
                    done=done,
                    termination_reason=episode_termination_reason(
                        collision=collision,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        reached_horizon=reached_horizon,
                    ),
                    lane_change_count=lane_change_action_count,
                    exposure_t=spec.exposure_t,
                )
                row["post_ego_lane"] = post_step_diagnostics["ego_lane"]
                row["lane_delta_after_step"] = (
                    int(post_step_diagnostics["ego_lane"])
                    - int(pre_step_diagnostics["ego_lane"])
                )
                row["actual_lane_changed_after_step"] = (
                    row["lane_delta_after_step"] != 0
                )
                row["environment_terminated"] = bool(terminated)
                row["environment_truncated"] = bool(truncated)
                row["analysis_horizon_reached"] = reached_horizon
                row.update(policy_time_fields(t, spec.exposure_t, config))
                row["blockers"] = "none"
                row["background_vehicles"] = 0
                step_rows.append(row)
                obs = next_obs
                t += 1
            env.close()

    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(eval_dir / "steps.csv", step_rows)
    write_csv_rows(eval_dir / "exposures.csv", exposure_rows)
    return step_rows, exposure_rows


def exposure_row_from_spec(
    env,
    spec: OpenLaneSpec,
    config: ExperimentConfig,
) -> dict[str, Any]:
    diagnostics = extract_diagnostics(env)
    front_speed = diagnostics["front_vehicle_speed"]
    relative_closing = (
        max(0.0, float(diagnostics["ego_speed"]) - float(front_speed))
        if front_speed is not None
        else 0.0
    )
    return {
        "exposure_id": spec.exposure_id,
        "exposure_seed": spec.exposure_seed,
        "exposure_t": spec.exposure_t,
        "ego_lane": diagnostics["ego_lane"],
        "ego_speed_at_exposure": round(float(diagnostics["ego_speed"]), 6),
        "nearest_front_distance_at_exposure": finite_or_blank(
            diagnostics["nearest_front_distance"]
        ),
        "front_vehicle_speed_at_exposure": finite_or_blank(front_speed),
        "relative_closing_speed_at_exposure": round(relative_closing, 6),
        "vehicles_density": 0.0,
        "background_vehicle_state": json_dumps(
            {
                "source": "multilane_open_lane_change",
                "ego_lane": spec.ego_lane,
                "front_distance": spec.front_distance,
                "front_speed": spec.front_speed,
                "blockers": "none",
                "background_vehicles": 0,
                "lanes_count": config.lanes_count,
            }
        ),
        "exposure_source": "multilane_open_lane_change",
        "exposure_difficulty_bin": "open_lane_slow_front",
        "evaluation_duration_seconds": config.evaluation_duration,
        "policy_frequency_hz": config.policy_frequency,
        "policy_step_seconds": round(config.policy_step_seconds, 6),
        "evaluation_max_policy_steps": config.evaluation_max_policy_steps,
    }


def policy_time_fields(
    t: int,
    exposure_t: int,
    config: ExperimentConfig,
) -> dict[str, float | int]:
    return {
        "t_seconds": round(t * config.policy_step_seconds, 6),
        "post_step_t_seconds": round((t + 1) * config.policy_step_seconds, 6),
        "exposure_t_seconds": round(
            exposure_t * config.policy_step_seconds,
            6,
        ),
        "policy_frequency_hz": config.policy_frequency,
        "policy_step_seconds": round(config.policy_step_seconds, 6),
    }


def episode_termination_reason(
    collision: bool,
    terminated: bool,
    truncated: bool,
    reached_horizon: bool,
) -> str:
    if collision:
        return "collision"
    if truncated or reached_horizon:
        return "duration"
    if terminated:
        return "terminal"
    return ""


def write_analysis(
    output_dir: Path,
    steps: Sequence[dict[str, Any]],
    exposures: Sequence[dict[str, Any]],
    config: ExperimentConfig,
    bootstrap_samples: int,
    figures: bool,
) -> PipelineResult:
    result = run_pipeline(
        steps,
        exposures,
        AnalysisConfig(bootstrap_samples=bootstrap_samples),
        LANE_CHANGE_ONSET_TARGET,
        reward_config=reward_config_table_with_slow_down_penalty(
            config.slow_down_penalty,
            config.collision_risk_penalty,
            config.collision_penalty,
        ),
    )
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in result.tables().items():
        write_csv_rows(analysis_dir / f"{name}.csv", rows)
    write_csv_rows(
        analysis_dir / "actual_lane_change_summary.csv",
        summarize_actual_lane_changes(steps, config.policy_frequency),
    )
    write_csv_rows(analysis_dir / "action_summary.csv", summarize_actions(steps))
    if figures:
        write_optional_figures(
            analysis_dir,
            result.modes,
            result.timing_gaps,
            result.episode_outcomes,
        )
        write_training_convergence_figures(output_dir, analysis_dir)
    return result


def summarize_actual_lane_changes(
    steps: Sequence[Mapping[str, Any]],
    policy_frequency: int = 5,
) -> list[dict[str, Any]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in steps:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for episode_id, group in sorted(by_episode.items()):
        ordered = sorted(group, key=lambda row: int(row["t"]))
        if not ordered:
            continue
        initial_lane = int(ordered[0]["ego_lane"])
        initial_lateral_position = lateral_position(ordered[0])
        first_action_t = first_t(
            row for row in ordered if str(row["action"]) in LANE_CHANGE_ACTIONS
        )
        first_lateral_motion_t = (
            first_t(
                row
                for row in ordered
                if (
                    (position := lateral_position(row)) is not None
                    and abs(position - initial_lateral_position) > 0.05
                )
            )
            if initial_lateral_position is not None
            else None
        )
        first_actual_t = first_t(
            row
            for row in ordered
            if int(row.get("post_ego_lane", row["ego_lane"])) != initial_lane
            or int(row["ego_lane"]) != initial_lane
        )
        first_action = next(
            (
                str(row["action"])
                for row in ordered
                if str(row["action"]) in LANE_CHANGE_ACTIONS
            ),
            "",
        )
        first_actual_row = next(
            (
                row
                for row in ordered
                if int(row.get("post_ego_lane", row["ego_lane"])) != initial_lane
                or int(row["ego_lane"]) != initial_lane
            ),
            None,
        )
        actual_lane = (
            int(first_actual_row.get("post_ego_lane", first_actual_row["ego_lane"]))
            if first_actual_row is not None
            else None
        )
        actual_direction = (
            "LANE_RIGHT"
            if actual_lane is not None and actual_lane > initial_lane
            else "LANE_LEFT"
            if actual_lane is not None and actual_lane < initial_lane
            else ""
        )
        collision = any(parse_bool_like(row.get("collision_flag")) for row in ordered)
        rows.append(
            {
                "episode_id": episode_id,
                "agent_condition": ordered[0]["agent_condition"],
                "exposure_id": ordered[0]["exposure_id"],
                "initial_lane": initial_lane,
                "first_lane_change_action_t": blank_if_none(first_action_t),
                "first_lane_change_action_seconds": seconds_from_decision_t(
                    first_action_t,
                    policy_frequency,
                ),
                "first_lateral_motion_t": blank_if_none(first_lateral_motion_t),
                "first_lateral_motion_seconds": seconds_from_decision_t(
                    first_lateral_motion_t,
                    policy_frequency,
                ),
                "first_actual_lane_change_t": blank_if_none(first_actual_t),
                "first_actual_lane_change_seconds": seconds_from_post_step_t(
                    first_actual_t,
                    policy_frequency,
                ),
                "action_to_actual_delay": (
                    first_actual_t - first_action_t
                    if first_action_t is not None and first_actual_t is not None
                    else ""
                ),
                "lane_change_action_count": sum(
                    1 for row in ordered if str(row["action"]) in LANE_CHANGE_ACTIONS
                ),
                "actual_lane_change_observed": first_actual_t is not None,
                "first_lane_change_action_direction": first_action,
                "first_actual_lane_change_direction": actual_direction,
                "first_action_direction_matches_actual": (
                    first_action == actual_direction
                    if first_action and actual_direction
                    else ""
                ),
                "collision_flag": collision,
                "terminal_t": ordered[-1]["t"],
                "recording_horizon_seconds": round(
                    (int(ordered[-1]["t"]) + 1) / policy_frequency,
                    6,
                ),
            }
        )
    return rows


def lateral_position(row: Mapping[str, Any]) -> float | None:
    value = row.get("ego_position")
    if value in (None, ""):
        return None
    try:
        position = json.loads(value) if isinstance(value, str) else value
        return float(position[1])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def parse_bool_like(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def seconds_from_decision_t(value: int | None, policy_frequency: int) -> float | str:
    if value is None:
        return ""
    return round(value / policy_frequency, 6)


def seconds_from_post_step_t(value: int | None, policy_frequency: int) -> float | str:
    if value is None:
        return ""
    return round((value + 1) / policy_frequency, 6)


def summarize_actions(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in steps:
        grouped.setdefault(str(row["agent_condition"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for agent, group in sorted(grouped.items()):
        counts = Counter(str(row["action"]) for row in group)
        lane_change_count = sum(counts[action] for action in LANE_CHANGE_ACTIONS)
        actual_lane_change_count = sum(
            1 for row in group if bool(row.get("actual_lane_changed_after_step"))
        )
        rows.append(
            {
                "agent_condition": agent,
                "n_steps": len(group),
                "lane_change_action_count": lane_change_count,
                "lane_change_action_rate": round(lane_change_count / len(group), 6)
                if group
                else "",
                "actual_lane_change_step_count": actual_lane_change_count,
                "action_counts": json_dumps(dict(sorted(counts.items()))),
            }
        )
    return rows


def first_t(rows: Sequence[Mapping[str, Any]] | Any) -> int | None:
    for row in rows:
        return int(row["t"])
    return None


def blank_if_none(value: int | None) -> int | str:
    return "" if value is None else int(value)


def print_summary(result: PipelineResult, output_dir: Path) -> None:
    analysis_dir = output_dir / "analysis"
    print(f"Wrote controlled multi-lane lane-change analysis to {analysis_dir}")
    print(
        "Rows: "
        f"steps={len(result.steps)}, "
        f"modes={len(result.modes)}, "
        f"transitions={len(result.transitions)}, "
        f"outcomes={len(result.episode_outcomes)}, "
        f"gaps={len(result.timing_gaps)}"
    )
    if result.gap_summary:
        print("Gap summary:")
        for row in result.gap_summary:
            print(
                "  "
                f"{row['gap_label']}: "
                f"median={row['median_gap']}, "
                f"IQR=[{row['q1_gap']}, {row['q3_gap']}], "
                f"valid={row['n_valid_pairs']}/{row['n_pairs']}"
            )


def main(argv: list[str] | None = None) -> int:
    configure_headless_runtime()
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    config = config_from_args(args)
    output_dir = Path(args.out)
    train_agents(output_dir, args.agents, args.timesteps, config, args.verbose)
    steps, exposures = evaluate_agents(output_dir, args.agents, args.num_exposures, config)
    result = write_analysis(
        output_dir,
        steps,
        exposures,
        config,
        bootstrap_samples=args.bootstrap_samples,
        figures=not args.no_figures,
    )
    print_summary(result, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
