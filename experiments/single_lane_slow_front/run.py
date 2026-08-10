"""Isolated single-lane slow-front slowdown experiment.

This script intentionally keeps the longitudinal sanity check out of the main
Highway adapter. It trains FD/BAL/SP in a one-lane environment whose action
space contains only SLOWER, IDLE, and FASTER, then evaluates matched slow-front
scenes with the existing transition-timing analysis pipeline.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.constants import AGENTS, SLOWDOWN_ONSET_TARGET, SLOW_ACTIONS
from highway_transition_timing.highway_adapter import (
    action_name_from_env,
    configure_headless_runtime,
    dqn_class_for_variant,
    extract_diagnostics,
    finite_or_blank,
    predict_action_and_scores,
    require_highway_deps,
    step_row_from_diagnostics,
)
from highway_transition_timing.io import write_csv_rows
from highway_transition_timing.model_selection import (
    CHECKPOINT_INTERVAL_STEPS,
    GATE_VARIANTS,
    CheckpointSaver,
    available_checkpoints,
    select_latest_eligible_checkpoint,
    variant_outcome,
)
from highway_transition_timing.pipeline import PipelineResult, run_pipeline
from highway_transition_timing.plotting import (
    write_optional_figures,
    write_training_convergence_figures,
)
from highway_transition_timing.rewards import (
    MAIN_REWARD_WEIGHTS,
    REWARD_STRENGTH_RULES,
    RewardWeights,
    reward_config_table_with_strength_multiplier,
    with_common_slow_down_penalty,
    with_reward_strength_multiplier,
)
from highway_transition_timing.utils import json_dumps

try:
    import gymnasium as gym

    _GymWrapper = gym.Wrapper
except ImportError:
    _GymWrapper = object


COUNTERFACTUAL_VARIANTS = (
    "original",
    "no-front",
    "matched-speed-front",
    "far-front",
)
TRAINING_SCENARIO_PROFILES = ("stratified", "legacy-random")
REWARD_STRENGTH_MULTIPLIERS = (1.0, 2.0, 4.0)
DEFAULT_TARGET_SPEEDS = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
NEAR_MATCHED_SPEED_MIN_ABS_DELTA = 0.25
VEHICLE_LENGTH_METRES = 5.0
OBSERVATION_DISTANCE_METRES = 200.0
VISIBLE_TRAIN_DISTANCE_RANGE = (90.0, 195.0)
DELAYED_VISIBLE_TRAIN_DISTANCE_RANGE = (205.0, 220.0)
SLOW_FRONT_SPEED_RANGE = (10.0, 20.0)
EGO_SPEED_RANGE = (24.0, 31.0)
STRATIFIED_TRAINING_BLOCK = (
    ("train_no_front", "", ""),
    ("train_no_front", "", ""),
    ("train_no_front", "", ""),
    ("train_no_front", "", ""),
    ("train_non_closing_front", "non_closing", "none"),
    ("train_non_closing_front", "non_closing", "none"),
    ("train_non_closing_front", "non_closing", "none"),
    ("train_non_closing_front", "non_closing", "none"),
    ("train_near_closing_front", "gradual", "gentle"),
    ("train_near_closing_front", "gradual", "gentle"),
    ("train_visible_slow_front", "gradual", "gentle"),
    ("train_visible_slow_front", "gradual", "gentle"),
    ("train_visible_slow_front", "easy", "gentle"),
    ("train_visible_slow_front", "easy", "gentle"),
    ("train_visible_slow_front", "medium", "moderate"),
    ("train_visible_slow_front", "medium", "moderate"),
    ("train_visible_slow_front", "hard", "strong"),
    ("train_visible_slow_front", "hard", "strong"),
    ("train_delayed_visible_front", "gradual", "gentle"),
    ("train_delayed_visible_front", "gradual", "gentle"),
)
VISIBLE_DIFFICULTY_CELLS = {
    ("gradual", "gentle"): ((20.001, 30.0), (0.10, 0.25)),
    ("easy", "gentle"): ((12.001, 20.0), (0.20, 0.499)),
    ("medium", "moderate"): ((8.001, 12.0), (0.50, 0.999)),
    ("hard", "strong"): ((5.001, 8.0), (1.00, 1.999)),
}


@dataclass(frozen=True)
class ExperimentConfig:
    duration: int = 120
    evaluation_duration: int = 120
    seed: int = 0
    policy_frequency: int = 5
    simulation_frequency: int = 15
    observation_normalize: bool = True
    observation_absolute: bool = False
    slow_down_penalty: float = 0.0
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
    training_scenario_profile: str = "stratified"
    target_speeds: tuple[float, ...] = DEFAULT_TARGET_SPEEDS
    no_front_train_fraction: float = 0.2
    near_matched_speed_train_fraction: float = 0.2
    near_matched_speed_delta: float = 2.0
    reward_strength_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.duration <= 0 or self.evaluation_duration <= 0:
            raise ValueError("duration values must be positive seconds")
        if self.policy_frequency <= 0 or self.simulation_frequency <= 0:
            raise ValueError("frequency values must be positive")
        if self.training_scenario_profile not in TRAINING_SCENARIO_PROFILES:
            raise ValueError(
                "training_scenario_profile must be one of "
                f"{TRAINING_SCENARIO_PROFILES}"
            )
        if len(self.target_speeds) < 2:
            raise ValueError("target_speeds must contain at least two speeds")
        if any(speed < 0.0 for speed in self.target_speeds):
            raise ValueError("target_speeds must be non-negative")
        if any(
            current >= following
            for current, following in zip(
                self.target_speeds,
                self.target_speeds[1:],
                strict=False,
            )
        ):
            raise ValueError("target_speeds must be strictly increasing")
        if not 0.0 <= self.no_front_train_fraction <= 1.0:
            raise ValueError("no_front_train_fraction must be in [0, 1]")
        if not 0.0 <= self.near_matched_speed_train_fraction <= 1.0:
            raise ValueError(
                "near_matched_speed_train_fraction must be in [0, 1]"
            )
        if (
            self.no_front_train_fraction
            + self.near_matched_speed_train_fraction
            > 1.0
        ):
            raise ValueError(
                "no_front_train_fraction + near_matched_speed_train_fraction "
                "must be <= 1"
            )
        if self.near_matched_speed_delta < 0.25:
            raise ValueError("near_matched_speed_delta must be >= 0.25")
        if self.reward_strength_multiplier not in REWARD_STRENGTH_MULTIPLIERS:
            raise ValueError(
                "reward_strength_multiplier must be one of "
                f"{REWARD_STRENGTH_MULTIPLIERS}"
            )

    @property
    def policy_step_seconds(self) -> float:
        return 1.0 / self.policy_frequency

    @property
    def training_max_policy_steps(self) -> int:
        return int(round(self.duration * self.policy_frequency))

    @property
    def evaluation_max_policy_steps(self) -> int:
        return int(round(self.evaluation_duration * self.policy_frequency))


@dataclass(frozen=True)
class SingleLaneSpec:
    exposure_id: str
    exposure_seed: int
    ego_speed: float
    front_distance: float
    front_speed: float
    ego_longitudinal: float = 100.0
    exposure_t: int = 0
    scenario_type: str = "slow_front"
    include_front_vehicle: bool = True
    ttc_bin: str = ""
    required_deceleration_bin: str = ""
    visible_at_t0: bool = True

    @property
    def net_distance(self) -> float:
        if not self.include_front_vehicle:
            return float("inf")
        return max(self.front_distance - VEHICLE_LENGTH_METRES, 0.0)

    @property
    def closing_speed(self) -> float:
        if not self.include_front_vehicle:
            return 0.0
        return self.ego_speed - self.front_speed

    @property
    def ttc_seconds(self) -> float:
        if self.closing_speed <= 0.0 or self.net_distance <= 0.0:
            return float("inf")
        return self.net_distance / self.closing_speed

    @property
    def required_deceleration(self) -> float:
        if self.closing_speed <= 0.0 or self.net_distance <= 0.0:
            return 0.0
        return self.closing_speed**2 / (2.0 * self.net_distance)


class TrainingScenarioGenerator:
    """Generate one deterministic continuous RNG stream for a training run."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.reset_index = 0
        self.stratified_block: list[tuple[str, str, str]] = []

    def next_spec(self) -> SingleLaneSpec:
        reset_index = self.reset_index
        if self.config.training_scenario_profile == "legacy-random":
            spec = make_legacy_random_training_spec(
                reset_index,
                self.config,
                self.rng,
            )
        else:
            block_position = reset_index % len(STRATIFIED_TRAINING_BLOCK)
            if block_position == 0:
                self.stratified_block = list(STRATIFIED_TRAINING_BLOCK)
                self.rng.shuffle(self.stratified_block)
            spec = make_stratified_training_spec(
                reset_index,
                self.config,
                self.rng,
                self.stratified_block[block_position],
            )
        self.reset_index += 1
        return spec


class SingleLaneTrainingResetWrapper(_GymWrapper):
    """Replace resets with slow-front, no-front, or near-matched scenes."""

    def __init__(self, env, config: ExperimentConfig):
        if _GymWrapper is object:
            self.env = env
        else:
            super().__init__(env)
        self.config = config
        self.reset_count = 0
        self.episode_step_count = 0
        self.training_scenario_generator = TrainingScenarioGenerator(config)
        self.last_training_exposure: SingleLaneSpec | None = None
        self.training_variant_counts: Counter[str] = Counter()
        self.training_spec_rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_step_count = 0
        reset_index = self.reset_count
        spec = self.training_scenario_generator.next_spec()
        if self.training_scenario_generator.reset_index != reset_index + 1:
            raise RuntimeError("training scenario generator is out of sync")
        self.reset_count += 1
        self.last_training_exposure = spec
        self.training_variant_counts[spec.scenario_type] += 1
        self.training_spec_rows.append(training_spec_row(reset_index, spec, self.config))
        obs = apply_single_lane_scene(
            self.env,
            spec,
            include_front_vehicle=spec.include_front_vehicle,
        )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step_count += 1
        reached_horizon = (
            self.episode_step_count >= self.config.training_max_policy_steps
        )
        if reached_horizon and not terminated:
            truncated = True
            info = dict(info)
            info["training_horizon_reached"] = True
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()


class LongitudinalRewardWrapper(_GymWrapper):
    """Small local reward wrapper for the isolated experiment."""

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
        components = local_reward_components(diagnostics)
        action_name = action_name_from_env(self.env, int(action))
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
        help="Training episode duration in seconds.",
    )
    parser.add_argument(
        "--evaluation-duration",
        type=int,
        default=120,
        help="Evaluation rollout duration in seconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--collision-risk-penalty", type=float, default=3.0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument(
        "--reward-strength-multiplier",
        type=float,
        choices=list(REWARD_STRENGTH_MULTIPLIERS),
        default=1.0,
        help=(
            "Approved validation levels only: FD scales front-distance, SP "
            "scales speed, and BAL scales its complete effective reward."
        ),
    )
    parser.add_argument("--absolute-observation", action="store_true")
    parser.add_argument("--no-normalize-observation", action="store_true")
    parser.add_argument(
        "--training-scenario-profile",
        choices=list(TRAINING_SCENARIO_PROFILES),
        default="stratified",
        help=(
            "Use the balanced TTC/deceleration block by default; "
            "legacy-random reproduces the previous probabilistic mixture."
        ),
    )
    parser.add_argument(
        "--target-speeds",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_SPEEDS),
        help="Discrete longitudinal target speeds in m/s.",
    )
    parser.add_argument(
        "--no-front-train-fraction",
        type=float,
        default=0.2,
        help=(
            "Legacy-random fraction of training resets with no front vehicle. "
            "Set this and --near-matched-speed-train-fraction to 0.0 to "
            "reproduce the original always-slow-front baseline."
        ),
    )
    parser.add_argument(
        "--near-matched-speed-train-fraction",
        type=float,
        default=0.2,
        help=(
            "Legacy-random fraction with a front vehicle traveling within "
            "--near-matched-speed-delta of ego speed."
        ),
    )
    parser.add_argument(
        "--near-matched-speed-delta",
        type=float,
        default=2.0,
        help=(
            "Maximum absolute front-minus-ego speed difference in near-matched "
            "training resets; an absolute difference below "
            f"{NEAR_MATCHED_SPEED_MIN_ABS_DELTA} m/s is excluded."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_INTERVAL_STEPS,
        help="Policy steps between development checkpoints.",
    )
    parser.add_argument(
        "--no-model-selection",
        action="store_true",
        help=(
            "Skip the eligibility gate and keep the final checkpoint. "
            "For code smokes only; maintained runs must use selection."
        ),
    )
    return parser


def build_counterfactual_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run front-vehicle counterfactuals for a trained single-lane run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-exposures", type=int, default=36)
    parser.add_argument("--agents", nargs="+", default=list(AGENTS))
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds.")
    parser.add_argument(
        "--evaluation-duration",
        type=int,
        default=120,
        help="Rollout duration in seconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision-risk-penalty", type=float, default=3.0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument(
        "--reward-strength-multiplier",
        type=float,
        choices=list(REWARD_STRENGTH_MULTIPLIERS),
        default=1.0,
    )
    parser.add_argument("--absolute-observation", action="store_true")
    parser.add_argument("--no-normalize-observation", action="store_true")
    parser.add_argument(
        "--target-speeds",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_SPEEDS),
    )
    parser.add_argument(
        "--counterfactual-variants",
        nargs="+",
        default=list(COUNTERFACTUAL_VARIANTS),
        choices=list(COUNTERFACTUAL_VARIANTS),
    )
    return parser


def build_rollout_counterfactual_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-episode front-vehicle counterfactuals for a trained "
            "single-lane run."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-exposures", type=int, default=36)
    parser.add_argument("--agents", nargs="+", default=list(AGENTS))
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds.")
    parser.add_argument(
        "--evaluation-duration",
        type=int,
        default=120,
        help="Rollout duration in seconds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--collision-risk-penalty", type=float, default=3.0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument(
        "--reward-strength-multiplier",
        type=float,
        choices=list(REWARD_STRENGTH_MULTIPLIERS),
        default=1.0,
    )
    parser.add_argument("--absolute-observation", action="store_true")
    parser.add_argument("--no-normalize-observation", action="store_true")
    parser.add_argument(
        "--target-speeds",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_SPEEDS),
    )
    parser.add_argument(
        "--counterfactual-variants",
        nargs="+",
        default=list(COUNTERFACTUAL_VARIANTS),
        choices=list(COUNTERFACTUAL_VARIANTS),
    )
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the stratified reset distribution and immediate-SLOWER "
            "feasibility before training."
        )
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-resets", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--evaluation-duration", type=int, default=120)
    parser.add_argument(
        "--training-scenario-profile",
        choices=list(TRAINING_SCENARIO_PROFILES),
        default="stratified",
    )
    parser.add_argument(
        "--target-speeds",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_SPEEDS),
    )
    parser.add_argument("--near-matched-speed-delta", type=float, default=2.0)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        duration=args.duration,
        evaluation_duration=args.evaluation_duration,
        seed=args.seed,
        observation_normalize=not args.no_normalize_observation,
        observation_absolute=args.absolute_observation,
        collision_risk_penalty=args.collision_risk_penalty,
        collision_penalty=args.collision_penalty,
        learning_rate=getattr(args, "learning_rate", 5e-4),
        training_scenario_profile=getattr(
            args,
            "training_scenario_profile",
            "stratified",
        ),
        target_speeds=tuple(
            float(value)
            for value in getattr(args, "target_speeds", DEFAULT_TARGET_SPEEDS)
        ),
        no_front_train_fraction=getattr(args, "no_front_train_fraction", 0.2),
        near_matched_speed_train_fraction=getattr(
            args,
            "near_matched_speed_train_fraction",
            0.2,
        ),
        near_matched_speed_delta=getattr(
            args,
            "near_matched_speed_delta",
            2.0,
        ),
        reward_strength_multiplier=getattr(
            args,
            "reward_strength_multiplier",
            1.0,
        ),
    )


def effective_reward_weights(
    agent_condition: str,
    config: ExperimentConfig,
) -> RewardWeights:
    """Return common overrides plus the approved strength intervention."""

    weights = with_common_slow_down_penalty(
        MAIN_REWARD_WEIGHTS[agent_condition],
        config.slow_down_penalty,
        config.collision_risk_penalty,
        config.collision_penalty,
    )
    return with_reward_strength_multiplier(
        agent_condition,
        weights,
        config.reward_strength_multiplier,
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
                "vehicles_count": 2,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": config.observation_absolute,
                "normalize": config.observation_normalize,
                "order": "sorted",
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": list(config.target_speeds),
            },
            "lanes_count": 1,
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
        env = SingleLaneTrainingResetWrapper(env, config)
    weights = effective_reward_weights(agent_condition, config)
    return LongitudinalRewardWrapper(env, agent_condition, weights)


def classify_ttc(ttc_seconds: float, closing_speed: float) -> str:
    if closing_speed <= 0.0:
        return "non_closing"
    if ttc_seconds > 20.0:
        return "gradual"
    if ttc_seconds > 12.0:
        return "easy"
    if ttc_seconds > 8.0:
        return "medium"
    if ttc_seconds > 5.0:
        return "hard"
    return "critical"


def classify_required_deceleration(required_deceleration: float) -> str:
    if required_deceleration <= 0.0:
        return "none"
    if required_deceleration < 0.5:
        return "gentle"
    if required_deceleration < 1.0:
        return "moderate"
    if required_deceleration < 2.0:
        return "strong"
    if required_deceleration < 3.0:
        return "very_strong"
    return "extreme"


def build_single_lane_spec(
    *,
    exposure_id: str,
    exposure_seed: int,
    ego_speed: float,
    front_distance: float,
    front_speed: float,
    scenario_type: str,
    include_front_vehicle: bool,
    ego_longitudinal: float = 100.0,
    exposure_t: int = 0,
) -> SingleLaneSpec:
    rounded_ego_speed = round(float(ego_speed), 3)
    rounded_front_distance = round(float(front_distance), 3)
    rounded_front_speed = round(float(front_speed), 3)
    net_distance = (
        max(rounded_front_distance - VEHICLE_LENGTH_METRES, 0.0)
        if include_front_vehicle
        else float("inf")
    )
    closing_speed = (
        rounded_ego_speed - rounded_front_speed if include_front_vehicle else 0.0
    )
    ttc_seconds = (
        net_distance / closing_speed
        if closing_speed > 0.0 and net_distance > 0.0
        else float("inf")
    )
    required_deceleration = (
        closing_speed**2 / (2.0 * net_distance)
        if closing_speed > 0.0 and net_distance > 0.0
        else 0.0
    )
    return SingleLaneSpec(
        exposure_id=exposure_id,
        exposure_seed=exposure_seed,
        ego_speed=rounded_ego_speed,
        front_distance=rounded_front_distance,
        front_speed=rounded_front_speed,
        ego_longitudinal=ego_longitudinal,
        exposure_t=exposure_t,
        scenario_type=scenario_type,
        include_front_vehicle=include_front_vehicle,
        ttc_bin=(
            classify_ttc(ttc_seconds, closing_speed)
            if include_front_vehicle
            else "not_applicable"
        ),
        required_deceleration_bin=(
            classify_required_deceleration(required_deceleration)
            if include_front_vehicle
            else "not_applicable"
        ),
        visible_at_t0=(
            include_front_vehicle
            and rounded_front_distance < OBSERVATION_DISTANCE_METRES
        ),
    )


def make_eval_specs(num_exposures: int, config: ExperimentConfig) -> list[SingleLaneSpec]:
    front_distances = [90.0, 120.0, 150.0, 180.0]
    front_speeds = [10.0, 14.0, 18.0]
    ego_speeds = [26.0, 28.0, 30.0]
    specs: list[SingleLaneSpec] = []
    for idx in range(num_exposures):
        specs.append(
            build_single_lane_spec(
                exposure_id=f"S{idx:04d}",
                exposure_seed=config.seed + idx,
                ego_speed=ego_speeds[
                    (idx // (len(front_distances) * len(front_speeds)))
                    % len(ego_speeds)
                ],
                front_distance=front_distances[idx % len(front_distances)],
                front_speed=front_speeds[(idx // len(front_distances)) % len(front_speeds)],
                scenario_type="slow_front",
                include_front_vehicle=True,
            )
        )
    return specs


def make_training_spec(reset_index: int, config: ExperimentConfig) -> SingleLaneSpec:
    """Replay the run RNG stream and return one indexed training specification."""

    if reset_index < 0:
        raise ValueError("reset_index must be non-negative")
    return make_training_specs(reset_index + 1, config)[-1]


def make_training_specs(
    num_resets: int,
    config: ExperimentConfig,
) -> list[SingleLaneSpec]:
    """Generate a deterministic prefix from one RNG initialized by run seed."""

    if num_resets < 0:
        raise ValueError("num_resets must be non-negative")
    generator = TrainingScenarioGenerator(config)
    return [generator.next_spec() for _ in range(num_resets)]


def make_stratified_training_spec(
    reset_index: int,
    config: ExperimentConfig,
    rng: random.Random,
    slot: tuple[str, str, str],
) -> SingleLaneSpec:
    scenario_type, expected_ttc_bin, expected_deceleration_bin = slot

    if scenario_type == "train_no_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        return build_single_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=config.seed,
            ego_speed=ego_speed,
            front_distance=0.0,
            front_speed=ego_speed,
            scenario_type=scenario_type,
            include_front_vehicle=False,
        )

    if scenario_type == "train_non_closing_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed + rng.uniform(
            NEAR_MATCHED_SPEED_MIN_ABS_DELTA,
            config.near_matched_speed_delta,
        )
        return build_single_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=config.seed,
            ego_speed=ego_speed,
            front_distance=rng.uniform(*VISIBLE_TRAIN_DISTANCE_RANGE),
            front_speed=front_speed,
            scenario_type=scenario_type,
            include_front_vehicle=True,
        )

    if scenario_type == "train_near_closing_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed - rng.uniform(
            NEAR_MATCHED_SPEED_MIN_ABS_DELTA,
            config.near_matched_speed_delta,
        )
        return build_single_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=config.seed,
            ego_speed=ego_speed,
            front_distance=rng.uniform(*VISIBLE_TRAIN_DISTANCE_RANGE),
            front_speed=front_speed,
            scenario_type=scenario_type,
            include_front_vehicle=True,
        )

    if scenario_type == "train_delayed_visible_front":
        return sample_delayed_visible_training_spec(reset_index, config.seed, rng)

    return sample_visible_slow_front_training_spec(
        reset_index=reset_index,
        run_seed=config.seed,
        rng=rng,
        expected_ttc_bin=expected_ttc_bin,
        expected_deceleration_bin=expected_deceleration_bin,
    )


def sample_visible_slow_front_training_spec(
    *,
    reset_index: int,
    run_seed: int,
    rng: random.Random,
    expected_ttc_bin: str,
    expected_deceleration_bin: str,
) -> SingleLaneSpec:
    ttc_range, deceleration_range = VISIBLE_DIFFICULTY_CELLS[
        (expected_ttc_bin, expected_deceleration_bin)
    ]
    for _attempt in range(10_000):
        ttc_seconds = rng.uniform(*ttc_range)
        required_deceleration = rng.uniform(*deceleration_range)
        closing_speed = 2.0 * required_deceleration * ttc_seconds
        net_distance = closing_speed * ttc_seconds
        front_distance = net_distance + VEHICLE_LENGTH_METRES
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed - closing_speed
        if not (
            VISIBLE_TRAIN_DISTANCE_RANGE[0]
            <= front_distance
            <= VISIBLE_TRAIN_DISTANCE_RANGE[1]
        ):
            continue
        if not SLOW_FRONT_SPEED_RANGE[0] <= front_speed <= SLOW_FRONT_SPEED_RANGE[1]:
            continue
        spec = build_single_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=run_seed,
            ego_speed=ego_speed,
            front_distance=front_distance,
            front_speed=front_speed,
            scenario_type="train_visible_slow_front",
            include_front_vehicle=True,
        )
        if (
            spec.ttc_bin == expected_ttc_bin
            and spec.required_deceleration_bin == expected_deceleration_bin
        ):
            return spec
    raise RuntimeError(
        "Unable to sample feasible visible slow-front scene for "
        f"{expected_ttc_bin}/{expected_deceleration_bin}"
    )


def sample_delayed_visible_training_spec(
    reset_index: int,
    run_seed: int,
    rng: random.Random,
) -> SingleLaneSpec:
    for _attempt in range(10_000):
        front_distance = rng.uniform(*DELAYED_VISIBLE_TRAIN_DISTANCE_RANGE)
        net_distance = front_distance - VEHICLE_LENGTH_METRES
        ttc_seconds = rng.uniform(20.001, 35.0)
        closing_speed = net_distance / ttc_seconds
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed - closing_speed
        if not SLOW_FRONT_SPEED_RANGE[0] <= front_speed <= SLOW_FRONT_SPEED_RANGE[1]:
            continue
        spec = build_single_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=run_seed,
            ego_speed=ego_speed,
            front_distance=front_distance,
            front_speed=front_speed,
            scenario_type="train_delayed_visible_front",
            include_front_vehicle=True,
        )
        if spec.ttc_bin == "gradual" and spec.required_deceleration_bin == "gentle":
            return spec
    raise RuntimeError("Unable to sample feasible delayed-visible slow-front scene")


def make_legacy_random_training_spec(
    reset_index: int,
    config: ExperimentConfig,
    rng: random.Random,
) -> SingleLaneSpec:
    variant_draw = rng.random()
    ego_speed = rng.uniform(*EGO_SPEED_RANGE)
    front_distance = rng.uniform(90.0, 220.0)
    slow_front_speed = rng.uniform(*SLOW_FRONT_SPEED_RANGE)

    if variant_draw < config.no_front_train_fraction:
        scenario_type = "train_no_front"
        include_front_vehicle = False
        front_speed = slow_front_speed
    elif variant_draw < (
        config.no_front_train_fraction
        + config.near_matched_speed_train_fraction
    ):
        scenario_type = "train_near_matched_front"
        include_front_vehicle = True
        speed_delta_sign = -1.0 if rng.random() < 0.5 else 1.0
        speed_delta = speed_delta_sign * rng.uniform(
            NEAR_MATCHED_SPEED_MIN_ABS_DELTA,
            config.near_matched_speed_delta,
        )
        front_speed = round(ego_speed + speed_delta, 3)
    else:
        scenario_type = "train_slow_front"
        include_front_vehicle = True
        front_speed = slow_front_speed

    return build_single_lane_spec(
        exposure_id=f"train_reset_{reset_index:06d}",
        exposure_seed=config.seed,
        ego_speed=ego_speed,
        front_distance=front_distance,
        front_speed=front_speed,
        scenario_type=scenario_type,
        include_front_vehicle=include_front_vehicle,
    )


def training_spec_row(
    reset_index: int,
    spec: SingleLaneSpec,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "reset_index": reset_index,
        "exposure_id": spec.exposure_id,
        "run_seed": config.seed,
        "training_rng_mode": "continuous_per_run",
        "training_scenario_profile": config.training_scenario_profile,
        "scenario_type": spec.scenario_type,
        "include_front_vehicle": spec.include_front_vehicle,
        "ego_speed": spec.ego_speed,
        "front_center_distance": (
            spec.front_distance if spec.include_front_vehicle else ""
        ),
        "front_net_distance": finite_or_blank(spec.net_distance),
        "front_speed": spec.front_speed if spec.include_front_vehicle else "",
        "closing_speed": round(spec.closing_speed, 6),
        "ttc_seconds": finite_or_blank(spec.ttc_seconds),
        "required_deceleration_mps2": round(spec.required_deceleration, 6),
        "ttc_bin": spec.ttc_bin,
        "required_deceleration_bin": spec.required_deceleration_bin,
        "difficulty_bin": (
            f"{spec.ttc_bin}__{spec.required_deceleration_bin}"
        ),
        "visible_at_t0": spec.visible_at_t0,
        "policy_frequency_hz": config.policy_frequency,
        "policy_step_seconds": round(config.policy_step_seconds, 6),
        "target_speeds_mps": json_dumps(list(config.target_speeds)),
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


def counterfactual_settings(
    spec: SingleLaneSpec,
    variant: str,
) -> dict[str, float | bool | None]:
    normalized = variant.lower()
    if normalized == "original":
        return {
            "include_front_vehicle": True,
            "front_distance": spec.front_distance,
            "front_speed": spec.front_speed,
        }
    if normalized == "no-front":
        return {
            "include_front_vehicle": False,
            "front_distance": None,
            "front_speed": None,
        }
    if normalized == "matched-speed-front":
        return {
            "include_front_vehicle": True,
            "front_distance": spec.front_distance,
            "front_speed": spec.ego_speed,
        }
    if normalized == "far-front":
        return {
            "include_front_vehicle": True,
            "front_distance": max(260.0, spec.front_distance),
            "front_speed": spec.front_speed,
        }
    raise ValueError(f"Unsupported counterfactual variant: {variant}")


def apply_single_lane_scene(
    env,
    spec: SingleLaneSpec,
    include_front_vehicle: bool | None = None,
    front_distance: float | None = None,
    front_speed: float | None = None,
    reset_time: bool = True,
):
    from highway_env.vehicle.behavior import IDMVehicle

    unwrapped = env.unwrapped
    road = unwrapped.road
    lane_index = ("0", "1", 0)
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
    include_lead = (
        spec.include_front_vehicle
        if include_front_vehicle is None
        else include_front_vehicle
    )
    if include_lead:
        lead_distance = spec.front_distance if front_distance is None else front_distance
        lead_speed = spec.front_speed if front_speed is None else front_speed
        lead = IDMVehicle(
            road,
            lane.position(spec.ego_longitudinal + float(lead_distance), 0),
            lane.heading_at(spec.ego_longitudinal + float(lead_distance)),
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


def local_reward_components(diagnostics: dict[str, Any]) -> dict[str, float]:
    return {
        "speed_score": float(diagnostics["speed_score"]),
        "front_distance_score": float(diagnostics["front_distance_score"]),
        "collision_penalty": -1.0 if diagnostics["collision_flag"] else 0.0,
        "collision_risk_penalty": -float(diagnostics["collision_risk_score"]),
        "lane_change_penalty": 0.0,
        "slow_down_penalty": 0.0,
        "right_lane_score": 0.0,
    }


def weighted_reward(components: dict[str, float], weights: RewardWeights) -> float:
    return (
        weights.speed_score * components["speed_score"]
        + weights.front_distance_score * components["front_distance_score"]
        + weights.collision_penalty * components["collision_penalty"]
        + weights.collision_risk_penalty * components["collision_risk_penalty"]
        - weights.slow_down_penalty * abs(components["slow_down_penalty"])
    )


def action_names_from_env(env) -> dict[int, str]:
    action_type = getattr(env.unwrapped, "action_type", None)
    actions = getattr(action_type, "actions", None)
    if isinstance(actions, Mapping):
        return {int(key): str(value) for key, value in actions.items()}
    return {0: "SLOWER", 1: "IDLE", 2: "FASTER"}


def longitudinal_q_diagnostics(action_scores: Sequence[float], env) -> dict[str, Any]:
    values = [float(value) for value in action_scores]
    names = action_names_from_env(env)
    if not values:
        return {
            "q_argmax_action": "",
            "q_top1": "",
            "q_top2": "",
            "q_top_margin": "",
            "slower_q_advantage": "",
        }

    argmax_index = max(range(len(values)), key=values.__getitem__)
    sorted_values = sorted(values, reverse=True)
    slower_index = next(
        (
            action_index
            for action_index, action_name in names.items()
            if action_name == "SLOWER" and action_index < len(values)
        ),
        None,
    )
    slower_q_advantage = (
        values[slower_index]
        - max(value for index, value in enumerate(values) if index != slower_index)
        if slower_index is not None and len(values) > 1
        else ""
    )
    return {
        "q_argmax_action": names.get(argmax_index, str(argmax_index)),
        "q_top1": round(sorted_values[0], 6),
        "q_top2": round(sorted_values[1], 6) if len(sorted_values) > 1 else "",
        "q_top_margin": round(sorted_values[0] - sorted_values[1], 6)
        if len(sorted_values) > 1
        else "",
        "slower_q_advantage": round(float(slower_q_advantage), 6)
        if slower_q_advantage != ""
        else "",
    }


def train_agents(
    output_dir: Path,
    agents: Sequence[str],
    total_timesteps: int,
    config: ExperimentConfig,
    verbose: int,
    checkpoint_every: int = CHECKPOINT_INTERVAL_STEPS,
) -> None:
    require_highway_deps(include_training=True)
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    dqn_class = dqn_class_for_variant(config.dqn_variant)
    models_dir = output_dir / "models"
    checkpoint_dir = models_dir / "checkpoints"
    logs_dir = output_dir / "training_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    block_family_counts = Counter(
        scenario_type
        for scenario_type, _ttc_bin, _deceleration_bin in STRATIFIED_TRAINING_BLOCK
    )

    for agent in agents:
        effective_weights = effective_reward_weights(agent, config)
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
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False,
            callback=CheckpointSaver(
                agent=agent,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                verbose=verbose,
            ),
        )
        # The final checkpoint is the provisional policy. Model selection
        # overwrites this archive with the latest eligible checkpoint.
        model_path = models_dir / f"{agent}_main.zip"
        model.save(model_path)
        training_variant_counts = dict(
            sorted(getattr(training_env, "training_variant_counts", Counter()).items())
        )
        agent_scenario_rows = list(
            getattr(training_env, "training_spec_rows", [])
        )
        scenario_rows.extend(
            {"agent_condition": agent, **row}
            for row in agent_scenario_rows
        )
        env.close()
        rows.append(
            {
                "agent_condition": agent,
                "policy_id": f"{agent}_main",
                "model_path": str(model_path),
                "monitor_path": str(agent_logs_dir / "monitor.csv"),
                "progress_path": str(agent_logs_dir / "progress.csv"),
                "total_timesteps": total_timesteps,
                "checkpoint_every": checkpoint_every,
                "seed": config.seed,
                **{
                    f"base_reward_{key}": value
                    for key, value in asdict(MAIN_REWARD_WEIGHTS[agent]).items()
                },
                **{
                    f"reward_{key}": value
                    for key, value in asdict(effective_weights).items()
                },
                "reward_strength_multiplier": config.reward_strength_multiplier,
                "reward_strength_rule": REWARD_STRENGTH_RULES[agent],
                "configured_common_slow_down_penalty": config.slow_down_penalty,
                "configured_common_collision_risk_penalty": (
                    config.collision_risk_penalty
                ),
                "configured_common_collision_penalty": config.collision_penalty
                if config.collision_penalty is not None
                else "",
                "experiment": "single_lane_slow_front",
                "action_space": "SLOWER,IDLE,FASTER",
                "target_speeds_mps": json_dumps(list(config.target_speeds)),
                "training_scenario_profile": config.training_scenario_profile,
                "training_rng_mode": "continuous_per_run",
                "training_scenario_block_size": len(STRATIFIED_TRAINING_BLOCK),
                "training_scenario_block_family_counts": json_dumps(
                    dict(sorted(block_family_counts.items()))
                ),
                "training_no_front_fraction": config.no_front_train_fraction,
                "training_near_matched_speed_fraction": (
                    config.near_matched_speed_train_fraction
                ),
                "training_near_matched_speed_min_abs_delta": (
                    NEAR_MATCHED_SPEED_MIN_ABS_DELTA
                ),
                "training_near_matched_speed_max_abs_delta": (
                    config.near_matched_speed_delta
                ),
                "training_scene_counts": json_dumps(training_variant_counts),
                "training_reset_count": len(agent_scenario_rows),
                "training_duration_seconds": config.duration,
                "training_max_policy_steps": config.training_max_policy_steps,
                "evaluation_duration_seconds": config.evaluation_duration,
                "policy_frequency_hz": config.policy_frequency,
                "policy_step_seconds": round(config.policy_step_seconds, 6),
                "evaluation_max_policy_steps": config.evaluation_max_policy_steps,
                "observation_normalize": config.observation_normalize,
                "observation_absolute": config.observation_absolute,
                "dqn_variant": config.dqn_variant,
            }
        )
    write_csv_rows(output_dir / "training_runs.csv", rows)
    write_csv_rows(output_dir / "training_scenarios.csv", scenario_rows)


def development_gate_outcomes(
    model: Any,
    agent: str,
    num_exposures: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Evaluate one policy on the development grid under every gate variant."""

    specs = make_eval_specs(num_exposures, config)
    models = {agent: model}
    outcomes: dict[str, Any] = {}
    for variant in GATE_VARIANTS:
        step_rows, exposure_rows = rollout_counterfactual_variant(
            models,
            [agent],
            specs,
            config,
            variant,
        )
        result = run_pipeline(
            step_rows,
            exposure_rows,
            AnalysisConfig(bootstrap_samples=0),
            SLOWDOWN_ONSET_TARGET,
        )
        outcomes[variant] = variant_outcome(
            result.episode_outcomes,
            agent,
            config.policy_step_seconds,
        )
    return outcomes


def select_models(
    output_dir: Path,
    agents: Sequence[str],
    num_exposures: int,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Replace each `{agent}_main.zip` with its latest eligible checkpoint."""

    require_highway_deps(include_training=True)
    import shutil

    dqn_class = dqn_class_for_variant(config.dqn_variant)
    models_dir = output_dir / "models"
    checkpoint_dir = models_dir / "checkpoints"
    records: list[dict[str, Any]] = []
    unselected: list[str] = []

    for agent in agents:
        checkpoints = available_checkpoints(checkpoint_dir, agent)

        def evaluate(path: Path, agent: str = agent) -> dict[str, Any]:
            print(f"Gating {agent} checkpoint {path.name}", flush=True)
            return development_gate_outcomes(
                dqn_class.load(path),
                agent,
                num_exposures,
                config,
            )

        selected, agent_records = select_latest_eligible_checkpoint(
            agent,
            checkpoints,
            evaluate,
        )
        records.extend(agent_records)
        if selected is None:
            unselected.append(agent)
            continue
        step, path = selected
        shutil.copyfile(path, models_dir / f"{agent}_main.zip")
        print(f"Selected {agent} checkpoint at step {step}", flush=True)

    write_csv_rows(output_dir / "model_selection.csv", records)
    if unselected:
        raise RuntimeError(
            "No checkpoint passed the eligibility gate for: "
            f"{sorted(unselected)}. See model_selection.csv."
        )
    return records


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
            obs = apply_single_lane_scene(env, spec)
            if not exposure_recorded:
                exposure_rows.append(exposure_row_from_spec(env, spec))
                exposure_recorded = True

            model = models[agent]
            policy_id = f"{agent}_main"
            episode_id = f"{agent}_{spec.exposure_id}_r0"
            done = False
            t = 0
            while not done and t < config.evaluation_max_policy_steps:
                action, q_values = predict_action_and_scores(model, obs, deterministic=True)
                action_name = action_name_from_env(env, int(action))
                pre_step_diagnostics = extract_diagnostics(env)
                next_obs, reward, terminated, truncated, _info = env.step(int(action))
                post_step_diagnostics = extract_diagnostics(env)
                env_done = bool(terminated or truncated)
                reached_horizon = t + 1 >= config.evaluation_max_policy_steps
                done = env_done or reached_horizon
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
                    termination_reason=(
                        "collision" if collision else ("duration" if done else "")
                    ),
                    lane_change_count=0,
                    exposure_t=spec.exposure_t,
                )
                row.update(policy_time_fields(t, spec.exposure_t, config))
                step_rows.append(row)
                obs = next_obs
                t += 1
            env.close()

    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(eval_dir / "steps.csv", step_rows)
    write_csv_rows(eval_dir / "exposures.csv", exposure_rows)
    return step_rows, exposure_rows


def evaluate_counterfactuals(
    run_dir: Path,
    output_dir: Path,
    agents: Sequence[str],
    num_exposures: int,
    config: ExperimentConfig,
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    require_highway_deps(include_training=True)
    dqn_class = dqn_class_for_variant(config.dqn_variant)
    models = {
        agent: dqn_class.load(run_dir / "models" / f"{agent}_main.zip")
        for agent in agents
    }
    specs = make_eval_specs(num_exposures, config)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        for variant in variants:
            settings = counterfactual_settings(spec, variant)
            for agent in agents:
                env = make_env(agent, config, training=False)
                obs, _info = env.reset(seed=spec.exposure_seed)
                obs = apply_single_lane_scene(
                    env,
                    spec,
                    include_front_vehicle=bool(settings["include_front_vehicle"]),
                    front_distance=settings["front_distance"],
                    front_speed=settings["front_speed"],
                )
                diagnostics = extract_diagnostics(env)
                action, q_values = predict_action_and_scores(
                    models[agent],
                    obs,
                    deterministic=True,
                )
                action_name = action_name_from_env(env, int(action))
                front_speed = diagnostics["front_vehicle_speed"]
                relative_closing = (
                    max(0.0, float(diagnostics["ego_speed"]) - float(front_speed))
                    if front_speed is not None
                    else 0.0
                )
                rows.append(
                    {
                        "exposure_id": spec.exposure_id,
                        "exposure_seed": spec.exposure_seed,
                        "agent_condition": agent,
                        "policy_id": f"{agent}_main",
                        "counterfactual_variant": variant,
                        "include_front_vehicle": settings["include_front_vehicle"],
                        "configured_front_distance": finite_or_blank(
                            settings["front_distance"]
                        ),
                        "configured_front_speed": finite_or_blank(
                            settings["front_speed"]
                        ),
                        "ego_speed": round(float(diagnostics["ego_speed"]), 6),
                        "nearest_front_distance": finite_or_blank(
                            diagnostics["nearest_front_distance"]
                        ),
                        "front_vehicle_speed": finite_or_blank(front_speed),
                        "relative_closing_speed": round(relative_closing, 6),
                        "collision_risk_score": round(
                            float(diagnostics["collision_risk_score"]),
                            6,
                        ),
                        "closest_k_distances": [
                            round(float(value), 6)
                            for value in diagnostics["closest_k_distances"]
                        ],
                        "action": action_name,
                        "is_slow_action": action_name in SLOW_ACTIONS,
                        "q_values_or_action_scores": [
                            round(float(value), 6) for value in q_values
                        ],
                        **longitudinal_q_diagnostics(q_values, env),
                    }
                )
                env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "counterfactual_initial_actions.csv", rows)
    summary_rows = summarize_counterfactual_rows(rows)
    write_csv_rows(output_dir / "counterfactual_action_summary.csv", summary_rows)
    return rows


def evaluate_rollout_counterfactuals(
    run_dir: Path,
    output_dir: Path,
    agents: Sequence[str],
    num_exposures: int,
    config: ExperimentConfig,
    variants: Sequence[str],
    bootstrap_samples: int,
    figures: bool,
) -> list[dict[str, Any]]:
    require_highway_deps(include_training=True)
    dqn_class = dqn_class_for_variant(config.dqn_variant)
    models = {
        agent: dqn_class.load(run_dir / "models" / f"{agent}_main.zip")
        for agent in agents
    }
    specs = make_eval_specs(num_exposures, config)
    summary_rows: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        steps, exposures = rollout_counterfactual_variant(
            models=models,
            agents=agents,
            specs=specs,
            config=config,
            variant=variant,
        )
        variant_dir = output_dir / variant
        eval_dir = variant_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(eval_dir / "steps.csv", steps)
        write_csv_rows(eval_dir / "exposures.csv", exposures)
        result = write_analysis(
            variant_dir,
            steps,
            exposures,
            config,
            bootstrap_samples=bootstrap_samples,
            figures=figures,
        )
        summary_rows.extend(rollout_summary_rows(variant, result))

    write_csv_rows(output_dir / "counterfactual_rollout_summary.csv", summary_rows)
    return summary_rows


def rollout_counterfactual_variant(
    models: Mapping[str, Any],
    agents: Sequence[str],
    specs: Sequence[SingleLaneSpec],
    config: ExperimentConfig,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exposure_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for spec in specs:
        exposure_recorded = False
        settings = counterfactual_settings(spec, variant)
        for agent in agents:
            env = make_env(agent, config, training=False)
            obs, _info = env.reset(seed=spec.exposure_seed)
            obs = apply_single_lane_scene(
                env,
                spec,
                include_front_vehicle=bool(settings["include_front_vehicle"]),
                front_distance=settings["front_distance"],
                front_speed=settings["front_speed"],
            )
            if not exposure_recorded:
                exposure_rows.append(
                    counterfactual_exposure_row_from_env(
                        env,
                        spec,
                        variant=variant,
                        settings=settings,
                    )
                )
                exposure_recorded = True

            model = models[agent]
            policy_id = f"{agent}_main"
            episode_id = f"{agent}_{variant}_{spec.exposure_id}_r0"
            done = False
            t = 0
            while not done and t < config.evaluation_max_policy_steps:
                action, q_values = predict_action_and_scores(
                    model,
                    obs,
                    deterministic=True,
                )
                action_name = action_name_from_env(env, int(action))
                pre_step_diagnostics = extract_diagnostics(env)
                next_obs, reward, terminated, truncated, _info = env.step(int(action))
                post_step_diagnostics = extract_diagnostics(env)
                env_done = bool(terminated or truncated)
                reached_horizon = t + 1 >= config.evaluation_max_policy_steps
                done = env_done or reached_horizon
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
                    termination_reason="collision" if collision else ("duration" if done else ""),
                    lane_change_count=0,
                    exposure_t=spec.exposure_t,
                )
                row["counterfactual_variant"] = variant
                row["include_front_vehicle"] = settings["include_front_vehicle"]
                row["configured_front_distance"] = finite_or_blank(
                    settings["front_distance"]
                )
                row["configured_front_speed"] = finite_or_blank(settings["front_speed"])
                row.update(policy_time_fields(t, spec.exposure_t, config))
                step_rows.append(row)
                obs = next_obs
                t += 1
            env.close()

    return step_rows, exposure_rows


def counterfactual_exposure_row_from_env(
    env,
    spec: SingleLaneSpec,
    variant: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = extract_diagnostics(env)
    front_speed = diagnostics["front_vehicle_speed"]
    include_front_vehicle = bool(settings["include_front_vehicle"])
    actual_spec = build_single_lane_spec(
        exposure_id=spec.exposure_id,
        exposure_seed=spec.exposure_seed,
        ego_speed=float(diagnostics["ego_speed"]),
        front_distance=(
            float(settings["front_distance"])
            if include_front_vehicle
            else 0.0
        ),
        front_speed=(
            float(settings["front_speed"])
            if include_front_vehicle
            else float(diagnostics["ego_speed"])
        ),
        scenario_type=f"counterfactual_{variant}",
        include_front_vehicle=include_front_vehicle,
        exposure_t=spec.exposure_t,
    )
    relative_closing = (
        max(0.0, float(diagnostics["ego_speed"]) - float(front_speed))
        if front_speed is not None
        else 0.0
    )
    policy_frequency = int(env.unwrapped.config["policy_frequency"])
    return {
        "exposure_id": spec.exposure_id,
        "exposure_seed": spec.exposure_seed,
        "ego_lane": diagnostics["ego_lane"],
        "ego_speed_at_exposure": round(float(diagnostics["ego_speed"]), 6),
        "nearest_front_distance_at_exposure": finite_or_blank(
            diagnostics["nearest_front_distance"]
        ),
        "front_net_distance_at_exposure": finite_or_blank(actual_spec.net_distance),
        "front_vehicle_speed_at_exposure": finite_or_blank(front_speed),
        "relative_closing_speed_at_exposure": round(relative_closing, 6),
        "signed_closing_speed_at_exposure": round(actual_spec.closing_speed, 6),
        "ttc_seconds_at_exposure": finite_or_blank(actual_spec.ttc_seconds),
        "required_deceleration_mps2_at_exposure": round(
            actual_spec.required_deceleration,
            6,
        ),
        "ttc_bin": actual_spec.ttc_bin,
        "required_deceleration_bin": actual_spec.required_deceleration_bin,
        "visible_at_t0": actual_spec.visible_at_t0,
        "vehicles_density": 0.0,
        "background_vehicle_state": json_dumps(
            {
                "source": "single_lane_slow_front_counterfactual",
                "counterfactual_variant": variant,
                "ego_lane": 0,
                "include_front_vehicle": settings["include_front_vehicle"],
                "configured_front_distance": finite_or_blank(
                    settings["front_distance"]
                ),
                "configured_front_speed": finite_or_blank(settings["front_speed"]),
                "action_space": ["SLOWER", "IDLE", "FASTER"],
            }
        ),
        "exposure_source": "single_lane_slow_front_counterfactual",
        "exposure_difficulty_bin": (
            f"{variant}:{actual_spec.ttc_bin}__"
            f"{actual_spec.required_deceleration_bin}"
        ),
        "counterfactual_variant": variant,
        "include_front_vehicle": settings["include_front_vehicle"],
        "configured_front_distance": finite_or_blank(settings["front_distance"]),
        "configured_front_speed": finite_or_blank(settings["front_speed"]),
        "exposure_t": spec.exposure_t,
        "exposure_t_seconds": round(spec.exposure_t / policy_frequency, 6),
        "policy_frequency_hz": policy_frequency,
    }


def rollout_summary_rows(variant: str, result: PipelineResult) -> list[dict[str, Any]]:
    policy_step_seconds = (
        float(result.steps[0].get("policy_step_seconds", 1.0))
        if result.steps
        else 1.0
    )
    action_counts: dict[str, Counter[str]] = {}
    for row in result.steps:
        agent = str(row["agent_condition"])
        action_counts.setdefault(agent, Counter())[str(row["action"])] += 1

    rows: list[dict[str, Any]] = []
    outcomes_by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in result.episode_outcomes:
        outcomes_by_agent.setdefault(str(row["agent_condition"]), []).append(row)

    for agent, outcomes in sorted(outcomes_by_agent.items()):
        valid_latencies = [
            float(row["response_latency"])
            for row in outcomes
            if row.get("response_latency") not in (None, "")
        ]
        outcome_counts = Counter(str(row["episode_outcome"]) for row in outcomes)
        rows.append(
            {
                "record_type": "agent_outcome",
                "counterfactual_variant": variant,
                "agent_condition": agent,
                "gap_label": "",
                "n": len(outcomes),
                "valid_onset_count": outcome_counts.get("valid_onset", 0),
                "no_onset_censored_count": outcome_counts.get("no_onset_censored", 0),
                "terminal_failure_count": outcome_counts.get("terminal_failure", 0),
                "valid_onset_rate": round(
                    outcome_counts.get("valid_onset", 0) / len(outcomes),
                    6,
                )
                if outcomes
                else "",
                "median_response_latency": median_value(valid_latencies),
                "mean_response_latency": mean_value(valid_latencies),
                "median_response_latency_seconds": median_value(
                    [value * policy_step_seconds for value in valid_latencies]
                ),
                "mean_response_latency_seconds": mean_value(
                    [value * policy_step_seconds for value in valid_latencies]
                ),
                "action_counts": json_dumps(
                    dict(sorted(action_counts.get(agent, Counter()).items()))
                ),
                "median_gap": "",
                "median_gap_seconds": "",
                "q1_gap": "",
                "q1_gap_seconds": "",
                "q3_gap": "",
                "q3_gap_seconds": "",
                "n_valid_pairs": "",
                "n_pairs": "",
            }
        )

    for row in result.gap_summary:
        rows.append(
            {
                "record_type": "gap_summary",
                "counterfactual_variant": variant,
                "agent_condition": "",
                "gap_label": row["gap_label"],
                "n": "",
                "valid_onset_count": "",
                "no_onset_censored_count": "",
                "terminal_failure_count": "",
                "valid_onset_rate": "",
                "median_response_latency": "",
                "mean_response_latency": "",
                "median_response_latency_seconds": "",
                "mean_response_latency_seconds": "",
                "action_counts": "",
                "median_gap": row["median_gap"],
                "median_gap_seconds": seconds_from_policy_steps(
                    row["median_gap"],
                    policy_step_seconds,
                ),
                "q1_gap": row["q1_gap"],
                "q1_gap_seconds": seconds_from_policy_steps(
                    row["q1_gap"],
                    policy_step_seconds,
                ),
                "q3_gap": row["q3_gap"],
                "q3_gap_seconds": seconds_from_policy_steps(
                    row["q3_gap"],
                    policy_step_seconds,
                ),
                "n_valid_pairs": row["n_valid_pairs"],
                "n_pairs": row["n_pairs"],
            }
        )
    return rows


def median_value(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 6)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 6)


def mean_value(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return round(sum(float(value) for value in values) / len(values), 6)


def seconds_from_policy_steps(
    value: Any,
    policy_step_seconds: float,
) -> float | str:
    if value in (None, ""):
        return ""
    return round(float(value) * policy_step_seconds, 6)


def summarize_counterfactual_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["agent_condition"]), str(row["counterfactual_variant"])),
            [],
        ).append(row)

    summary: list[dict[str, Any]] = []
    for (agent, variant), group in sorted(grouped.items()):
        counts = Counter(str(row["action"]) for row in group)
        slow = sum(1 for row in group if row["is_slow_action"])
        q_advantages = [
            float(row["slower_q_advantage"])
            for row in group
            if row.get("slower_q_advantage") not in (None, "")
        ]
        summary.append(
            {
                "agent_condition": agent,
                "counterfactual_variant": variant,
                "n": len(group),
                "slow_action_count": slow,
                "slow_action_rate": round(slow / len(group), 6) if group else "",
                "action_counts": json_dumps(dict(sorted(counts.items()))),
                "mean_slower_q_advantage": round(
                    sum(q_advantages) / len(q_advantages),
                    6,
                )
                if q_advantages
                else "",
            }
        )
    return summary


def exposure_row_from_spec(env, spec: SingleLaneSpec) -> dict[str, Any]:
    diagnostics = extract_diagnostics(env)
    policy_frequency = int(env.unwrapped.config["policy_frequency"])
    return {
        "exposure_id": spec.exposure_id,
        "exposure_seed": spec.exposure_seed,
        "ego_lane": diagnostics["ego_lane"],
        "ego_speed_at_exposure": round(float(diagnostics["ego_speed"]), 6),
        "nearest_front_distance_at_exposure": round(spec.front_distance, 6),
        "front_net_distance_at_exposure": round(spec.net_distance, 6),
        "front_vehicle_speed_at_exposure": round(spec.front_speed, 6),
        "relative_closing_speed_at_exposure": round(max(0.0, spec.closing_speed), 6),
        "signed_closing_speed_at_exposure": round(spec.closing_speed, 6),
        "ttc_seconds_at_exposure": finite_or_blank(spec.ttc_seconds),
        "required_deceleration_mps2_at_exposure": round(
            spec.required_deceleration,
            6,
        ),
        "ttc_bin": spec.ttc_bin,
        "required_deceleration_bin": spec.required_deceleration_bin,
        "visible_at_t0": spec.visible_at_t0,
        "vehicles_density": 0.0,
        "background_vehicle_state": json_dumps(
            {
                "source": "single_lane_slow_front",
                "ego_lane": 0,
                "front_distance": spec.front_distance,
                "front_net_distance": spec.net_distance,
                "front_speed": spec.front_speed,
                "ttc_seconds": finite_or_blank(spec.ttc_seconds),
                "required_deceleration_mps2": spec.required_deceleration,
                "action_space": ["SLOWER", "IDLE", "FASTER"],
            }
        ),
        "exposure_source": "single_lane_slow_front",
        "exposure_difficulty_bin": (
            f"{spec.ttc_bin}__{spec.required_deceleration_bin}"
        ),
        "exposure_t": spec.exposure_t,
        "exposure_t_seconds": round(spec.exposure_t / policy_frequency, 6),
        "policy_frequency_hz": policy_frequency,
    }


def add_analysis_seconds(result: PipelineResult, config: ExperimentConfig) -> None:
    step_seconds = config.policy_step_seconds
    for row in result.episode_outcomes:
        for key in ("termination_t", "response_latency", "censoring_time"):
            row[f"{key}_seconds"] = seconds_from_policy_steps(
                row.get(key),
                step_seconds,
            )
    for row in result.timing_gaps:
        for key in ("latency_a", "latency_b", "gap_b_minus_a"):
            row[f"{key}_seconds"] = seconds_from_policy_steps(
                row.get(key),
                step_seconds,
            )
    for row in result.gap_summary:
        for key in (
            "median_gap",
            "mean_gap",
            "q1_gap",
            "q3_gap",
            "iqr_gap",
            "bootstrap_median_ci_low",
            "bootstrap_median_ci_high",
        ):
            row[f"{key}_seconds"] = seconds_from_policy_steps(
                row.get(key),
                step_seconds,
            )


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
        SLOWDOWN_ONSET_TARGET,
        reward_config=reward_config_table_with_strength_multiplier(
            config.slow_down_penalty,
            config.collision_risk_penalty,
            config.reward_strength_multiplier,
            config.collision_penalty,
        ),
    )
    add_analysis_seconds(result, config)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in result.tables().items():
        write_csv_rows(analysis_dir / f"{name}.csv", rows)
    if figures:
        write_optional_figures(
            analysis_dir,
            result.timing_gaps,
            result.episode_outcomes,
        )
        write_training_convergence_figures(output_dir, analysis_dir)
    return result


def print_summary(result: PipelineResult, output_dir: Path) -> None:
    analysis_dir = output_dir / "analysis"
    print(f"Wrote single-lane slowdown analysis to {analysis_dir}")
    print(
        "Rows: "
        f"steps={len(result.steps)}, "
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


def print_counterfactual_summary(output_dir: Path) -> None:
    import csv

    path = output_dir / "counterfactual_action_summary.csv"
    if not path.exists():
        return
    print("Counterfactual slow-action summary:")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            print(
                "  "
                f"{row['agent_condition']} {row['counterfactual_variant']}: "
                f"slow={row['slow_action_count']}/{row['n']}, "
                f"actions={row['action_counts']}"
            )


def print_rollout_counterfactual_summary(output_dir: Path) -> None:
    import csv

    path = output_dir / "counterfactual_rollout_summary.csv"
    if not path.exists():
        return
    print("Rollout counterfactual slowdown summary:")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] != "agent_outcome":
                continue
            print(
                "  "
                f"{row['counterfactual_variant']} {row['agent_condition']}: "
                f"valid={row['valid_onset_count']}/{row['n']}, "
                f"median_latency={row['median_response_latency']}, "
                f"censored={row['no_onset_censored_count']}, "
                f"terminal={row['terminal_failure_count']}, "
                f"actions={row['action_counts']}"
            )


def audit_training_scenarios(
    output_dir: Path,
    num_resets: int,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if num_resets < len(STRATIFIED_TRAINING_BLOCK):
        raise ValueError(
            "num_resets must cover at least one complete stratified block "
            f"({len(STRATIFIED_TRAINING_BLOCK)})"
        )
    specs = make_training_specs(num_resets, config)
    rows = [
        training_spec_row(index, spec, config)
        for index, spec in enumerate(specs)
    ]
    family_counts = Counter(spec.scenario_type for spec in specs)
    difficulty_counts = Counter(
        (
            spec.scenario_type,
            spec.ttc_bin,
            spec.required_deceleration_bin,
        )
        for spec in specs
    )
    summary_rows: list[dict[str, Any]] = []
    for family, count in sorted(family_counts.items()):
        summary_rows.append(
            {
                "record_type": "family",
                "scenario_type": family,
                "ttc_bin": "",
                "required_deceleration_bin": "",
                "n": count,
                "fraction": round(count / num_resets, 6),
            }
        )
    for (family, ttc_bin, deceleration_bin), count in sorted(
        difficulty_counts.items()
    ):
        summary_rows.append(
            {
                "record_type": "difficulty",
                "scenario_type": family,
                "ttc_bin": ttc_bin,
                "required_deceleration_bin": deceleration_bin,
                "n": count,
                "fraction": round(count / num_resets, 6),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "training_scenario_audit.csv", rows)
    write_csv_rows(output_dir / "training_scenario_audit_summary.csv", summary_rows)
    return rows, summary_rows


def immediate_slowdown_oracle_rows(
    specs: Sequence[SingleLaneSpec],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    representative_specs: dict[tuple[str, str, str], SingleLaneSpec] = {}
    for spec in specs:
        if not spec.include_front_vehicle or spec.closing_speed <= 0.0:
            continue
        key = (
            spec.scenario_type,
            spec.ttc_bin,
            spec.required_deceleration_bin,
        )
        representative_specs.setdefault(key, spec)

    rows: list[dict[str, Any]] = []
    for key, spec in sorted(representative_specs.items()):
        env = make_env("FD", config, training=False)
        _obs, _info = env.reset(seed=spec.exposure_seed)
        apply_single_lane_scene(env, spec)
        slower_action = next(
            action_index
            for action_index, action_name in action_names_from_env(env).items()
            if action_name == "SLOWER"
        )
        collision = False
        min_front_distance = float("inf")
        steps = 0
        for t in range(config.evaluation_max_policy_steps):
            _obs, _reward, terminated, truncated, _info = env.step(slower_action)
            diagnostics = extract_diagnostics(env)
            min_front_distance = min(
                min_front_distance,
                float(diagnostics["nearest_front_distance"]),
            )
            collision = bool(diagnostics["collision_flag"])
            steps = t + 1
            if collision or terminated or truncated:
                break
        final_diagnostics = extract_diagnostics(env)
        rows.append(
            {
                "scenario_type": key[0],
                "ttc_bin": key[1],
                "required_deceleration_bin": key[2],
                "exposure_id": spec.exposure_id,
                "run_seed": config.seed,
                "initial_ttc_seconds": round(spec.ttc_seconds, 6),
                "initial_required_deceleration_mps2": round(
                    spec.required_deceleration,
                    6,
                ),
                "initial_ego_speed": spec.ego_speed,
                "initial_front_speed": spec.front_speed,
                "initial_front_center_distance": spec.front_distance,
                "oracle_action": "SLOWER",
                "oracle_collision": collision,
                "oracle_pass": not collision,
                "oracle_steps": steps,
                "oracle_duration_seconds": round(
                    steps * config.policy_step_seconds,
                    6,
                ),
                "minimum_front_center_distance": finite_or_blank(
                    min_front_distance
                ),
                "final_ego_speed": round(
                    float(final_diagnostics["ego_speed"]),
                    6,
                ),
                "target_speeds_mps": json_dumps(list(config.target_speeds)),
            }
        )
        env.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    configure_headless_runtime()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "audit":
        args = build_audit_parser().parse_args(raw_args[1:])
        config = ExperimentConfig(
            duration=args.duration,
            evaluation_duration=args.evaluation_duration,
            seed=args.seed,
            training_scenario_profile=args.training_scenario_profile,
            target_speeds=tuple(args.target_speeds),
            near_matched_speed_delta=args.near_matched_speed_delta,
        )
        output_dir = Path(args.out)
        rows, summary_rows = audit_training_scenarios(
            output_dir,
            args.num_resets,
            config,
        )
        specs = make_training_specs(args.num_resets, config)
        oracle_rows = immediate_slowdown_oracle_rows(specs, config)
        write_csv_rows(output_dir / "immediate_slowdown_oracle.csv", oracle_rows)
        oracle_pass = bool(oracle_rows) and all(
            bool(row["oracle_pass"]) for row in oracle_rows
        )
        print(
            f"Wrote scenario audit to {output_dir} "
            f"(resets={len(rows)}, summary_rows={len(summary_rows)}, "
            f"oracle_cases={len(oracle_rows)}, oracle_pass={oracle_pass})"
        )
        return 0 if oracle_pass else 1

    if raw_args and raw_args[0] == "counterfactual":
        args = build_counterfactual_parser().parse_args(raw_args[1:])
        config = config_from_args(args)
        run_dir = Path(args.run_dir)
        output_dir = (
            Path(args.out)
            if args.out
            else run_dir / "counterfactual_front_vehicle"
        )
        rows = evaluate_counterfactuals(
            run_dir=run_dir,
            output_dir=output_dir,
            agents=args.agents,
            num_exposures=args.num_exposures,
            config=config,
            variants=args.counterfactual_variants,
        )
        print(
            f"Wrote single-lane counterfactual analysis to {output_dir} "
            f"(rows={len(rows)})"
        )
        print_counterfactual_summary(output_dir)
        return 0

    if raw_args and raw_args[0] == "rollout-counterfactual":
        args = build_rollout_counterfactual_parser().parse_args(raw_args[1:])
        config = config_from_args(args)
        run_dir = Path(args.run_dir)
        output_dir = (
            Path(args.out)
            if args.out
            else run_dir / "rollout_counterfactual_front_vehicle"
        )
        summary_rows = evaluate_rollout_counterfactuals(
            run_dir=run_dir,
            output_dir=output_dir,
            agents=args.agents,
            num_exposures=args.num_exposures,
            config=config,
            variants=args.counterfactual_variants,
            bootstrap_samples=args.bootstrap_samples,
            figures=not args.no_figures,
        )
        print(
            f"Wrote single-lane rollout counterfactual analysis to {output_dir} "
            f"(summary_rows={len(summary_rows)})"
        )
        print_rollout_counterfactual_summary(output_dir)
        return 0

    args = build_parser().parse_args(raw_args)
    config = config_from_args(args)
    output_dir = Path(args.out)
    train_agents(
        output_dir,
        args.agents,
        args.timesteps,
        config,
        args.verbose,
        checkpoint_every=args.checkpoint_every,
    )
    if args.no_model_selection:
        print("Model selection skipped; using the final checkpoint.", flush=True)
    else:
        select_models(output_dir, args.agents, args.num_exposures, config)
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
