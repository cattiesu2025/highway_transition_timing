"""Isolated open-lane multi-lane lane-change timing experiment.

Training uses a deterministic difficulty-stratified mixture in a controlled
multi-lane scene. Matched evaluation episodes contain the ego vehicle and one
slower front vehicle, with empty adjacent lanes and no random traffic. The goal
is to measure whether FD/BAL/SP policies produce different lane-change timing
when lane change is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
TRAINING_SCENARIO_PROFILES = ("stratified", "legacy-random")
TRAINING_SPEC_SEED_OFFSET = 1_000_000
TRAINING_BLOCK_SEED_OFFSET = 2_000_000
NEAR_MATCHED_SPEED_MIN_ABS_DELTA = 0.25
VEHICLE_LENGTH_METRES = 5.0
OBSERVATION_DISTANCE_METRES = 200.0
VISIBLE_TRAIN_DISTANCE_RANGE = (90.0, 195.0)
BOUNDARY_VISIBLE_TRAIN_DISTANCE_RANGE = (180.0, 195.0)
SLOW_FRONT_SPEED_RANGE = (10.0, 20.0)
EGO_SPEED_RANGE = (24.0, 31.0)
STRATIFIED_TRAINING_BLOCK = (
    ("train_no_front", "", ""),
    ("train_no_front", "", ""),
    ("train_no_front", "", ""),
    ("train_no_front", "", ""),
    ("train_exact_matched_speed_front", "non_closing", "none"),
    ("train_exact_matched_speed_front", "non_closing", "none"),
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
    ("train_boundary_visible_front", "gradual", "gentle"),
    ("train_boundary_visible_front", "gradual", "gentle"),
)
VISIBLE_DIFFICULTY_CELLS = {
    ("gradual", "gentle"): ((20.001, 30.0), (0.10, 0.25)),
    ("easy", "gentle"): ((12.001, 20.0), (0.20, 0.499)),
    ("medium", "moderate"): ((8.001, 12.0), (0.50, 0.999)),
    ("hard", "strong"): ((5.001, 8.0), (1.00, 1.999)),
}
SEALED_HELDOUT_GRID_PATH = Path(__file__).with_name("heldout_grid_v1.csv")
SEALED_HELDOUT_GRID_SHA256 = (
    "d861b169fb618b0053f972f128fa498d3249d48969209db99e342af7732e6964"
)


@dataclass(frozen=True)
class ExperimentConfig:
    duration: int = 120
    evaluation_duration: int = 120
    seed: int = 0
    lanes_count: int = 2
    ego_lane: int = 1
    policy_frequency: int = 5
    simulation_frequency: int = 15
    observation_normalize: bool = True
    observation_absolute: bool = False
    slow_down_penalty: float = 0.2
    lane_change_penalty: float = 0.2
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
    no_front_train_fraction: float = 0.2
    near_matched_speed_delta: float = 2.0

    def __post_init__(self) -> None:
        if self.duration <= 0 or self.evaluation_duration <= 0:
            raise ValueError("duration values must be positive seconds")
        if self.policy_frequency <= 0 or self.simulation_frequency <= 0:
            raise ValueError("simulation frequencies must be positive")
        if self.lanes_count < 2:
            raise ValueError("lanes_count must be >= 2 for a multi-lane experiment")
        if not 0 <= self.ego_lane < self.lanes_count:
            raise ValueError("ego_lane must be a valid lane index")
        if self.training_scenario_profile not in TRAINING_SCENARIO_PROFILES:
            raise ValueError(
                "training_scenario_profile must be one of "
                f"{TRAINING_SCENARIO_PROFILES}"
            )
        if not 0.0 <= self.no_front_train_fraction <= 1.0:
            raise ValueError("no_front_train_fraction must be in [0, 1]")
        if self.near_matched_speed_delta < NEAR_MATCHED_SPEED_MIN_ABS_DELTA:
            raise ValueError(
                "near_matched_speed_delta must be >= "
                f"{NEAR_MATCHED_SPEED_MIN_ABS_DELTA}"
            )
        if self.lane_change_penalty < 0.0:
            raise ValueError("lane_change_penalty must be non-negative")

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


class OpenLaneTrainingResetWrapper(_GymWrapper):
    """Replace training resets with sampled slow-front or no-front scenes."""

    def __init__(self, env, config: ExperimentConfig):
        if _GymWrapper is object:
            self.env = env
        else:
            super().__init__(env)
        self.config = config
        self.reset_count = 0
        self.episode_step_count = 0
        self.last_training_exposure: OpenLaneSpec | None = None
        self.training_variant_counts: Counter[str] = Counter()
        self.training_spec_rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_step_count = 0
        reset_index = self.reset_count
        spec = make_training_spec(reset_index, self.config)
        self.reset_count += 1
        self.last_training_exposure = spec
        self.training_variant_counts[spec.scenario_type] += 1
        self.training_spec_rows.append(training_spec_row(reset_index, spec, self.config))
        obs = apply_open_lane_scene(
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
    parser.add_argument("--lanes-count", type=int, default=2)
    parser.add_argument("--ego-lane", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
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
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--collision-risk-penalty", type=float, default=3.0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--slow-down-penalty", type=float, default=0.2)
    parser.add_argument("--lane-change-penalty", type=float, default=0.2)
    parser.add_argument("--absolute-observation", action="store_true")
    parser.add_argument("--no-normalize-observation", action="store_true")
    parser.add_argument(
        "--training-scenario-profile",
        choices=list(TRAINING_SCENARIO_PROFILES),
        default="stratified",
        help=(
            "Use the deterministic TTC/deceleration block by default; "
            "legacy-random reproduces the previous probabilistic mixture."
        ),
    )
    parser.add_argument(
        "--no-front-train-fraction",
        type=float,
        default=0.2,
        help=(
            "Legacy-random fraction of training resets with no front vehicle. "
            "This option does not alter the fixed stratified block."
        ),
    )
    parser.add_argument(
        "--near-matched-speed-delta",
        type=float,
        default=2.0,
        help="Maximum absolute speed difference for near-matched training scenes.",
    )
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit multi-lane stratified training scenes without training."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-resets", type=int, default=2_000)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--evaluation-duration", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lanes-count", type=int, default=2)
    parser.add_argument("--ego-lane", type=int, default=1)
    parser.add_argument(
        "--training-scenario-profile",
        choices=list(TRAINING_SCENARIO_PROFILES),
        default="stratified",
    )
    parser.add_argument("--no-front-train-fraction", type=float, default=0.2)
    parser.add_argument("--near-matched-speed-delta", type=float, default=2.0)
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
        lane_change_penalty=args.lane_change_penalty,
        collision_risk_penalty=args.collision_risk_penalty,
        collision_penalty=args.collision_penalty,
        learning_rate=args.learning_rate,
        training_scenario_profile=args.training_scenario_profile,
        no_front_train_fraction=args.no_front_train_fraction,
        near_matched_speed_delta=args.near_matched_speed_delta,
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
    # highway-env refreshes its observation space on reset after configure().
    # Do this before adding the training wrapper so the initialization scene is
    # not recorded as a scene consumed by model.learn().
    env.reset(seed=config.seed)
    if training:
        env = OpenLaneTrainingResetWrapper(env, config)
    weights = with_common_slow_down_penalty(
        MAIN_REWARD_WEIGHTS[agent_condition],
        config.slow_down_penalty,
        config.collision_risk_penalty,
        config.collision_penalty,
        config.lane_change_penalty,
    )
    return OpenLaneRewardWrapper(env, agent_condition, weights)


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


def build_open_lane_spec(
    *,
    exposure_id: str,
    exposure_seed: int,
    ego_speed: float,
    front_distance: float,
    front_speed: float,
    scenario_type: str,
    include_front_vehicle: bool,
    ego_lane: int,
    ego_longitudinal: float = 100.0,
    exposure_t: int = 0,
) -> OpenLaneSpec:
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
    return OpenLaneSpec(
        exposure_id=exposure_id,
        exposure_seed=exposure_seed,
        ego_speed=rounded_ego_speed,
        front_distance=rounded_front_distance,
        front_speed=rounded_front_speed,
        ego_lane=ego_lane,
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


def make_training_spec(reset_index: int, config: ExperimentConfig) -> OpenLaneSpec:
    if config.training_scenario_profile == "legacy-random":
        return make_legacy_random_training_spec(reset_index, config)
    return make_stratified_training_spec(reset_index, config)


def make_stratified_training_spec(
    reset_index: int,
    config: ExperimentConfig,
) -> OpenLaneSpec:
    training_seed = config.seed + TRAINING_SPEC_SEED_OFFSET + reset_index
    rng = random.Random(training_seed)
    scenario_type, expected_ttc_bin, expected_deceleration_bin = (
        stratified_training_slot(reset_index, config.seed)
    )

    if scenario_type == "train_no_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        return build_open_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=training_seed,
            ego_speed=ego_speed,
            front_distance=0.0,
            front_speed=ego_speed,
            scenario_type=scenario_type,
            include_front_vehicle=False,
            ego_lane=config.ego_lane,
        )

    if scenario_type == "train_exact_matched_speed_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        return build_open_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=training_seed,
            ego_speed=ego_speed,
            front_distance=rng.uniform(*VISIBLE_TRAIN_DISTANCE_RANGE),
            front_speed=ego_speed,
            scenario_type=scenario_type,
            include_front_vehicle=True,
            ego_lane=config.ego_lane,
        )

    if scenario_type == "train_non_closing_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed + rng.uniform(
            NEAR_MATCHED_SPEED_MIN_ABS_DELTA,
            config.near_matched_speed_delta,
        )
        return build_open_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=training_seed,
            ego_speed=ego_speed,
            front_distance=rng.uniform(*VISIBLE_TRAIN_DISTANCE_RANGE),
            front_speed=front_speed,
            scenario_type=scenario_type,
            include_front_vehicle=True,
            ego_lane=config.ego_lane,
        )

    if scenario_type == "train_near_closing_front":
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed - rng.uniform(
            NEAR_MATCHED_SPEED_MIN_ABS_DELTA,
            config.near_matched_speed_delta,
        )
        return build_open_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=training_seed,
            ego_speed=ego_speed,
            front_distance=rng.uniform(*VISIBLE_TRAIN_DISTANCE_RANGE),
            front_speed=front_speed,
            scenario_type=scenario_type,
            include_front_vehicle=True,
            ego_lane=config.ego_lane,
        )

    if scenario_type == "train_boundary_visible_front":
        return sample_boundary_visible_training_spec(
            reset_index,
            training_seed,
            rng,
            config,
        )

    return sample_visible_slow_front_training_spec(
        reset_index=reset_index,
        training_seed=training_seed,
        rng=rng,
        expected_ttc_bin=expected_ttc_bin,
        expected_deceleration_bin=expected_deceleration_bin,
        config=config,
    )


def stratified_training_slot(
    reset_index: int,
    seed: int,
) -> tuple[str, str, str]:
    block_index, block_position = divmod(
        reset_index,
        len(STRATIFIED_TRAINING_BLOCK),
    )
    block = list(STRATIFIED_TRAINING_BLOCK)
    block_rng = random.Random(seed + TRAINING_BLOCK_SEED_OFFSET + block_index)
    block_rng.shuffle(block)
    return block[block_position]


def sample_visible_slow_front_training_spec(
    *,
    reset_index: int,
    training_seed: int,
    rng: random.Random,
    expected_ttc_bin: str,
    expected_deceleration_bin: str,
    config: ExperimentConfig,
) -> OpenLaneSpec:
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
        spec = build_open_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=training_seed,
            ego_speed=ego_speed,
            front_distance=front_distance,
            front_speed=front_speed,
            scenario_type="train_visible_slow_front",
            include_front_vehicle=True,
            ego_lane=config.ego_lane,
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


def sample_boundary_visible_training_spec(
    reset_index: int,
    training_seed: int,
    rng: random.Random,
    config: ExperimentConfig,
) -> OpenLaneSpec:
    for _attempt in range(10_000):
        front_distance = rng.uniform(*BOUNDARY_VISIBLE_TRAIN_DISTANCE_RANGE)
        net_distance = front_distance - VEHICLE_LENGTH_METRES
        ttc_seconds = rng.uniform(20.001, 35.0)
        closing_speed = net_distance / ttc_seconds
        ego_speed = rng.uniform(*EGO_SPEED_RANGE)
        front_speed = ego_speed - closing_speed
        if not SLOW_FRONT_SPEED_RANGE[0] <= front_speed <= SLOW_FRONT_SPEED_RANGE[1]:
            continue
        spec = build_open_lane_spec(
            exposure_id=f"train_reset_{reset_index:06d}",
            exposure_seed=training_seed,
            ego_speed=ego_speed,
            front_distance=front_distance,
            front_speed=front_speed,
            scenario_type="train_boundary_visible_front",
            include_front_vehicle=True,
            ego_lane=config.ego_lane,
        )
        if spec.ttc_bin == "gradual" and spec.required_deceleration_bin == "gentle":
            return spec
    raise RuntimeError("Unable to sample feasible boundary-visible slow-front scene")


def make_legacy_random_training_spec(
    reset_index: int,
    config: ExperimentConfig,
) -> OpenLaneSpec:
    rng = random.Random(config.seed + 1009 * reset_index)
    include_front_vehicle = rng.random() >= config.no_front_train_fraction
    return build_open_lane_spec(
        exposure_id=f"train_reset_{reset_index:06d}",
        exposure_seed=config.seed + reset_index,
        ego_lane=config.ego_lane,
        ego_speed=rng.uniform(24.0, 31.0),
        front_distance=rng.uniform(120.0, 280.0),
        front_speed=rng.uniform(10.0, 20.0),
        scenario_type=(
            "train_slow_front" if include_front_vehicle else "train_no_front"
        ),
        include_front_vehicle=include_front_vehicle,
    )


def training_spec_row(
    reset_index: int,
    spec: OpenLaneSpec,
    config: ExperimentConfig,
) -> dict[str, Any]:
    block_index, block_position = divmod(
        reset_index,
        len(STRATIFIED_TRAINING_BLOCK),
    )
    return {
        "reset_index": reset_index,
        "block_index": (
            block_index if config.training_scenario_profile == "stratified" else ""
        ),
        "block_position": (
            block_position if config.training_scenario_profile == "stratified" else ""
        ),
        "block_size": (
            len(STRATIFIED_TRAINING_BLOCK)
            if config.training_scenario_profile == "stratified"
            else ""
        ),
        "exposure_id": spec.exposure_id,
        "scene_seed": spec.exposure_seed,
        "training_scenario_profile": config.training_scenario_profile,
        "scenario_type": spec.scenario_type,
        "include_front_vehicle": spec.include_front_vehicle,
        "ego_lane": spec.ego_lane,
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
        "difficulty_bin": f"{spec.ttc_bin}__{spec.required_deceleration_bin}",
        "visible_at_t0": spec.visible_at_t0,
        "adjacent_lanes_open": True,
        "background_vehicles": 0,
        "policy_frequency_hz": config.policy_frequency,
        "policy_step_seconds": round(config.policy_step_seconds, 6),
    }


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
    if (
        config.training_scenario_profile == "stratified"
        and num_resets % len(STRATIFIED_TRAINING_BLOCK) != 0
    ):
        raise ValueError(
            "num_resets must be a multiple of the stratified block size "
            f"({len(STRATIFIED_TRAINING_BLOCK)})"
        )
    specs = [make_training_spec(index, config) for index in range(num_resets)]
    if config.training_scenario_profile == "stratified":
        validate_stratified_training_specs(specs)
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


def validate_stratified_training_specs(
    specs: Sequence[OpenLaneSpec],
) -> None:
    block_size = len(STRATIFIED_TRAINING_BLOCK)
    expected_family_counts = Counter(
        scenario_type
        for scenario_type, _ttc_bin, _deceleration_bin in STRATIFIED_TRAINING_BLOCK
    )
    expected_visible_difficulty_counts = Counter(
        (ttc_bin, deceleration_bin)
        for scenario_type, ttc_bin, deceleration_bin in STRATIFIED_TRAINING_BLOCK
        if scenario_type == "train_visible_slow_front"
    )
    for block_index, start in enumerate(range(0, len(specs), block_size)):
        block = specs[start : start + block_size]
        family_counts = Counter(spec.scenario_type for spec in block)
        if family_counts != expected_family_counts:
            raise RuntimeError(
                f"Stratified family quota mismatch in block {block_index}: "
                f"{dict(sorted(family_counts.items()))}"
            )
        visible_difficulty_counts = Counter(
            (spec.ttc_bin, spec.required_deceleration_bin)
            for spec in block
            if spec.scenario_type == "train_visible_slow_front"
        )
        if visible_difficulty_counts != expected_visible_difficulty_counts:
            raise RuntimeError(
                f"Visible difficulty quota mismatch in block {block_index}: "
                f"{dict(sorted(visible_difficulty_counts.items()))}"
            )

    for spec in specs:
        if spec.scenario_type == "train_no_front":
            if spec.include_front_vehicle or spec.visible_at_t0:
                raise RuntimeError(f"Invalid no-front scene: {spec.exposure_id}")
            continue
        if not spec.include_front_vehicle:
            raise RuntimeError(f"Missing front vehicle: {spec.exposure_id}")
        if spec.ttc_bin != classify_ttc(spec.ttc_seconds, spec.closing_speed):
            raise RuntimeError(f"TTC label mismatch: {spec.exposure_id}")
        if spec.required_deceleration_bin != classify_required_deceleration(
            spec.required_deceleration
        ):
            raise RuntimeError(
                f"Required-deceleration label mismatch: {spec.exposure_id}"
            )
        if spec.ttc_bin == "critical":
            raise RuntimeError(
                f"Critical TTC leaked into routine training: {spec.exposure_id}"
            )
        if not spec.visible_at_t0:
            raise RuntimeError(
                f"Visible/control scene is hidden at t0: {spec.exposure_id}"
            )


def make_eval_specs(num_exposures: int, config: ExperimentConfig) -> list[OpenLaneSpec]:
    front_distances = [140.0, 180.0, 220.0, 260.0]
    front_speeds = [10.0, 14.0, 18.0]
    ego_speeds = [26.0, 28.0, 30.0]
    specs: list[OpenLaneSpec] = []
    for idx in range(num_exposures):
        specs.append(
            build_open_lane_spec(
                exposure_id=f"M{idx:04d}",
                exposure_seed=config.seed + idx,
                ego_lane=config.ego_lane,
                ego_speed=ego_speeds[
                    (idx // (len(front_distances) * len(front_speeds)))
                    % len(ego_speeds)
                ],
                front_distance=front_distances[idx % len(front_distances)],
                front_speed=front_speeds[(idx // len(front_distances)) % len(front_speeds)],
                scenario_type="open_lane_slow_front",
                include_front_vehicle=True,
            )
        )
    return specs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_specs(
    path: Path,
    config: ExperimentConfig,
    expected_sha256: str | None = None,
) -> list[OpenLaneSpec]:
    """Load a fixed evaluation grid and validate its physical scene fields."""

    if not path.exists():
        raise FileNotFoundError(f"Evaluation grid does not exist: {path}")
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Evaluation grid checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    required = {
        "exposure_id",
        "exposure_seed",
        "ego_speed",
        "front_distance",
        "front_speed",
        "ego_lane",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Evaluation grid is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Evaluation grid must contain at least one exposure")

    specs: list[OpenLaneSpec] = []
    for row in rows:
        spec = build_open_lane_spec(
            exposure_id=str(row["exposure_id"]),
            exposure_seed=int(row["exposure_seed"]),
            ego_lane=int(row["ego_lane"]),
            ego_speed=float(row["ego_speed"]),
            front_distance=float(row["front_distance"]),
            front_speed=float(row["front_speed"]),
            scenario_type="heldout_open_lane_slow_front",
            include_front_vehicle=True,
        )
        if spec.ego_lane != config.ego_lane:
            raise ValueError(
                f"Held-out ego lane {spec.ego_lane} does not match configured "
                f"ego lane {config.ego_lane}: {spec.exposure_id}"
            )
        if spec.front_distance <= VEHICLE_LENGTH_METRES:
            raise ValueError(f"Non-positive held-out net gap: {spec.exposure_id}")
        if spec.front_speed <= 0.0 or spec.ego_speed <= spec.front_speed:
            raise ValueError(
                f"Held-out original must be a closing slow-front scene: "
                f"{spec.exposure_id}"
            )
        specs.append(spec)

    exposure_ids = [spec.exposure_id for spec in specs]
    exposure_seeds = [spec.exposure_seed for spec in specs]
    physical_rows = [
        (spec.ego_speed, spec.front_distance, spec.front_speed, spec.ego_lane)
        for spec in specs
    ]
    if len(set(exposure_ids)) != len(exposure_ids):
        raise ValueError("Evaluation grid exposure_id values must be unique")
    if len(set(exposure_seeds)) != len(exposure_seeds):
        raise ValueError("Evaluation grid exposure_seed values must be unique")
    if len(set(physical_rows)) != len(physical_rows):
        raise ValueError("Evaluation grid physical scene rows must be unique")
    return specs


def make_sealed_heldout_specs(config: ExperimentConfig) -> list[OpenLaneSpec]:
    """Load the versioned final grid and enforce its precommitted checksum."""

    specs = load_eval_specs(
        SEALED_HELDOUT_GRID_PATH,
        config,
        expected_sha256=SEALED_HELDOUT_GRID_SHA256,
    )
    if len(specs) != 36:
        raise RuntimeError(f"Sealed held-out grid must have 36 rows, got {len(specs)}")

    development = make_eval_specs(36, config)
    for field in ("ego_speed", "front_distance", "front_speed"):
        development_values = {getattr(spec, field) for spec in development}
        heldout_values = {getattr(spec, field) for spec in specs}
        overlap = development_values.intersection(heldout_values)
        if overlap:
            raise RuntimeError(
                f"Sealed held-out {field} values overlap development: "
                f"{sorted(overlap)}"
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
    callback_factory: Callable[[str, Any, Any], Any] | None = None,
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
        effective_reward_weights = with_common_slow_down_penalty(
            MAIN_REWARD_WEIGHTS[agent],
            config.slow_down_penalty,
            config.collision_risk_penalty,
            config.collision_penalty,
            config.lane_change_penalty,
        )
        training_env = make_env(agent, config, training=True)
        agent_logs_dir = logs_dir / agent
        agent_logs_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(training_env, filename=str(agent_logs_dir / "monitor.csv"))
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
        callbacks: list[Any] = [
            CheckpointSaver(
                agent=agent,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                verbose=verbose,
            )
        ]
        if callback_factory is not None:
            callbacks.append(callback_factory(agent, model, training_env))
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False,
            callback=callbacks,
        )
        # The final checkpoint is the provisional policy. Model selection
        # overwrites this archive with the latest eligible checkpoint.
        model_path = models_dir / f"{agent}_main.zip"
        model.save(model_path)
        training_counts = dict(
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
                    f"reward_{key}": value
                    for key, value in asdict(effective_reward_weights).items()
                },
                "reward_slow_down_penalty": config.slow_down_penalty,
                "reward_collision_risk_penalty": config.collision_risk_penalty,
                "reward_common_collision_penalty": config.collision_penalty
                if config.collision_penalty is not None
                else "",
                "experiment": "multilane_open_lane_change",
                "lanes_count": config.lanes_count,
                "ego_lane": config.ego_lane,
                "training_duration_seconds": config.duration,
                "evaluation_duration_seconds": config.evaluation_duration,
                "policy_frequency_hz": config.policy_frequency,
                "policy_step_seconds": round(config.policy_step_seconds, 6),
                "training_max_policy_steps": config.training_max_policy_steps,
                "evaluation_max_policy_steps": config.evaluation_max_policy_steps,
                "training_scenario_profile": config.training_scenario_profile,
                "training_scenario_block_size": len(STRATIFIED_TRAINING_BLOCK),
                "training_scenario_block_family_counts": json_dumps(
                    dict(sorted(block_family_counts.items()))
                ),
                "training_no_front_fraction": config.no_front_train_fraction,
                "training_near_matched_speed_min_abs_delta": (
                    NEAR_MATCHED_SPEED_MIN_ABS_DELTA
                ),
                "training_near_matched_speed_max_abs_delta": (
                    config.near_matched_speed_delta
                ),
                "training_scene_counts": json_dumps(training_counts),
                "training_reset_count": len(agent_scenario_rows),
                "action_space": "LANE_LEFT,IDLE,LANE_RIGHT,FASTER,SLOWER",
                "blockers": "none",
                "background_vehicles": 0,
                "observation_normalize": config.observation_normalize,
                "observation_absolute": config.observation_absolute,
                "dqn_variant": config.dqn_variant,
            }
        )
    write_csv_rows(output_dir / "training_runs.csv", rows)
    write_csv_rows(output_dir / "training_scenarios.csv", scenario_rows)


def load_rollout_counterfactual_module():
    """Load the sibling counterfactual script for its variant rollout."""

    import importlib.util

    path = Path(__file__).with_name("rollout_counterfactual.py")
    spec = importlib.util.spec_from_file_location(
        "multilane_open_lane_change_rollout_counterfactual",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load counterfactual module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def development_gate_outcomes(
    model: Any,
    agent: str,
    num_exposures: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Evaluate one policy on the development grid under every gate variant.

    The sealed held-out grid is never loaded here; only `make_eval_specs`
    development exposures are used.
    """

    counterfactual = load_rollout_counterfactual_module()
    specs = make_eval_specs(num_exposures, config)
    models = {agent: model}
    outcomes: dict[str, Any] = {}
    for variant in GATE_VARIANTS:
        step_rows, exposure_rows = counterfactual.rollout_variant(
            sys.modules[__name__],
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
            LANE_CHANGE_ONSET_TARGET,
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
            config.lane_change_penalty,
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
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "audit":
        args = build_audit_parser().parse_args(raw_args[1:])
        config = ExperimentConfig(
            duration=args.duration,
            evaluation_duration=args.evaluation_duration,
            seed=args.seed,
            lanes_count=args.lanes_count,
            ego_lane=args.ego_lane,
            training_scenario_profile=args.training_scenario_profile,
            no_front_train_fraction=args.no_front_train_fraction,
            near_matched_speed_delta=args.near_matched_speed_delta,
        )
        output_dir = Path(args.out)
        rows, summary_rows = audit_training_scenarios(
            output_dir,
            args.num_resets,
            config,
        )
        print(
            f"Wrote multi-lane training scenario audit to {output_dir} "
            f"(resets={len(rows)}, summary_rows={len(summary_rows)})"
        )
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
