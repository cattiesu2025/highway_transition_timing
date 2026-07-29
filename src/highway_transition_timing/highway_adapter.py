"""Shared ``highway-env`` helpers for the maintained experiments.

The experiment-specific environment construction, reset distributions,
training loops, and evaluation loops live under ``experiments/``. This module
contains only simulator diagnostics and model helpers used by both experiments.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import FRONT_DISTANCE_SCORE_HORIZON
from .rewards import RewardWeights


ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def require_highway_deps(include_training: bool = False) -> None:
    """Raise a clear error when simulator dependencies are unavailable."""

    configure_headless_runtime()
    missing: list[str] = []
    for module_name in ["gymnasium", "highway_env", "numpy"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if include_training:
        try:
            __import__("stable_baselines3")
        except ImportError:
            missing.append("stable_baselines3")
    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"Missing optional Highway dependencies: {packages}. "
            "Install them with `python -m pip install -r requirements-pilot.txt`."
        )


def configure_headless_runtime() -> None:
    """Route GUI/cache state to a project-local ignored directory."""

    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def dqn_class_for_variant(variant: str):
    """Return the project's fixed Double DQN implementation."""

    if variant.lower() == "double-dqn":
        from .double_dqn import DoubleDQN

        return DoubleDQN
    raise ValueError(
        f"Unsupported DQN variant: {variant}. This project supports only double-dqn."
    )


def q_value_diagnostics(action_scores: Sequence[float]) -> dict[str, Any]:
    """Summarize action-score margins for diagnostics and report figures."""

    values = [float(value) for value in action_scores]
    if not values:
        return {
            "q_argmax_action": "",
            "q_top1": "",
            "q_top2": "",
            "q_top_margin": "",
            "lane_change_q_advantage": "",
            "slower_q_advantage": "",
        }

    argmax_index = max(range(len(values)), key=values.__getitem__)
    sorted_values = sorted(values, reverse=True)
    lane_indices = [
        index
        for index, name in ACTION_NAMES.items()
        if name in {"LANE_LEFT", "LANE_RIGHT"} and index < len(values)
    ]
    non_lane_indices = [
        index for index in range(len(values)) if index not in lane_indices
    ]
    slower_index = next(
        (
            index
            for index, name in ACTION_NAMES.items()
            if name == "SLOWER" and index < len(values)
        ),
        None,
    )
    lane_change_advantage = (
        max(values[index] for index in lane_indices)
        - max(values[index] for index in non_lane_indices)
        if lane_indices and non_lane_indices
        else ""
    )
    slower_advantage = (
        values[slower_index]
        - max(value for index, value in enumerate(values) if index != slower_index)
        if slower_index is not None and len(values) > 1
        else ""
    )
    return {
        "q_argmax_action": ACTION_NAMES.get(argmax_index, str(argmax_index)),
        "q_top1": round(sorted_values[0], 6),
        "q_top2": round(sorted_values[1], 6) if len(sorted_values) > 1 else "",
        "q_top_margin": (
            round(sorted_values[0] - sorted_values[1], 6)
            if len(sorted_values) > 1
            else ""
        ),
        "lane_change_q_advantage": (
            round(float(lane_change_advantage), 6)
            if lane_change_advantage != ""
            else ""
        ),
        "slower_q_advantage": (
            round(float(slower_advantage), 6) if slower_advantage != "" else ""
        ),
    }


def predict_action_and_scores(
    model,
    obs,
    deterministic: bool,
) -> tuple[int, list[float]]:
    """Predict one action and, when available, extract its Q values."""

    action, _state = model.predict(obs, deterministic=deterministic)
    action_int = int(action)
    q_values: list[float] = []
    try:
        import torch

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            values = model.q_net(obs_tensor).detach().cpu().numpy()[0]
        q_values = [float(value) for value in values]
    except Exception:
        q_values = []
    return action_int, q_values


def extract_diagnostics(env) -> dict[str, Any]:
    """Read observable Highway state and diagnostic scores."""

    vehicle = env.unwrapped.vehicle
    road_vehicles = list(getattr(env.unwrapped.road, "vehicles", []))
    ego_position = [float(vehicle.position[0]), float(vehicle.position[1])]
    ego_speed = float(getattr(vehicle, "speed", 0.0))
    ego_lane = ego_lane_index(vehicle)
    nearest = nearest_front_vehicle(vehicle, road_vehicles)
    front_distance = nearest["distance"]
    closest_distances = closest_k_distances(vehicle, road_vehicles, k=3)

    speed_score = clamp01((ego_speed - 20.0) / 15.0)
    front_distance_score = (
        clamp01(front_distance / FRONT_DISTANCE_SCORE_HORIZON)
        if front_distance != float("inf")
        else 1.0
    )
    closest_k_distance_score = (
        clamp01(min(closest_distances) / 50.0) if closest_distances else 1.0
    )
    collision_risk_score = closing_collision_risk_score(
        front_distance,
        ego_speed,
        nearest["speed"],
    )

    return {
        "ego_lane": ego_lane,
        "ego_speed": ego_speed,
        "ego_position": ego_position,
        "nearest_front_vehicle_id": nearest["vehicle_id"],
        "nearest_front_distance": front_distance,
        "front_vehicle_speed": nearest["speed"],
        "closest_k_distances": closest_distances,
        "speed_score": speed_score,
        "front_distance_score": front_distance_score,
        "closest_k_distance_score": closest_k_distance_score,
        "collision_risk_score": collision_risk_score,
        "lane_position_score": 0.0,
        "collision_flag": bool(getattr(vehicle, "crashed", False)),
    }


def reward_components(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "speed_score": float(diagnostics["speed_score"]),
        "front_distance_score": float(diagnostics["front_distance_score"]),
        "collision_penalty": -1.0 if diagnostics["collision_flag"] else 0.0,
        "collision_risk_penalty": -float(diagnostics["collision_risk_score"]),
        "lane_change_penalty": 0.0,
        "slow_down_penalty": 0.0,
        "right_lane_score": 0.0,
    }


def weighted_reward(
    components: Mapping[str, float],
    weights: RewardWeights,
) -> float:
    return (
        weights.speed_score * components["speed_score"]
        + weights.front_distance_score * components["front_distance_score"]
        + weights.collision_penalty * components["collision_penalty"]
        + weights.collision_risk_penalty
        * components.get("collision_risk_penalty", 0.0)
        - weights.lane_change_penalty * abs(components["lane_change_penalty"])
        - weights.slow_down_penalty
        * abs(components.get("slow_down_penalty", 0.0))
        + weights.right_lane_score * components["right_lane_score"]
    )


def closing_collision_risk_score(
    front_distance: float,
    ego_speed: float,
    front_speed: float | None,
) -> float:
    """Return an anticipatory score for quickly closing on a front vehicle."""

    if front_distance == float("inf"):
        return 0.0

    front_speed_value = (
        float(ego_speed) if front_speed is None else float(front_speed)
    )
    closing_speed = max(0.0, float(ego_speed) - front_speed_value)
    if closing_speed <= 0.1:
        time_to_collision_risk = 0.0
    else:
        time_to_collision_seconds = float(front_distance) / closing_speed
        time_to_collision_risk = clamp01(
            (6.0 - time_to_collision_seconds) / 6.0
        )

    very_close_risk = clamp01((24.0 - float(front_distance)) / 24.0)
    return max(time_to_collision_risk, very_close_risk)


def step_row_from_diagnostics(
    diagnostics: Mapping[str, Any],
    post_step_diagnostics: Mapping[str, Any],
    agent_condition: str,
    policy_id: str,
    episode_id: str,
    exposure_id: str,
    exposure_seed: int,
    rollout_id: str,
    rollout_seed: int,
    t: int,
    action_name: str,
    action_scores: Sequence[float],
    reward_total: float,
    reward_components: Mapping[str, float],
    collision_flag: bool,
    done: bool,
    termination_reason: str,
    lane_change_count: int,
    exposure_t: int = 0,
) -> dict[str, Any]:
    """Build one pipeline-compatible evaluation step row."""

    return {
        "episode_id": episode_id,
        "agent_condition": agent_condition,
        "policy_id": policy_id,
        "exposure_seed": exposure_seed,
        "exposure_id": exposure_id,
        "rollout_id": rollout_id,
        "rollout_seed": rollout_seed,
        "evaluation_policy_mode": "deterministic",
        "t": t,
        "ego_lane": diagnostics["ego_lane"],
        "ego_speed": round(float(diagnostics["ego_speed"]), 6),
        "ego_position": diagnostics["ego_position"],
        "action": action_name,
        "q_values_or_action_scores": [
            round(float(value), 6) for value in action_scores
        ],
        "policy_hidden_activations_or_probe_features": "",
        "nearest_front_vehicle_id": diagnostics["nearest_front_vehicle_id"],
        "nearest_front_distance": finite_or_blank(
            diagnostics["nearest_front_distance"]
        ),
        "front_vehicle_speed": finite_or_blank(
            diagnostics["front_vehicle_speed"]
        ),
        "closest_k_distances": [
            round(float(value), 6)
            for value in diagnostics["closest_k_distances"]
        ],
        "speed_score": round(float(diagnostics["speed_score"]), 6),
        "front_distance_score": round(
            float(diagnostics["front_distance_score"]),
            6,
        ),
        "closest_k_distance_score": round(
            float(diagnostics["closest_k_distance_score"]),
            6,
        ),
        "collision_risk_score": round(
            float(diagnostics["collision_risk_score"]),
            6,
        ),
        "lane_position_score": 0.0,
        "lane_change_count": lane_change_count,
        "reward_total": round(float(reward_total), 6),
        "reward_components": dict(reward_components),
        "collision_flag": collision_flag,
        "done": done,
        "termination_reason": termination_reason,
        "episode_outcome": (
            "collision" if post_step_diagnostics["collision_flag"] else ""
        ),
        "exposure_t": exposure_t,
    }


def nearest_front_vehicle(
    vehicle,
    road_vehicles: Sequence[Any],
) -> dict[str, Any]:
    ego_lane = ego_lane_index(vehicle)
    ego_x = float(vehicle.position[0])
    best_vehicle = None
    best_distance = float("inf")
    for other in road_vehicles:
        if other is vehicle or ego_lane_index(other) != ego_lane:
            continue
        distance = float(other.position[0]) - ego_x
        if 0.0 < distance < best_distance:
            best_distance = distance
            best_vehicle = other
    return {
        "vehicle_id": "" if best_vehicle is None else str(id(best_vehicle)),
        "distance": best_distance,
        "speed": (
            None
            if best_vehicle is None
            else float(getattr(best_vehicle, "speed", 0.0))
        ),
    }


def closest_k_distances(
    vehicle,
    road_vehicles: Sequence[Any],
    k: int = 3,
) -> list[float]:
    import numpy as np

    distances: list[float] = []
    ego_position = np.asarray(vehicle.position, dtype=float)
    for other in road_vehicles:
        if other is vehicle:
            continue
        other_position = np.asarray(other.position, dtype=float)
        distances.append(float(np.linalg.norm(other_position - ego_position)))
    return sorted(distances)[:k]


def ego_lane_index(vehicle) -> int:
    lane_index = getattr(vehicle, "lane_index", None)
    if isinstance(lane_index, tuple) and lane_index:
        return int(lane_index[-1])
    try:
        return int(lane_index)
    except (TypeError, ValueError):
        return 0


def action_name_from_env(env, action: int) -> str:
    action_type = getattr(env.unwrapped, "action_type", None)
    actions = getattr(action_type, "actions", None)
    if isinstance(actions, Mapping):
        value = actions.get(action)
        if value:
            return str(value)
    return ACTION_NAMES.get(action, str(action))


def finite_or_blank(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric == float("inf"):
        return ""
    return round(numeric, 6)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
