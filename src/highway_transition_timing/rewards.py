"""Reward-preference condition definitions for future Highway training."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Shared reward-component weights for one agent condition."""

    speed_score: float
    front_distance_score: float
    collision_penalty: float
    lane_change_penalty: float
    slow_down_penalty: float = 0.0
    right_lane_score: float = 0.0
    collision_risk_penalty: float = 0.0


MAIN_REWARD_WEIGHTS: dict[str, RewardWeights] = {
    "FD": RewardWeights(
        speed_score=0.45,
        front_distance_score=1.00,
        collision_penalty=2.00,
        lane_change_penalty=0.10,
        right_lane_score=0.0,
    ),
    "BAL": RewardWeights(
        speed_score=0.70,
        front_distance_score=0.70,
        collision_penalty=2.00,
        lane_change_penalty=0.10,
        right_lane_score=0.0,
    ),
    "SP": RewardWeights(
        speed_score=1.00,
        front_distance_score=0.25,
        collision_penalty=2.00,
        lane_change_penalty=0.10,
        right_lane_score=0.0,
    ),
}

REWARD_STRENGTH_RULES = {
    "FD": "front_distance_score_only",
    "BAL": "all_effective_reward_weights",
    "SP": "speed_score_only",
}


def reward_config_table() -> list[dict[str, float | str]]:
    """Return reward weights in table form for reports."""

    return reward_config_table_with_slow_down_penalty(0.0, 0.0)


def reward_config_table_with_slow_down_penalty(
    common_slow_down_penalty: float,
    common_collision_risk_penalty: float = 0.0,
    common_collision_penalty: float | None = None,
    common_lane_change_penalty: float | None = None,
) -> list[dict[str, float | str]]:
    """Return reward weights with optional common action/risk penalties."""

    rows: list[dict[str, float | str]] = []
    for agent, weights in MAIN_REWARD_WEIGHTS.items():
        effective_weights = with_common_slow_down_penalty(
            weights,
            common_slow_down_penalty,
            common_collision_risk_penalty,
            common_collision_penalty,
            common_lane_change_penalty,
        )
        rows.append(
            {
                "agent_condition": agent,
                "speed_score": effective_weights.speed_score,
                "front_distance_score": effective_weights.front_distance_score,
                "collision_penalty": effective_weights.collision_penalty,
                "collision_risk_penalty": effective_weights.collision_risk_penalty,
                "lane_change_penalty": effective_weights.lane_change_penalty,
                "slow_down_penalty": effective_weights.slow_down_penalty,
                "right_lane_score": effective_weights.right_lane_score,
            }
        )
    return rows


def reward_config_table_with_strength_multiplier(
    common_slow_down_penalty: float,
    common_collision_risk_penalty: float,
    reward_strength_multiplier: float,
    common_collision_penalty: float | None = None,
    common_lane_change_penalty: float | None = None,
) -> list[dict[str, float | str]]:
    """Return effective reward weights for the reward-strength validation."""

    rows: list[dict[str, float | str]] = []
    for agent, weights in MAIN_REWARD_WEIGHTS.items():
        effective_weights = with_common_slow_down_penalty(
            weights,
            common_slow_down_penalty,
            common_collision_risk_penalty,
            common_collision_penalty,
            common_lane_change_penalty,
        )
        effective_weights = with_reward_strength_multiplier(
            agent,
            effective_weights,
            reward_strength_multiplier,
        )
        rows.append(
            {
                "agent_condition": agent,
                "speed_score": effective_weights.speed_score,
                "front_distance_score": effective_weights.front_distance_score,
                "collision_penalty": effective_weights.collision_penalty,
                "collision_risk_penalty": effective_weights.collision_risk_penalty,
                "lane_change_penalty": effective_weights.lane_change_penalty,
                "slow_down_penalty": effective_weights.slow_down_penalty,
                "right_lane_score": effective_weights.right_lane_score,
                "reward_strength_multiplier": reward_strength_multiplier,
                "reward_strength_rule": REWARD_STRENGTH_RULES[agent],
            }
        )
    return rows


def with_reward_strength_multiplier(
    agent_condition: str,
    weights: RewardWeights,
    multiplier: float,
) -> RewardWeights:
    """Apply the predeclared agent-specific reward-strength intervention."""

    if agent_condition not in REWARD_STRENGTH_RULES:
        raise ValueError(f"unknown agent condition: {agent_condition}")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("reward strength multiplier must be positive and finite")
    if agent_condition == "FD":
        return RewardWeights(
            speed_score=weights.speed_score,
            front_distance_score=weights.front_distance_score * multiplier,
            collision_penalty=weights.collision_penalty,
            lane_change_penalty=weights.lane_change_penalty,
            slow_down_penalty=weights.slow_down_penalty,
            right_lane_score=weights.right_lane_score,
            collision_risk_penalty=weights.collision_risk_penalty,
        )
    if agent_condition == "SP":
        return RewardWeights(
            speed_score=weights.speed_score * multiplier,
            front_distance_score=weights.front_distance_score,
            collision_penalty=weights.collision_penalty,
            lane_change_penalty=weights.lane_change_penalty,
            slow_down_penalty=weights.slow_down_penalty,
            right_lane_score=weights.right_lane_score,
            collision_risk_penalty=weights.collision_risk_penalty,
        )
    return RewardWeights(
        speed_score=weights.speed_score * multiplier,
        front_distance_score=weights.front_distance_score * multiplier,
        collision_penalty=weights.collision_penalty * multiplier,
        lane_change_penalty=weights.lane_change_penalty * multiplier,
        slow_down_penalty=weights.slow_down_penalty * multiplier,
        right_lane_score=weights.right_lane_score * multiplier,
        collision_risk_penalty=weights.collision_risk_penalty * multiplier,
    )


def with_common_slow_down_penalty(
    weights: RewardWeights,
    common_slow_down_penalty: float,
    common_collision_risk_penalty: float = 0.0,
    common_collision_penalty: float | None = None,
    common_lane_change_penalty: float | None = None,
) -> RewardWeights:
    """Apply common action/risk costs to every reward condition."""

    return RewardWeights(
        speed_score=weights.speed_score,
        front_distance_score=weights.front_distance_score,
        collision_penalty=(
            weights.collision_penalty
            if common_collision_penalty is None
            else common_collision_penalty
        ),
        lane_change_penalty=(
            weights.lane_change_penalty
            if common_lane_change_penalty is None
            else common_lane_change_penalty
        ),
        slow_down_penalty=common_slow_down_penalty,
        right_lane_score=weights.right_lane_score,
        collision_risk_penalty=common_collision_risk_penalty,
    )
