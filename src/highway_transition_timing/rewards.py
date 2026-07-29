"""Reward-preference condition definitions for future Highway training."""

from __future__ import annotations

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


def reward_config_table() -> list[dict[str, float | str]]:
    """Return reward weights in table form for reports."""

    return reward_config_table_with_slow_down_penalty(0.0, 0.0)


def reward_config_table_with_slow_down_penalty(
    common_slow_down_penalty: float,
    common_collision_risk_penalty: float = 0.0,
    common_collision_penalty: float | None = None,
) -> list[dict[str, float | str]]:
    """Return reward weights with optional common action/risk penalties."""

    rows: list[dict[str, float | str]] = []
    for agent, weights in MAIN_REWARD_WEIGHTS.items():
        effective_weights = with_common_slow_down_penalty(
            weights,
            common_slow_down_penalty,
            common_collision_risk_penalty,
            common_collision_penalty,
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


def with_common_slow_down_penalty(
    weights: RewardWeights,
    common_slow_down_penalty: float,
    common_collision_risk_penalty: float = 0.0,
    common_collision_penalty: float | None = None,
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
        lane_change_penalty=weights.lane_change_penalty,
        slow_down_penalty=common_slow_down_penalty,
        right_lane_score=weights.right_lane_score,
        collision_risk_penalty=common_collision_risk_penalty,
    )
