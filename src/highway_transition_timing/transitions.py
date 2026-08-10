"""Persistence-confirmed action-onset extraction and outcome classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import AnalysisConfig
from .constants import (
    LANE_CHANGE_ACTIONS,
    LANE_CHANGE_ONSET_TARGET,
    LANE_CHANGE_TARGET_LABEL,
    NO_ONSET_CENSORED,
    SLOW_ACTIONS,
    SLOWDOWN_ONSET_TARGET,
    SLOWDOWN_TARGET_LABEL,
    TERMINAL_FAILURE,
    VALID_ONSET,
)
from .utils import get_bool, get_float, get_int, get_str


def classify_episode_outcomes(
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
    analysis_target: str,
) -> list[dict[str, Any]]:
    """Classify rollout-level target outcomes without dropping censored/failures."""

    steps_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        steps_by_episode.setdefault(get_str(row, "episode_id"), []).append(row)

    outcomes: list[dict[str, Any]] = []
    for episode_id in sorted(steps_by_episode):
        steps = sorted(steps_by_episode[episode_id], key=lambda row: get_int(row, "t", 0))
        first_step = steps[0]
        exposure_t = _episode_exposure_t(steps, config)
        horizon = _episode_horizon(steps, config)
        collision_t = _first_collision_t(steps)
        if analysis_target == LANE_CHANGE_ONSET_TARGET:
            target_transition = select_first_confirmed_lane_change_onset(
                steps,
                config=config,
                exposure_t=exposure_t,
            )
        elif analysis_target == SLOWDOWN_ONSET_TARGET:
            target_transition = select_first_stable_slowdown_onset(
                steps,
                config=config,
                exposure_t=exposure_t,
            )
        else:
            raise ValueError(f"Unsupported analysis target: {analysis_target}")

        if target_transition and (
            collision_t is None or get_int(target_transition, "onset_t", 0) < collision_t
        ):
            episode_outcome = VALID_ONSET
        elif collision_t is not None:
            episode_outcome = TERMINAL_FAILURE
        else:
            episode_outcome = NO_ONSET_CENSORED

        outcomes.append(
            {
                "episode_id": episode_id,
                "agent_condition": get_str(first_step, "agent_condition"),
                "policy_id": get_str(first_step, "policy_id"),
                "exposure_seed": get_str(first_step, "exposure_seed"),
                "exposure_id": get_str(first_step, "exposure_id"),
                "rollout_id": get_str(first_step, "rollout_id", "r0"),
                "rollout_seed": get_str(first_step, "rollout_seed", "0"),
                "analysis_target": analysis_target,
                "episode_outcome": episode_outcome,
                "termination_t": collision_t if collision_t is not None else horizon,
                "termination_reason": _termination_reason(steps, episode_outcome),
                "collision_flag": collision_t is not None,
                "no_onset_censored": episode_outcome == NO_ONSET_CENSORED,
                "target_transition_id": get_str(target_transition or {}, "transition_id"),
                "target_record_type": get_str(target_transition or {}, "record_type"),
                "target_mode_before": get_str(target_transition or {}, "mode_before"),
                "target_mode_after": get_str(target_transition or {}, "mode_after"),
                "target_onset_t": get_int(target_transition or {}, "onset_t", -1)
                if target_transition
                else "",
                "target_confirmation_t": get_int(
                    target_transition or {}, "confirmation_t", -1
                )
                if target_transition
                else "",
                "response_latency": get_int(target_transition or {}, "response_latency", 0)
                if episode_outcome == VALID_ONSET and target_transition
                else "",
                "censoring_time": horizon - exposure_t
                if episode_outcome == NO_ONSET_CENSORED
                else "",
            }
        )

    return outcomes


def select_first_confirmed_lane_change_onset(
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
    exposure_t: int = 0,
) -> dict[str, Any] | None:
    """Select the lane-change onset confirmed by a realised lane-index change.

    A lane change is one meta-action followed by roughly two seconds of lateral
    motion, so command persistence cannot confirm it. Confirmation is instead
    the first realised physical lane-index change after exposure, and onset is
    the earliest lane-change command inside the confirmation window that
    precedes it. Episodes that never complete a lane change have no onset,
    however many lane-change commands they issued.
    """

    if not step_rows:
        return None

    ordered_steps = sorted(step_rows, key=lambda row: get_int(row, "t", 0))
    episode_id = get_str(ordered_steps[0], "episode_id")
    post_exposure_steps = [
        row for row in ordered_steps if get_int(row, "t", 0) >= exposure_t
    ]
    if not post_exposure_steps:
        return None

    initial_lane = get_int(post_exposure_steps[0], "ego_lane", 0)
    confirmation_t: int | None = None
    for row in post_exposure_steps:
        pre_lane = get_int(row, "ego_lane", initial_lane)
        post_lane = get_int(row, "post_ego_lane", pre_lane)
        if pre_lane != initial_lane or post_lane != initial_lane:
            confirmation_t = get_int(row, "t", 0)
            break
    if confirmation_t is None:
        return None

    window_start = max(exposure_t, confirmation_t - config.lane_change_confirmation_window)
    command_times = [
        get_int(row, "t", 0)
        for row in post_exposure_steps
        if get_str(row, "action").upper() in LANE_CHANGE_ACTIONS
        and window_start <= get_int(row, "t", 0) <= confirmation_t
    ]
    onset_t = min(command_times) if command_times else confirmation_t
    record_type = (
        "confirmed_lane_change_onset"
        if command_times
        else "confirmed_lane_change_onset_without_command"
    )

    return {
        "transition_id": f"{episode_id}:{LANE_CHANGE_TARGET_LABEL}:onset000",
        "record_type": record_type,
        "episode_id": episode_id,
        "mode_before": "",
        "mode_after": LANE_CHANGE_TARGET_LABEL,
        "onset_t": onset_t,
        "confirmation_t": confirmation_t,
        "transition_interval": [onset_t, confirmation_t],
        "persistence_length": confirmation_t - onset_t + 1,
        "transition_confidence": 1.0,
        "local_context_before": {},
        "local_context_after": _context_at_or_before(ordered_steps, onset_t),
        "response_latency": onset_t - exposure_t,
    }


def select_first_stable_slowdown_onset(
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
    exposure_t: int = 0,
) -> dict[str, Any] | None:
    """Select first stable slowdown evidence after exposure."""

    if not step_rows:
        return None

    ordered_steps = sorted(step_rows, key=lambda row: get_int(row, "t", 0))
    episode_id = get_str(ordered_steps[0], "episode_id")
    post_exposure_steps = [
        row for row in ordered_steps if get_int(row, "t", 0) >= exposure_t
    ]
    evidence_segments = _slowdown_evidence_segments(post_exposure_steps)

    for segment in evidence_segments:
        onset_t = max(int(segment["start_t"]), exposure_t)
        confirmation_t = onset_t + config.persistence_k - 1
        if confirmation_t > segment["end_t"]:
            continue
        return {
            "transition_id": f"{episode_id}:{SLOWDOWN_TARGET_LABEL}:onset000",
            "record_type": "stable_action_onset",
            "episode_id": episode_id,
            "mode_before": "",
            "mode_after": SLOWDOWN_TARGET_LABEL,
            "onset_t": onset_t,
            "confirmation_t": confirmation_t,
            "transition_interval": [onset_t, confirmation_t],
            "persistence_length": segment["length"],
            "transition_confidence": 1.0,
            "local_context_before": {},
            "local_context_after": _context_at_or_before(ordered_steps, onset_t),
            "response_latency": onset_t - exposure_t,
        }
    return None


def _slowdown_evidence_segments(
    step_rows: Sequence[Mapping[str, Any]],
    speed_drop_threshold: float = -0.35,
) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    previous_speed: float | None = None

    for row in step_rows:
        t = get_int(row, "t", 0)
        action = get_str(row, "action").upper()
        speed = get_float(row, "ego_speed", 0.0)
        speed_delta = 0.0 if previous_speed is None else speed - previous_speed
        has_evidence = action in SLOW_ACTIONS or speed_delta <= speed_drop_threshold
        previous_speed = speed

        if has_evidence:
            if current is None or t != current["end_t"] + 1:
                if current is not None:
                    segments.append(current)
                current = {"start_t": t, "end_t": t, "length": 1}
            else:
                current["end_t"] = t
                current["length"] += 1
        elif current is not None:
            segments.append(current)
            current = None

    if current is not None:
        segments.append(current)
    return segments


def _episode_exposure_t(
    step_rows: Sequence[Mapping[str, Any]], config: AnalysisConfig
) -> int:
    if not step_rows:
        return config.exposure_t_default
    return get_int(step_rows[0], "exposure_t", config.exposure_t_default)


def _episode_horizon(
    step_rows: Sequence[Mapping[str, Any]], config: AnalysisConfig
) -> int:
    if config.response_horizon is not None:
        return config.response_horizon
    if not step_rows:
        return config.exposure_t_default
    return max(get_int(row, "t", 0) for row in step_rows)


def _first_collision_t(step_rows: Sequence[Mapping[str, Any]]) -> int | None:
    collision_times = [
        get_int(row, "t", 0) for row in step_rows if get_bool(row, "collision_flag", False)
    ]
    return min(collision_times) if collision_times else None


def _termination_reason(
    step_rows: Sequence[Mapping[str, Any]], episode_outcome: str
) -> str:
    for row in sorted(step_rows, key=lambda item: get_int(item, "t", 0), reverse=True):
        reason = get_str(row, "termination_reason")
        if reason:
            return reason
    if episode_outcome == TERMINAL_FAILURE:
        return "collision_before_target"
    if episode_outcome == NO_ONSET_CENSORED:
        return "duration_before_target"
    return "target_observed"


def _context_at_or_before(
    step_rows: Sequence[Mapping[str, Any]], t: int
) -> dict[str, Any]:
    eligible = [
        row for row in step_rows if get_int(row, "t", 0) <= t
    ]
    if not eligible:
        return {}
    row = max(eligible, key=lambda item: get_int(item, "t", 0))
    keys = (
        "t",
        "ego_lane",
        "ego_speed",
        "action",
        "nearest_front_distance",
        "closest_k_distances",
        "speed_score",
        "front_distance_score",
        "closest_k_distance_score",
        "collision_risk_score",
        "lane_position_score",
    )
    return {key: row[key] for key in keys if key in row}
