"""Persistence-confirmed transition extraction and outcome classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import AnalysisConfig
from .constants import (
    FIRST_STABLE_MODE_ONSET_TARGET,
    LANE_CHANGE_ACTIONS,
    LANE_CHANGE_ONSET_TARGET,
    LANE_CHANGE_TARGET_LABEL,
    NO_ONSET_CENSORED,
    PRIMARY_MODES,
    PRIMARY_TARGET,
    SLOW_ACTIONS,
    SLOWDOWN_ONSET_TARGET,
    SLOWDOWN_TARGET_LABEL,
    TERMINAL_FAILURE,
    TRAFFIC_SPACING_ADJUSTMENT,
    VALID_ONSET,
)
from .utils import get_bool, get_float, get_int, get_str, median


def extract_transitions_for_episode(
    mode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Extract accepted mode transitions for one episode."""

    config.validate()
    if not mode_rows:
        return []

    ordered_modes = sorted(mode_rows, key=lambda row: get_int(row, "t", 0))
    episode_id = get_str(ordered_modes[0], "episode_id")
    exposure_t = _episode_exposure_t(step_rows, config)
    outcome = _raw_episode_outcome(step_rows)
    segments = run_length_segments(ordered_modes)

    transitions: list[dict[str, Any]] = []
    previous_stable: dict[str, Any] | None = None
    for segment in segments:
        if segment["mode_label"] not in PRIMARY_MODES:
            continue
        if segment["length"] < config.persistence_k:
            continue
        if previous_stable is None:
            previous_stable = segment
            continue
        if segment["mode_label"] == previous_stable["mode_label"]:
            previous_stable = segment
            continue

        onset_t = int(segment["start_t"])
        confirmation_t = onset_t + config.persistence_k - 1
        transition_id = f"{episode_id}:tr{len(transitions):03d}"
        context_before = _context_at_or_before(step_rows, previous_stable["end_t"])
        context_after = _context_at_or_before(step_rows, onset_t)
        confidence_values = [
            get_float(row, "mode_confidence", 0.0)
            for row in ordered_modes
            if onset_t <= get_int(row, "t", 0) <= confirmation_t
        ]

        transitions.append(
            {
                "transition_id": transition_id,
                "episode_id": episode_id,
                "agent_condition": get_str(segment, "agent_condition"),
                "policy_id": get_str(segment, "policy_id"),
                "exposure_seed": get_str(segment, "exposure_seed"),
                "exposure_id": get_str(segment, "exposure_id"),
                "rollout_id": get_str(segment, "rollout_id", "r0"),
                "rollout_seed": get_str(segment, "rollout_seed", "0"),
                "mode_before": previous_stable["mode_label"],
                "mode_after": segment["mode_label"],
                "onset_t": onset_t,
                "confirmation_t": confirmation_t,
                "transition_interval": [onset_t, confirmation_t],
                "persistence_length": segment["length"],
                "transition_confidence": round(float(median(confidence_values) or 0.0), 6),
                "local_context_before": context_before,
                "local_context_after": context_after,
                "episode_outcome": outcome,
                "response_latency": onset_t - exposure_t,
            }
        )
        previous_stable = segment

    return transitions


def extract_transitions(
    mode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Extract accepted transitions for all episodes."""

    modes_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in mode_rows:
        modes_by_episode.setdefault(get_str(row, "episode_id"), []).append(row)

    steps_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        steps_by_episode.setdefault(get_str(row, "episode_id"), []).append(row)

    transitions: list[dict[str, Any]] = []
    for episode_id in sorted(modes_by_episode):
        transitions.extend(
            extract_transitions_for_episode(
                modes_by_episode[episode_id],
                steps_by_episode.get(episode_id, []),
                config,
            )
        )
    return transitions


def run_length_segments(mode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress smoothed mode labels into contiguous segments."""

    ordered = sorted(mode_rows, key=lambda row: get_int(row, "t", 0))
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in ordered:
        label = get_str(row, "mode_label")
        if not label:
            if current is not None:
                segments.append(current)
                current = None
            continue

        t = get_int(row, "t", 0)
        if current is None or current["mode_label"] != label or t != current["end_t"] + 1:
            if current is not None:
                segments.append(current)
            current = _new_segment(row)
        else:
            current["end_t"] = t
            current["length"] += 1
            current["rows"].append(row)

    if current is not None:
        segments.append(current)
    return segments


def classify_episode_outcomes(
    step_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
    analysis_target: str = PRIMARY_TARGET,
    mode_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Classify rollout-level target outcomes without dropping censored/failures."""

    steps_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        steps_by_episode.setdefault(get_str(row, "episode_id"), []).append(row)

    transitions_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in transition_rows:
        transitions_by_episode.setdefault(get_str(row, "episode_id"), []).append(row)

    modes_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in mode_rows or []:
        modes_by_episode.setdefault(get_str(row, "episode_id"), []).append(row)

    outcomes: list[dict[str, Any]] = []
    for episode_id in sorted(steps_by_episode):
        steps = sorted(steps_by_episode[episode_id], key=lambda row: get_int(row, "t", 0))
        transitions = sorted(
            transitions_by_episode.get(episode_id, []),
            key=lambda row: get_int(row, "onset_t", 0),
        )
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
        elif analysis_target == FIRST_STABLE_MODE_ONSET_TARGET or analysis_target.startswith(
            "mode_onset:"
        ):
            target_mode = (
                analysis_target.split(":", 1)[1]
                if analysis_target.startswith("mode_onset:")
                else None
            )
            target_transition = select_first_stable_mode_onset(
                modes_by_episode.get(episode_id, []),
                step_rows=steps,
                config=config,
                exposure_t=exposure_t,
                target_mode=target_mode,
            )
        else:
            target_transition = select_target_transition(transitions, analysis_target, exposure_t)

        if target_transition and (
            collision_t is None or get_int(target_transition, "onset_t", 0) < collision_t
        ):
            episode_outcome = VALID_ONSET
        elif collision_t is not None:
            episode_outcome = TERMINAL_FAILURE
        else:
            episode_outcome = NO_ONSET_CENSORED

        spacing_transition = select_target_transition(
            transitions, f"mode:{TRAFFIC_SPACING_ADJUSTMENT}", exposure_t
        )
        first_collision_before_spacing = (
            collision_t is not None
            and (
                spacing_transition is None
                or collision_t < get_int(spacing_transition, "onset_t", 0)
            )
        )

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
                "valid_transition_count": len(transitions),
                "no_onset_modes": episode_outcome == NO_ONSET_CENSORED,
                "first_collision_before_spacing_adjustment": first_collision_before_spacing,
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


def select_first_stable_mode_onset(
    mode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
    exposure_t: int = 0,
    target_mode: str | None = None,
) -> dict[str, Any] | None:
    """Select first stable mode segment onset, including onset at exposure_t."""

    if not mode_rows:
        return None
    ordered_modes = sorted(mode_rows, key=lambda row: get_int(row, "t", 0))
    episode_id = get_str(ordered_modes[0], "episode_id")
    for segment in run_length_segments(ordered_modes):
        if segment["mode_label"] not in PRIMARY_MODES:
            continue
        if target_mode is not None and segment["mode_label"] != target_mode:
            continue
        if segment["length"] < config.persistence_k:
            continue
        segment_end = int(segment["end_t"])
        if segment_end < exposure_t:
            continue
        onset_t = max(int(segment["start_t"]), exposure_t)
        confirmation_t = onset_t + config.persistence_k - 1
        if confirmation_t > segment_end:
            continue
        confidence_values = [
            get_float(row, "mode_confidence", 0.0)
            for row in ordered_modes
            if onset_t <= get_int(row, "t", 0) <= confirmation_t
        ]
        return {
            "transition_id": f"{episode_id}:onset000",
            "record_type": "stable_mode_onset",
            "episode_id": episode_id,
            "mode_before": "",
            "mode_after": segment["mode_label"],
            "onset_t": onset_t,
            "confirmation_t": confirmation_t,
            "transition_interval": [onset_t, confirmation_t],
            "persistence_length": segment["length"],
            "transition_confidence": round(float(median(confidence_values) or 0.0), 6),
            "local_context_before": {},
            "local_context_after": _context_at_or_before(step_rows, onset_t),
            "response_latency": onset_t - exposure_t,
        }
    return None


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


def select_target_transition(
    transitions: Sequence[Mapping[str, Any]],
    analysis_target: str,
    exposure_t: int = 0,
) -> Mapping[str, Any] | None:
    """Select the transition corresponding to an analysis target."""

    candidates = [
        row for row in transitions if get_int(row, "onset_t", -1) > exposure_t
    ]
    if analysis_target == PRIMARY_TARGET:
        return candidates[0] if candidates else None

    if analysis_target.startswith("mode:"):
        target_mode = analysis_target.split(":", 1)[1]
        for row in candidates:
            if get_str(row, "mode_after") == target_mode:
                return row
        return None

    if analysis_target.startswith("transition:"):
        spec = analysis_target.split(":", 1)[1]
        if "->" not in spec:
            raise ValueError(f"Invalid transition target: {analysis_target}")
        before, after = [part.strip() for part in spec.split("->", 1)]
        for row in candidates:
            if get_str(row, "mode_before") == before and get_str(row, "mode_after") == after:
                return row
        return None

    raise ValueError(f"Unsupported analysis target: {analysis_target}")


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


def _new_segment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": get_str(row, "episode_id"),
        "agent_condition": get_str(row, "agent_condition"),
        "policy_id": get_str(row, "policy_id"),
        "exposure_seed": get_str(row, "exposure_seed"),
        "exposure_id": get_str(row, "exposure_id"),
        "rollout_id": get_str(row, "rollout_id", "r0"),
        "rollout_seed": get_str(row, "rollout_seed", "0"),
        "mode_label": get_str(row, "mode_label"),
        "start_t": get_int(row, "t", 0),
        "end_t": get_int(row, "t", 0),
        "length": 1,
        "rows": [row],
    }


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


def _raw_episode_outcome(step_rows: Sequence[Mapping[str, Any]]) -> str:
    for row in step_rows:
        value = get_str(row, "episode_outcome")
        if value:
            return value
    return "collision" if _first_collision_t(step_rows) is not None else "completed"


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
