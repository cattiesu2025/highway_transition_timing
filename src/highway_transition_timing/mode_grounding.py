"""Rule-based fallback mode grounding.

The plan prefers TCAV/probe-assisted mode evidence when activations are
available. This module implements the planned fallback path over observable
state, action, and diagnostic-score features so the transition-timing analysis
can run on ordinary rollout tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import AnalysisConfig
from .constants import (
    AMBIGUOUS,
    FAST_ACTIONS,
    HIGH_SPEED_CRUISE,
    IDLE_ACTIONS,
    LANE_CHANGE_ACTIONS,
    LANE_KEEPING_CRUISE,
    PRIMARY_MODES,
    SLOW_ACTIONS,
    TRAFFIC_SPACING_ADJUSTMENT,
)
from .utils import get_bool, get_float, get_int, get_str, json_dumps, parse_number_list


def classify_raw_intent(
    row: Mapping[str, Any],
    previous_row: Mapping[str, Any] | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Assign a one-step raw behavioural evidence label.

    The returned label is intentionally named ``raw_intent_label`` in output
    tables to avoid overclaiming true hidden intent.
    """

    action = get_str(row, "action", "IDLE").upper()
    ego_speed = get_float(row, "ego_speed", 0.0)
    ego_lane = get_int(row, "ego_lane", 0)
    front_distance = get_float(row, "nearest_front_distance", float("inf"))
    speed_score = get_float(row, "speed_score", min(1.0, ego_speed / 35.0 if ego_speed else 0.0))
    front_distance_score = get_float(row, "front_distance_score", min(1.0, front_distance / 60.0))
    closest_k_distances = parse_number_list(row.get("closest_k_distances", ""))
    min_closest_k = min(closest_k_distances) if closest_k_distances else front_distance
    closest_k_score = get_float(
        row,
        "closest_k_distance_score",
        min(1.0, min_closest_k / 50.0) if min_closest_k != float("inf") else 1.0,
    )

    previous_speed = get_float(previous_row or {}, "ego_speed", ego_speed)
    previous_lane = get_int(previous_row or {}, "ego_lane", ego_lane)
    previous_front_distance = get_float(previous_row or {}, "nearest_front_distance", front_distance)

    speed_delta = ego_speed - previous_speed
    lane_delta = ego_lane - previous_lane
    front_distance_delta = front_distance - previous_front_distance

    close_front = (
        front_distance <= config.close_front_distance_m
        or front_distance_score <= config.low_front_distance_score
    )
    very_close_front = front_distance <= config.very_close_front_distance_m
    dense_neighbourhood = closest_k_score <= config.low_closest_k_score

    braking = action in SLOW_ACTIONS or speed_delta < -0.35
    lane_change = action in LANE_CHANGE_ACTIONS or lane_delta != 0
    accelerating = action in FAST_ACTIONS or speed_delta > 0.35
    idle_like = action in IDLE_ACTIONS or action == ""

    spacing_evidence = 0.0
    if close_front:
        spacing_evidence += 0.35
    if very_close_front:
        spacing_evidence += 0.2
    if dense_neighbourhood:
        spacing_evidence += 0.15
    if braking:
        spacing_evidence += 0.25
    if lane_change and (close_front or dense_neighbourhood):
        spacing_evidence += 0.2
    if front_distance_delta > 0.4 and (braking or lane_change):
        spacing_evidence += 0.15
    if front_distance_delta < -0.7 and close_front and accelerating:
        spacing_evidence -= 0.1

    high_speed_evidence = 0.0
    if speed_score >= config.high_speed_score or ego_speed >= config.high_speed_mps:
        high_speed_evidence += 0.45
    if accelerating:
        high_speed_evidence += 0.4
    if not close_front:
        high_speed_evidence += 0.15
    if braking:
        high_speed_evidence -= 0.2

    lane_keep_evidence = 0.25
    if idle_like:
        lane_keep_evidence += 0.25
    if not lane_change:
        lane_keep_evidence += 0.2
    if abs(speed_delta) <= 0.35:
        lane_keep_evidence += 0.15
    if close_front and braking:
        lane_keep_evidence -= 0.2
    if accelerating and speed_score >= config.high_speed_score:
        lane_keep_evidence -= 0.1

    scores = {
        LANE_KEEPING_CRUISE: clamp01(lane_keep_evidence),
        HIGH_SPEED_CRUISE: clamp01(high_speed_evidence),
        TRAFFIC_SPACING_ADJUSTMENT: clamp01(spacing_evidence),
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1]
    margin = best_score - second_score

    if margin < config.min_confidence_margin:
        label = AMBIGUOUS
        confidence = best_score
    else:
        label = best_label
        confidence = best_score

    adjustment_method = ""
    if label == TRAFFIC_SPACING_ADJUSTMENT:
        if braking and lane_change:
            adjustment_method = "mixed"
        elif braking:
            adjustment_method = "brake"
        elif lane_change:
            adjustment_method = "lane_change"
        else:
            adjustment_method = "implicit_spacing"

    features = {
        "action": action,
        "ego_speed": ego_speed,
        "speed_delta": speed_delta,
        "ego_lane": ego_lane,
        "lane_delta": lane_delta,
        "nearest_front_distance": front_distance,
        "front_distance_delta": front_distance_delta,
        "min_closest_k_distance": min_closest_k,
        "speed_score": speed_score,
        "front_distance_score": front_distance_score,
        "closest_k_distance_score": closest_k_score,
        "scores": scores,
        "adjustment_method": adjustment_method,
    }

    return {
        "raw_intent_label": label,
        "raw_intent_confidence": round(confidence, 6),
        "mode_features": features,
    }


def bridge_short_interruptions(
    labels: Sequence[str | None],
    max_gap: int,
    bridgeable_labels: set[str] | None = None,
) -> tuple[list[str | None], list[str]]:
    """Merge short ``A B A`` interruptions after raw timestep labelling."""

    if max_gap <= 0:
        return list(labels), ["unchanged" for _ in labels]

    bridgeable = bridgeable_labels or set(PRIMARY_MODES)
    smoothed = list(labels)
    status = ["unchanged" for _ in labels]
    n = len(smoothed)
    i = 0
    while i < n:
        label = smoothed[i]
        j = i + 1
        while j < n and smoothed[j] == label:
            j += 1

        segment_len = j - i
        left = smoothed[i - 1] if i > 0 else None
        right = smoothed[j] if j < n else None
        can_bridge = (
            segment_len <= max_gap
            and left is not None
            and left == right
            and left in bridgeable
            and label is not None
            and label != left
        )
        if can_bridge:
            for idx in range(i, j):
                smoothed[idx] = left
                status[idx] = f"bridged_gap_len_{segment_len}"
        i = j

    return smoothed, status


def apply_idle_continuation(
    raw_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str | None],
    config: AnalysisConfig,
) -> tuple[list[str | None], list[str]]:
    """Carry ``IDLE`` timesteps forward when they maintain the previous mode.

    ``IDLE`` in Highway means "keep lane/target speed"; it is not necessarily
    lane-keeping cruise. A high-speed policy that switches from ``FASTER`` to
    ``IDLE`` is often still high-speed cruising, and a spacing manoeuvre can be
    followed by ``IDLE`` while the gap is being maintained.
    """

    if not config.idle_continuation_enabled:
        return list(labels), ["unchanged" for _ in labels]

    adjusted = list(labels)
    status = ["unchanged" for _ in labels]
    previous_primary: str | None = None

    for idx, raw_row in enumerate(raw_rows):
        label = adjusted[idx]
        if label is None:
            previous_primary = None
            continue

        raw = raw_row["raw"]
        features = raw.get("mode_features", {})
        action = str(features.get("action", "")).upper()

        if (
            action in IDLE_ACTIONS
            and previous_primary in PRIMARY_MODES
            and label in {LANE_KEEPING_CRUISE, AMBIGUOUS}
            and should_continue_idle_mode(previous_primary, features, config)
        ):
            adjusted[idx] = previous_primary
            status[idx] = f"idle_continuation:{previous_primary}"
            label = previous_primary

        if label in PRIMARY_MODES:
            previous_primary = label

    return adjusted, status


def should_continue_idle_mode(
    previous_label: str,
    features: Mapping[str, Any],
    config: AnalysisConfig,
) -> bool:
    if previous_label == HIGH_SPEED_CRUISE:
        return (
            get_float(features, "speed_score", 0.0) >= config.high_speed_idle_score
            or get_float(features, "ego_speed", 0.0) >= config.high_speed_idle_mps
        )
    if previous_label == TRAFFIC_SPACING_ADJUSTMENT:
        return (
            get_float(features, "nearest_front_distance", float("inf"))
            <= config.close_front_distance_m
            or get_float(features, "front_distance_score", 1.0)
            <= config.low_front_distance_score
            or get_float(features, "closest_k_distance_score", 1.0)
            <= config.low_closest_k_score
        )
    return previous_label == LANE_KEEPING_CRUISE


def build_mode_table_for_episode(
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Build a mode table for one episode's step rows."""

    config.validate()
    ordered = sorted(step_rows, key=lambda row: get_int(row, "t", 0))
    raw_rows: list[dict[str, Any]] = []
    labels: list[str | None] = []
    terminal_seen = False

    previous_row: Mapping[str, Any] | None = None
    for row in ordered:
        collision = get_bool(row, "collision_flag", False)
        terminal_here = terminal_seen or collision
        if terminal_here:
            raw = {
                "raw_intent_label": "",
                "raw_intent_confidence": "",
                "mode_features": {
                    "terminal": True,
                    "collision_flag": collision,
                    "termination_reason": get_str(row, "termination_reason", ""),
                },
            }
            labels.append(None)
            terminal_seen = True
        else:
            raw = classify_raw_intent(row, previous_row, config)
            labels.append(raw["raw_intent_label"])

        raw_rows.append({"step": row, "raw": raw})
        previous_row = row

    continued, continuation_status = apply_idle_continuation(raw_rows, labels, config)
    smoothed, smoothing_status = bridge_short_interruptions(continued, config.bridge_max_gap)
    mode_rows: list[dict[str, Any]] = []

    for raw_row, label, carry_status, bridge_status in zip(
        raw_rows, smoothed, continuation_status, smoothing_status, strict=True
    ):
        step = raw_row["step"]
        raw = raw_row["raw"]
        mode_confidence = raw["raw_intent_confidence"] if label else ""
        if bridge_status != "unchanged":
            status = bridge_status
        elif carry_status != "unchanged":
            status = carry_status
        else:
            status = "unchanged"
        row = {
            "episode_id": get_str(step, "episode_id"),
            "agent_condition": get_str(step, "agent_condition"),
            "policy_id": get_str(step, "policy_id"),
            "exposure_seed": get_str(step, "exposure_seed"),
            "exposure_id": get_str(step, "exposure_id"),
            "rollout_id": get_str(step, "rollout_id", "r0"),
            "rollout_seed": get_str(step, "rollout_seed", "0"),
            "t": get_int(step, "t", 0),
            "mode_label": label or "",
            "mode_confidence": mode_confidence,
            "raw_intent_label": raw["raw_intent_label"],
            "raw_intent_confidence": raw["raw_intent_confidence"],
            "mode_features": raw["mode_features"],
            "smoothing_status": "terminal_or_post_terminal" if label is None else status,
        }
        mode_rows.append(row)

    return mode_rows


def build_mode_table(
    step_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Build mode rows for all episodes."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        grouped.setdefault(get_str(row, "episode_id"), []).append(row)

    mode_rows: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        mode_rows.extend(build_mode_table_for_episode(grouped[episode_id], config))
    return mode_rows


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mode_features_json(row: Mapping[str, Any]) -> str:
    return json_dumps(row.get("mode_features", {}))
