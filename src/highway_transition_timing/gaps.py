"""Paired timing-gap computation and summaries."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from statistics import mean
from typing import Any

from .config import AnalysisConfig
from .constants import (
    AGENTS,
    CENSORED_PAIR,
    NO_ONSET_CENSORED,
    PRIMARY_TARGET,
    TERMINAL_FAILURE,
    TERMINAL_FAILURE_PAIR,
    VALID_ONSET,
    VALID_PAIR,
)
from .utils import get_float, get_int, get_str, median, quantile


def compute_timing_gaps(
    outcome_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]] | None = None,
    agent_pairs: Sequence[tuple[str, str]] | None = None,
    analysis_target: str = PRIMARY_TARGET,
) -> list[dict[str, Any]]:
    """Compute paired timing gaps within matched exposure/rollout blocks."""

    pairs = tuple(agent_pairs or (("FD", "BAL"), ("FD", "SP"), ("BAL", "SP")))
    exposure_lookup = {
        get_str(row, "exposure_id"): row for row in (exposure_rows or [])
    }

    grouped: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in outcome_rows:
        if get_str(row, "analysis_target", analysis_target) != analysis_target:
            continue
        key = (
            get_str(row, "exposure_seed"),
            get_str(row, "exposure_id"),
            get_str(row, "rollout_id", "r0"),
            get_str(row, "rollout_seed", "0"),
        )
        grouped[key][get_str(row, "agent_condition")] = row

    gap_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        exposure_seed, exposure_id, rollout_id, rollout_seed = key
        by_agent = grouped[key]
        exposure = exposure_lookup.get(exposure_id, {})
        for agent_a, agent_b in pairs:
            row_a = by_agent.get(agent_a)
            row_b = by_agent.get(agent_b)
            if row_a is None or row_b is None:
                continue

            outcome_a = get_str(row_a, "episode_outcome")
            outcome_b = get_str(row_b, "episode_outcome")
            valid_pair = outcome_a == VALID_ONSET and outcome_b == VALID_ONSET
            if valid_pair:
                latency_a = get_float(row_a, "response_latency", 0.0)
                latency_b = get_float(row_b, "response_latency", 0.0)
                gap = latency_b - latency_a
                status = VALID_PAIR
            elif TERMINAL_FAILURE in {outcome_a, outcome_b}:
                latency_a = _maybe_latency(row_a)
                latency_b = _maybe_latency(row_b)
                gap = ""
                status = TERMINAL_FAILURE_PAIR
            else:
                latency_a = _maybe_latency(row_a)
                latency_b = _maybe_latency(row_b)
                gap = ""
                status = CENSORED_PAIR

            gap_rows.append(
                {
                    "gap_id": f"{analysis_target}:{exposure_id}:{rollout_id}:{rollout_seed}:{agent_b}-{agent_a}",
                    "policy_id_a": get_str(row_a, "policy_id"),
                    "policy_id_b": get_str(row_b, "policy_id"),
                    "exposure_seed": exposure_seed,
                    "exposure_id": exposure_id,
                    "rollout_id": rollout_id,
                    "rollout_seed": rollout_seed,
                    "analysis_target": analysis_target,
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "latency_a": latency_a,
                    "latency_b": latency_b,
                    "gap_b_minus_a": gap,
                    "outcome_a": outcome_a,
                    "outcome_b": outcome_b,
                    "gap_status": status,
                    "exposure_difficulty_bin": get_str(
                        exposure, "exposure_difficulty_bin", infer_difficulty_bin(exposure)
                    ),
                }
            )

    return gap_rows


def summarize_gaps(
    gap_rows: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Summarize valid paired timing gaps by agent pair."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in gap_rows:
        key = (
            get_str(row, "analysis_target", PRIMARY_TARGET),
            get_str(row, "agent_a"),
            get_str(row, "agent_b"),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        analysis_target, agent_a, agent_b = key
        rows = grouped[key]
        valid_gaps = [
            get_float(row, "gap_b_minus_a", 0.0)
            for row in rows
            if get_str(row, "gap_status") == VALID_PAIR
        ]
        statuses = Counter(get_str(row, "gap_status") for row in rows)
        med = median(valid_gaps)
        q1 = quantile(valid_gaps, 0.25)
        q3 = quantile(valid_gaps, 0.75)
        ci_low, ci_high = bootstrap_ci(valid_gaps, config)
        summaries.append(
            {
                "analysis_target": analysis_target,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "gap_label": f"{agent_b} - {agent_a}",
                "n_pairs": len(rows),
                "n_valid_pairs": statuses[VALID_PAIR],
                "n_censored_pairs": statuses[CENSORED_PAIR],
                "n_terminal_failure_pairs": statuses[TERMINAL_FAILURE_PAIR],
                "median_gap": med if med is not None else "",
                "mean_gap": round(mean(valid_gaps), 6) if valid_gaps else "",
                "q1_gap": q1 if q1 is not None else "",
                "q3_gap": q3 if q3 is not None else "",
                "iqr_gap": (q3 - q1) if q1 is not None and q3 is not None else "",
                "bootstrap_median_ci_low": ci_low if ci_low is not None else "",
                "bootstrap_median_ci_high": ci_high if ci_high is not None else "",
            }
        )

    return summaries


def summarize_outcomes(
    outcome_rows: Sequence[Mapping[str, Any]],
    analysis_target: str = PRIMARY_TARGET,
) -> list[dict[str, Any]]:
    """Count rollout outcomes by agent condition."""

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in outcome_rows:
        if get_str(row, "analysis_target", analysis_target) != analysis_target:
            continue
        grouped[get_str(row, "agent_condition")][get_str(row, "episode_outcome")] += 1

    rows: list[dict[str, Any]] = []
    for agent in sorted(grouped):
        counts = grouped[agent]
        total = sum(counts.values())
        rows.append(
            {
                "analysis_target": analysis_target,
                "agent_condition": agent,
                "n": total,
                "valid_onset": counts[VALID_ONSET],
                "no_onset_censored": counts[NO_ONSET_CENSORED],
                "terminal_failure": counts[TERMINAL_FAILURE],
                "valid_onset_rate": round(counts[VALID_ONSET] / total, 6) if total else "",
                "no_onset_rate": round(counts[NO_ONSET_CENSORED] / total, 6)
                if total
                else "",
                "terminal_failure_rate": round(counts[TERMINAL_FAILURE] / total, 6)
                if total
                else "",
            }
        )
    return rows


def latency_rank_counts(
    outcome_rows: Sequence[Mapping[str, Any]],
    analysis_target: str = PRIMARY_TARGET,
) -> list[dict[str, Any]]:
    """Count which agent transitions first/second/last within valid exposure blocks."""

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcome_rows:
        if get_str(row, "analysis_target", analysis_target) != analysis_target:
            continue
        if get_str(row, "episode_outcome") != VALID_ONSET:
            continue
        key = (
            get_str(row, "exposure_seed"),
            get_str(row, "exposure_id"),
            get_str(row, "rollout_id", "r0"),
            get_str(row, "rollout_seed", "0"),
        )
        grouped[key].append(row)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rows in grouped.values():
        present_agents = {get_str(row, "agent_condition") for row in rows}
        if not set(AGENTS).issubset(present_agents):
            continue
        ranked = sorted(
            (row for row in rows if get_str(row, "agent_condition") in AGENTS),
            key=lambda row: (
                get_float(row, "response_latency", 0.0),
                get_str(row, "agent_condition"),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            counts[get_str(row, "agent_condition")][f"rank_{rank}"] += 1

    return [
        {
            "analysis_target": analysis_target,
            "agent_condition": agent,
            "rank_1": counts[agent]["rank_1"],
            "rank_2": counts[agent]["rank_2"],
            "rank_3": counts[agent]["rank_3"],
        }
        for agent in sorted(counts)
    ]


def bootstrap_ci(values: list[float], config: AnalysisConfig) -> tuple[float | None, float | None]:
    if not values or config.bootstrap_samples <= 0:
        return None, None
    rng = random.Random(config.random_seed)
    samples: list[float] = []
    for _ in range(config.bootstrap_samples):
        draw = [rng.choice(values) for _ in values]
        sample_median = median(draw)
        if sample_median is not None:
            samples.append(sample_median)
    low = quantile(samples, 0.025)
    high = quantile(samples, 0.975)
    return low, high


def infer_difficulty_bin(exposure: Mapping[str, Any]) -> str:
    if not exposure:
        return ""
    front_distance = get_float(exposure, "nearest_front_distance_at_exposure", 0.0)
    closing = get_float(exposure, "relative_closing_speed_at_exposure", 0.0)
    density = get_float(exposure, "vehicles_density", 0.0)

    if front_distance <= 18:
        front_bin = "close"
    elif front_distance <= 32:
        front_bin = "medium"
    else:
        front_bin = "far"

    if closing <= 1.5:
        closing_bin = "low_closing"
    elif closing <= 5.0:
        closing_bin = "medium_closing"
    else:
        closing_bin = "high_closing"

    if density <= 0.75:
        density_bin = "sparse"
    elif density <= 1.25:
        density_bin = "medium_density"
    else:
        density_bin = "dense"

    return f"{front_bin}_{closing_bin}_{density_bin}"


def _maybe_latency(row: Mapping[str, Any]) -> float | str:
    value = get_str(row, "response_latency")
    if value == "":
        return ""
    return get_float(row, "response_latency", 0.0)
