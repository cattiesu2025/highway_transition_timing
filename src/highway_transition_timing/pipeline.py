"""End-to-end analysis pipeline for matched-exposure rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import AnalysisConfig
from .constants import SLOWDOWN_ONSET_TARGET
from .gaps import (
    compute_timing_gaps,
    latency_rank_counts,
    summarize_gaps,
    summarize_outcomes,
)
from .rewards import reward_config_table
from .transitions import classify_episode_outcomes


@dataclass
class PipelineResult:
    steps: list[dict[str, Any]]
    exposures: list[dict[str, Any]]
    episode_outcomes: list[dict[str, Any]]
    timing_gaps: list[dict[str, Any]]
    gap_summary: list[dict[str, Any]]
    outcome_summary: list[dict[str, Any]]
    latency_rank_counts: list[dict[str, Any]]
    reward_config: list[dict[str, Any]]

    def tables(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "steps": self.steps,
            "exposures": self.exposures,
            "episode_outcomes": self.episode_outcomes,
            "timing_gaps": self.timing_gaps,
            "gap_summary": self.gap_summary,
            "outcome_summary": self.outcome_summary,
            "latency_rank_counts": self.latency_rank_counts,
            "reward_config": self.reward_config,
        }


def run_pipeline(
    step_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]] | None = None,
    config: AnalysisConfig | None = None,
    analysis_target: str = SLOWDOWN_ONSET_TARGET,
    reward_config: Sequence[Mapping[str, Any]] | None = None,
) -> PipelineResult:
    """Run the action-onset timing analysis over rollout rows."""

    cfg = config or AnalysisConfig()
    cfg.validate()
    steps = [dict(row) for row in step_rows]
    exposures = [dict(row) for row in (exposure_rows or [])]

    outcomes = classify_episode_outcomes(steps, cfg, analysis_target)
    gaps = compute_timing_gaps(outcomes, exposures, analysis_target=analysis_target)
    gap_summary = summarize_gaps(gaps, cfg)
    outcome_summary = summarize_outcomes(outcomes, analysis_target)
    ranks = latency_rank_counts(outcomes, analysis_target)
    reward_config_rows = (
        [dict(row) for row in reward_config]
        if reward_config is not None
        else reward_config_table()
    )

    return PipelineResult(
        steps=steps,
        exposures=exposures,
        episode_outcomes=outcomes,
        timing_gaps=gaps,
        gap_summary=gap_summary,
        outcome_summary=outcome_summary,
        latency_rank_counts=ranks,
        reward_config=reward_config_rows,
    )
