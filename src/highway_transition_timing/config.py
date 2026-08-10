"""Configuration for action-onset extraction and timing analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable settings for the matched-exposure transition-timing analysis.

    The defaults use a persistence window of ``k = 3``: a slowdown onset counts
    only once the slowdown evidence holds for ``k`` consecutive steps.

    ``lane_change_confirmation_window`` is the maximum number of policy steps
    between a lane-change command and the realised physical lane-index change
    that confirms it. A lane change is a single meta-action followed by roughly
    two seconds of lateral motion, so the ``persistence_k`` rule does not apply
    to that target; completion is the confirmation instead.
    """

    persistence_k: int = 3
    lane_change_confirmation_window: int = 15
    exposure_t_default: int = 0
    response_horizon: int | None = None
    bootstrap_samples: int = 1000
    random_seed: int = 7

    def validate(self) -> None:
        if self.persistence_k < 1:
            raise ValueError("persistence_k must be >= 1")
        if self.lane_change_confirmation_window < 1:
            raise ValueError("lane_change_confirmation_window must be >= 1")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0")
