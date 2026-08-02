"""Configuration for mode grounding and transition extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable settings for the matched-exposure transition-timing analysis.

    The defaults use a persistence window of ``k = 3`` and a short-gap bridge
    length of ``r = 1``. The remaining thresholds control rule-based mode
    grounding from observable rollout features.
    """

    persistence_k: int = 3
    bridge_max_gap: int = 1
    exposure_t_default: int = 0
    response_horizon: int | None = None
    close_front_distance_m: float = 24.0
    very_close_front_distance_m: float = 14.0
    low_front_distance_score: float = 0.42
    low_closest_k_score: float = 0.38
    high_speed_score: float = 0.78
    high_speed_mps: float = 30.0
    idle_continuation_enabled: bool = True
    high_speed_idle_score: float = 0.55
    high_speed_idle_mps: float = 28.0
    min_confidence_margin: float = 0.05
    bootstrap_samples: int = 1000
    random_seed: int = 7

    def validate(self) -> None:
        if self.persistence_k < 1:
            raise ValueError("persistence_k must be >= 1")
        if self.bridge_max_gap < 0:
            raise ValueError("bridge_max_gap must be >= 0")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0")
