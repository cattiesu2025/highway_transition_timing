"""Checkpoint saving and eligibility-gated model selection.

Both maintained experiments use the same protocol:

1. save a checkpoint every :data:`CHECKPOINT_INTERVAL_STEPS` policy steps;
2. evaluate checkpoints only on the development grid;
3. keep the latest checkpoint whose frozen eligibility gate passes.

Gates are evaluated lazily from the newest checkpoint backwards and stop at the
first eligible checkpoint, which is equivalent to selecting the latest eligible
checkpoint but usually costs one or two evaluations instead of one per
checkpoint.

Every gate criterion refers only to the behavioural validity of a single agent:
a collision ceiling, an estimability floor, and control specificity. No
criterion refers to any FD/BAL/SP contrast, so selection cannot be steered
towards a wanted between-condition difference. The thresholds below are frozen
before the sealed held-out grid is opened and are applied identically to every
agent condition.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from .constants import VALID_ONSET, TERMINAL_FAILURE
from .utils import get_float, get_str, median

CHECKPOINT_INTERVAL_STEPS = 5_000

# The approved thresholds are stated on the maintained 36-scene development
# grid: at most 1 terminal failure and at least 27 valid onsets. They are held
# as rates so the gate stays well defined on a smaller grid during code smokes,
# and they reproduce exactly 1 and 27 when the grid has 36 exposures.
DEVELOPMENT_GRID_EXPOSURES = 36
MAX_TERMINAL_FAILURES = 1
MIN_VALID_ONSETS = 27
MAX_TERMINAL_FAILURE_RATE = MAX_TERMINAL_FAILURES / DEVELOPMENT_GRID_EXPOSURES
MIN_VALID_ONSET_RATE = MIN_VALID_ONSETS / DEVELOPMENT_GRID_EXPOSURES
MAX_NO_FRONT_VALID_ONSETS = 0
MIN_CONTROL_ONSET_DELAY_SECONDS = 1.0

ORIGINAL_VARIANT = "original"
NO_FRONT_VARIANT = "no-front"
DELAYED_CONTROL_VARIANTS = ("matched-speed-front", "far-front")
GATE_VARIANTS = (ORIGINAL_VARIANT, NO_FRONT_VARIANT, *DELAYED_CONTROL_VARIANTS)


@dataclass(frozen=True)
class VariantOutcome:
    """Development-grid outcome counts for one agent under one variant."""

    valid_onsets: int
    censored: int
    terminal_failures: int
    median_onset_seconds: float | None


@dataclass(frozen=True)
class GateResult:
    """Outcome of the frozen eligibility gate for one checkpoint."""

    eligible: bool
    criteria: dict[str, bool]
    detail: str


def variant_outcome(
    episode_outcomes: Sequence[Mapping[str, Any]],
    agent: str,
    policy_step_seconds: float,
) -> VariantOutcome:
    """Summarise one agent's episode outcomes into gate inputs."""

    rows = [
        row
        for row in episode_outcomes
        if get_str(row, "agent_condition") == agent
    ]
    onset_seconds = [
        get_float(row, "response_latency") * policy_step_seconds
        for row in rows
        if get_str(row, "episode_outcome") == VALID_ONSET
    ]
    return VariantOutcome(
        valid_onsets=len(onset_seconds),
        censored=sum(
            1
            for row in rows
            if get_str(row, "episode_outcome") not in {VALID_ONSET, TERMINAL_FAILURE}
        ),
        terminal_failures=sum(
            1
            for row in rows
            if get_str(row, "episode_outcome") == TERMINAL_FAILURE
        ),
        median_onset_seconds=median(onset_seconds),
    )


def evaluate_gate(outcomes: Mapping[str, VariantOutcome]) -> GateResult:
    """Apply the frozen eligibility gate to one checkpoint's variant outcomes."""

    missing = [variant for variant in GATE_VARIANTS if variant not in outcomes]
    if missing:
        raise ValueError(f"Gate is missing variant outcomes: {missing}")

    original = outcomes[ORIGINAL_VARIANT]
    episodes = original.valid_onsets + original.censored + original.terminal_failures
    if episodes <= 0:
        raise ValueError("The original variant has no episodes to gate on")
    max_terminal = math.floor(MAX_TERMINAL_FAILURE_RATE * episodes)
    min_valid = math.ceil(MIN_VALID_ONSET_RATE * episodes)

    criteria: dict[str, bool] = {
        "safety": original.terminal_failures <= max_terminal,
        "estimability": original.valid_onsets >= min_valid,
        "no_front_specificity": (
            outcomes[NO_FRONT_VARIANT].valid_onsets <= MAX_NO_FRONT_VALID_ONSETS
        ),
    }
    details = [
        f"terminal={original.terminal_failures}/{episodes} (max {max_terminal})",
        f"valid={original.valid_onsets}/{episodes} (min {min_valid})",
        f"no_front_valid={outcomes[NO_FRONT_VARIANT].valid_onsets}",
    ]

    for variant in DELAYED_CONTROL_VARIANTS:
        control = outcomes[variant]
        key = f"{variant.replace('-', '_')}_specificity"
        if control.valid_onsets == 0:
            criteria[key] = True
            details.append(f"{variant}=no_onset")
            continue
        if control.median_onset_seconds is None or original.median_onset_seconds is None:
            criteria[key] = False
            details.append(f"{variant}=median_unavailable")
            continue
        delay = control.median_onset_seconds - original.median_onset_seconds
        criteria[key] = delay >= MIN_CONTROL_ONSET_DELAY_SECONDS
        details.append(f"{variant}_delay={delay:.2f}s")

    return GateResult(
        eligible=all(criteria.values()),
        criteria=criteria,
        detail="; ".join(details),
    )


class CheckpointSaver(BaseCallback):
    """Persist a policy snapshot every ``checkpoint_every`` policy steps.

    The callback is read-only with respect to learning: it saves the model and
    does not touch the environment, the replay buffer, or any random stream.
    """

    def __init__(
        self,
        agent: str,
        checkpoint_dir: Path,
        checkpoint_every: int = CHECKPOINT_INTERVAL_STEPS,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.next_checkpoint = checkpoint_every
        self.last_saved_step = -1
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Stable-Baselines3 calls callbacks after env.step() but before the
        # current transition is stored and trained on. Label step N at callback
        # N+1 so a checkpoint contains exactly N completed transitions.
        completed_steps = self.num_timesteps - int(self.training_env.num_envs)
        total_target = int(getattr(self.model, "_total_timesteps", 0))
        while completed_steps >= self.next_checkpoint:
            step = self.next_checkpoint
            self.next_checkpoint += self.checkpoint_every
            if step < total_target:
                self._save(step)
        return True

    def _on_training_end(self) -> None:
        self._save(int(self.num_timesteps))

    def _save(self, step: int) -> None:
        if step == self.last_saved_step:
            return
        self.model.save(checkpoint_path(self.checkpoint_dir, self.agent, step))
        self.last_saved_step = step


def checkpoint_path(checkpoint_dir: Path, agent: str, step: int) -> Path:
    return Path(checkpoint_dir) / f"{agent}_step_{step:06d}.zip"


def available_checkpoints(checkpoint_dir: Path, agent: str) -> list[tuple[int, Path]]:
    """Return ``(step, path)`` pairs for one agent, oldest first."""

    prefix = f"{agent}_step_"
    found: list[tuple[int, Path]] = []
    for path in Path(checkpoint_dir).glob(f"{prefix}*.zip"):
        try:
            step = int(path.stem[len(prefix) :])
        except ValueError:
            continue
        found.append((step, path))
    return sorted(found)


def select_latest_eligible_checkpoint(
    agent: str,
    checkpoints: Sequence[tuple[int, Path]],
    evaluate: Callable[[Path], Mapping[str, VariantOutcome]],
) -> tuple[tuple[int, Path] | None, list[dict[str, Any]]]:
    """Walk checkpoints newest first and stop at the first eligible one.

    Returns the selected ``(step, path)`` pair, or ``None`` when no checkpoint
    passes the gate, together with one audit record per checkpoint examined.
    """

    if not checkpoints:
        raise ValueError(f"No checkpoints available for agent {agent}")

    records: list[dict[str, Any]] = []
    selected: tuple[int, Path] | None = None
    for step, path in sorted(checkpoints, reverse=True):
        outcomes = evaluate(path)
        gate = evaluate_gate(outcomes)
        record: dict[str, Any] = {
            "agent_condition": agent,
            "checkpoint_step": step,
            "checkpoint_path": str(path),
            "eligible": gate.eligible,
            "gate_detail": gate.detail,
            "selected": False,
        }
        for name, passed in gate.criteria.items():
            record[f"gate_{name}"] = passed
        for variant in GATE_VARIANTS:
            outcome = outcomes[variant]
            key = variant.replace("-", "_")
            record[f"{key}_valid_onsets"] = outcome.valid_onsets
            record[f"{key}_censored"] = outcome.censored
            record[f"{key}_terminal_failures"] = outcome.terminal_failures
            record[f"{key}_median_onset_seconds"] = (
                round(outcome.median_onset_seconds, 6)
                if outcome.median_onset_seconds is not None
                else ""
            )
        records.append(record)
        if gate.eligible:
            record["selected"] = True
            selected = (step, path)
            break

    return selected, records
