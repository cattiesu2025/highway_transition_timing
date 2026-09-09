"""Clip-selection baselines for the Stage-5 summary comparison.

These are re-implementations of the reference code archived under
`docs/reference_impl/`, not wrappers around it. The upstream package targets
Python 3.7 with `highway-env` 1.4 and `rl-agents`, so it cannot load this
project's Double DQN policies; only the algorithm transfers. The upstream
authors likewise re-implemented HIGHLIGHTS rather than reusing its original
code, so re-implementation is the established practice for this baseline.

Two importance variants exist and they are not interchangeable. On this
project's rollouts their top-k selections do not overlap at all, so both are
carried and reported separately:

- ``max_min`` is the definition published in HIGHLIGHTS (Amir and Amir 2018).
  It scores how much is at stake in a state.
- ``max_second`` is the variant of Huber et al., which is what the
  DISAGREEMENTS comparison actually used and what its code defaults to. It
  scores how decisively the best action beats its nearest rival.

Deliberate fidelity choices, each mirroring
`docs/reference_impl/highlights_Highway_Disagreements/highlights_state_selection.py`:

- the spacing constraint applies only within one trace, and the budget counts
  selected states, not the context frames around them (`:19-58`);
- the context window is right-open, so it spans ``[t - c, t + c - 1]`` rather
  than a symmetric ``2c + 1`` frames (`:57`);
- ties in importance are broken by input order, because the upstream sort is
  stable. This matters: ``max_min`` saturates on this project's policies, and
  a saturated plateau is ordered entirely by the tie-break.

HIGHLIGHTS-DIV is deliberately absent. It is disabled in the upstream run
configuration, and its similarity test sums signed feature differences rather
than distances, which makes its threshold uninterpretable. See
`docs/reference_impl/PROVENANCE.md` sections 3.3 and 6.
"""

from __future__ import annotations

from bisect import bisect, insort_left
from collections.abc import Sequence

IMPORTANCE_MAX_MIN = "max_min"
IMPORTANCE_MAX_SECOND = "max_second"
IMPORTANCE_VARIANTS = (IMPORTANCE_MAX_MIN, IMPORTANCE_MAX_SECOND)

#: Highway-domain values from Table 1 of the DISAGREEMENTS paper. The paper
#: states 20 states per Highway trajectory, and the upstream code passes the
#: trajectory length in as the half-window, which would give 40. We take the
#: paper's stated trajectory length, so the half-window is 10.
HIGHLIGHTS_BUDGET = 5
HIGHLIGHTS_CONTEXT_LENGTH = 10
HIGHLIGHTS_MINIMUM_GAP = 0

StateKey = tuple[str, int]


def state_importance(q_values: Sequence[float], variant: str) -> float:
    """Importance of one state under the requested variant."""
    if len(q_values) < 2:
        raise ValueError("importance needs at least two action values")
    if variant == IMPORTANCE_MAX_MIN:
        return max(q_values) - min(q_values)
    if variant == IMPORTANCE_MAX_SECOND:
        return max(q_values) - sorted(q_values)[-2]
    raise ValueError(f"unknown importance variant: {variant!r}")


def select_highlights(
    scored_states: Sequence[tuple[StateKey, float]],
    budget: int = HIGHLIGHTS_BUDGET,
    context_length: int = HIGHLIGHTS_CONTEXT_LENGTH,
    minimum_gap: int = HIGHLIGHTS_MINIMUM_GAP,
) -> list[StateKey]:
    """Greedy top-importance selection with the upstream spacing constraint.

    `scored_states` pairs a ``(trace_id, step)`` key with its importance. The
    returned keys are sorted, which is what the upstream implementation keeps
    so that the spacing test can use a binary search.
    """
    if budget <= 0:
        return []
    ordered = sorted(scored_states, key=lambda item: -item[1])
    selected: list[StateKey] = []
    for (trace_id, step), _importance in ordered:
        position = bisect(selected, (trace_id, step))
        before = selected[position - 1] if position > 0 else None
        after = selected[position] if position < len(selected) else None
        if after is not None and trace_id == after[0]:
            if step + context_length + minimum_gap > after[1]:
                continue
        if before is not None and trace_id == before[0]:
            if step - context_length - minimum_gap < before[1]:
                continue
        insort_left(selected, (trace_id, step))
        if len(selected) == budget:
            break
    return selected


def context_window(
    step: int,
    trace_length: int,
    context_length: int = HIGHLIGHTS_CONTEXT_LENGTH,
) -> range:
    """Frames shown around a selected state, reproducing the right-open span."""
    start = max(step - context_length, 0)
    end = min(step + context_length, trace_length - 1)
    return range(start, end)
