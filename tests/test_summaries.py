"""Tests for the clip-selection baselines ported from the reference code."""

from __future__ import annotations

import pytest

from highway_transition_timing.summaries import (
    IMPORTANCE_MAX_MIN,
    IMPORTANCE_MAX_SECOND,
    context_window,
    select_highlights,
    state_importance,
)


def test_importance_variants_measure_different_things() -> None:
    q_values = [1.0, 5.0, 4.9, 0.0, 2.0]
    assert state_importance(q_values, IMPORTANCE_MAX_MIN) == pytest.approx(5.0)
    assert state_importance(q_values, IMPORTANCE_MAX_SECOND) == pytest.approx(0.1)


def test_importance_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        state_importance([1.0, 2.0], "most_interesting")


def test_selection_takes_the_highest_importance_first() -> None:
    scored = [(("e0", 0), 0.1), (("e0", 100), 0.9), (("e0", 200), 0.5)]
    assert select_highlights(scored, budget=2, context_length=10) == [
        ("e0", 100),
        ("e0", 200),
    ]


def test_spacing_constraint_rejects_neighbours_in_the_same_trace() -> None:
    """A second peak 5 steps away is inside the context window and is skipped."""
    scored = [(("e0", 100), 0.9), (("e0", 105), 0.8), (("e0", 200), 0.7)]
    assert select_highlights(scored, budget=3, context_length=10) == [
        ("e0", 100),
        ("e0", 200),
    ]


def test_spacing_constraint_does_not_cross_traces() -> None:
    """The same step index in another episode is a different state."""
    scored = [(("e0", 100), 0.9), (("e1", 100), 0.8)]
    assert select_highlights(scored, budget=2, context_length=10) == [
        ("e0", 100),
        ("e1", 100),
    ]


def test_minimum_gap_widens_the_exclusion_zone() -> None:
    scored = [(("e0", 100), 0.9), (("e0", 115), 0.8)]
    assert select_highlights(scored, budget=2, context_length=10) == [
        ("e0", 100),
        ("e0", 115),
    ]
    assert select_highlights(scored, budget=2, context_length=10, minimum_gap=10) == [
        ("e0", 100)
    ]


def test_budget_counts_selected_states_not_context_frames() -> None:
    scored = [(("e0", step), float(step)) for step in range(0, 500, 50)]
    assert len(select_highlights(scored, budget=3, context_length=10)) == 3


def test_ties_are_broken_by_input_order() -> None:
    """Saturated importance leaves the tie-break in charge of the selection."""
    scored = [(("e0", 300), 1.0), (("e0", 100), 1.0), (("e0", 200), 1.0)]
    assert select_highlights(scored, budget=1, context_length=10) == [("e0", 300)]


def test_context_window_is_right_open() -> None:
    """Reproduces the upstream off-by-one: 2c frames, not 2c + 1."""
    window = context_window(100, trace_length=600, context_length=10)
    assert list(window) == list(range(90, 110))
    assert 110 not in window


def test_context_window_clips_at_both_ends() -> None:
    assert list(context_window(3, trace_length=600, context_length=10)) == list(
        range(0, 13)
    )
    assert list(context_window(598, trace_length=600, context_length=10)) == list(
        range(588, 599)
    )


def test_zero_budget_selects_nothing() -> None:
    assert select_highlights([(("e0", 1), 1.0)], budget=0) == []
