from pathlib import Path

import pytest

from highway_transition_timing.constants import VALID_PAIR
from highway_transition_timing.plotting import plot_paired_timing_gaps


def test_paired_gap_boxplot_does_not_use_version_specific_labels_keyword(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    original_boxplot = Axes.boxplot

    def reject_removed_labels_keyword(self, *args, **kwargs):
        assert "labels" not in kwargs
        return original_boxplot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "boxplot", reject_removed_labels_keyword)
    output = tmp_path / "paired_timing_gaps.png"

    result = plot_paired_timing_gaps(
        [
            {
                "gap_status": VALID_PAIR,
                "agent_a": "FD",
                "agent_b": "BAL",
                "gap_b_minus_a": 2.0,
            },
            {
                "gap_status": VALID_PAIR,
                "agent_a": "FD",
                "agent_b": "BAL",
                "gap_b_minus_a": 4.0,
            },
        ],
        output,
    )

    assert result == output
    assert output.exists()
