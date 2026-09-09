#!/usr/bin/env python3
"""Equivalence bounds for the nine cells of the Stage-5 summary comparison.

The summary comparison scores methods on two tasks that need different cells.
The identification task ("which policy begins its transition earlier?") is only
answerable where a difference exists. The over-claiming task ("is there a
difference at all?") is only measurable where one does not. Both need the split
fixed before any summary is generated, otherwise the classification can be
chosen to flatter whichever method it is applied to.

For each of the three arms and three agent pairs this reports the across-seed
median of the per-seed median paired gap, its bootstrap interval over seeds,
and the seed-level sign counts. The equivalence bound is the larger absolute
end of that interval: the largest difference the twenty seeds leave consistent
with the data. A method that claims a directional difference beyond the bound
in a cell classified as null is making a false claim, which is what the bound
exists to adjudicate.

Cells are classified `effect` when the seeds agree on a direction at the
project's usual sign-test level and the interval excludes zero, and `null`
otherwise. The classification is not read off the p-value alone: each null cell
must also have a reward-weight reason to be similar, recorded in the
pre-registration, so that the label does not depend on the outcome it will be
used to judge.

Primary measure is onset timing in seconds, which is the project's endpoint.
The onset-gap in metres is a separate corroborating measure and is not mixed in.

Run from the repository root:

    PYTHONPATH=src python3 scripts/preregistration_cell_bounds.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aggregate_selected_20seed import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    GAP_LABELS,
    bootstrap_median_ci,
    read_csv_rows,
    sign_test_p,
    step_seconds,
    write_csv,
)

#: A cell counts as an effect only if the seeds agree on direction at this
#: level and the across-seed interval excludes zero. Both are required.
SIGN_TEST_ALPHA = 0.05


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    run_prefix: str
    seeds: tuple[int, ...]
    analysis_target: str
    analysis_dir: str


ARMS = (
    Arm(
        "A",
        "single lane",
        "single_lane_slow_front_fixed_100k_seed",
        tuple(range(3100, 3120)),
        "slowdown_onset",
        "analysis",
    ),
    Arm(
        "B",
        "two lanes, open target lane",
        "multilane_twolane_open_fixed_100k_seed",
        tuple(range(4100, 4120)),
        "lane_change_onset",
        "analysis_onset_fixed",
    ),
    Arm(
        "C",
        "two lanes, occupied target lane",
        "multilane_twolane_occupied_fixed_100k_seed",
        tuple(range(4200, 4220)),
        "lane_change_onset",
        "analysis_onset_fixed",
    ),
)


def seed_median_gaps(run_dir: Path, arm: Arm) -> dict[str, float]:
    """Median valid paired gap per gap label, in seconds, for one seed."""
    per_label: dict[str, list[float]] = {label: [] for label in GAP_LABELS}
    seconds = step_seconds(run_dir)
    for row in read_csv_rows(run_dir / arm.analysis_dir / "timing_gaps.csv"):
        if row["analysis_target"] != arm.analysis_target:
            continue
        if row["gap_status"] != "valid_pair":
            continue
        label = f"{row['agent_b']} - {row['agent_a']}"
        if label not in per_label:
            continue
        per_label[label].append(float(row["gap_b_minus_a"]) * seconds)
    return {
        label: statistics.median(values)
        for label, values in per_label.items()
        if values
    }


def collect(arm: Arm, outputs_root: Path) -> list[dict]:
    per_label: dict[str, list[float]] = {label: [] for label in GAP_LABELS}
    for seed in arm.seeds:
        run_dir = outputs_root / f"{arm.run_prefix}{seed}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing run directory: {run_dir}")
        for label, value in seed_median_gaps(run_dir, arm).items():
            per_label[label].append(value)

    rows: list[dict] = []
    for label in GAP_LABELS:
        values = per_label[label]
        positive = sum(1 for v in values if v > 0)
        negative = sum(1 for v in values if v < 0)
        ties = sum(1 for v in values if v == 0)
        p = sign_test_p(positive, negative)
        low, high = bootstrap_median_ci(values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
        excludes_zero = (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)
        classification = "effect" if (p < SIGN_TEST_ALPHA and excludes_zero) else "null"
        rows.append(
            {
                "arm": arm.key,
                "arm_label": arm.label,
                "analysis_target": arm.analysis_target,
                "gap_label": label,
                "seeds": len(values),
                "median_gap_seconds": round(statistics.median(values), 3),
                "ci_low_seconds": round(low, 3),
                "ci_high_seconds": round(high, 3),
                "seeds_positive": positive,
                "seeds_negative": negative,
                "seeds_tied": ties,
                "sign_test_p": round(p, 6),
                "interval_excludes_zero": excludes_zero,
                "classification": classification,
                "equivalence_bound_seconds": round(max(abs(low), abs(high)), 3),
            }
        )
    return rows


def print_report(rows: list[dict]) -> None:
    header = (
        f"{'arm':>3s} {'contrast':10s} {'med(s)':>7s} {'95% CI':>17s} "
        f"{'+/-/tie':>10s} {'p':>9s} {'class':>7s} {'bound(s)':>9s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = f"[{row['ci_low_seconds']:+.2f},{row['ci_high_seconds']:+.2f}]"
        counts = f"{row['seeds_positive']}/{row['seeds_negative']}/{row['seeds_tied']}"
        print(
            f"{row['arm']:>3s} {row['gap_label']:10s} "
            f"{row['median_gap_seconds']:>+7.2f} {ci:>17s} {counts:>10s} "
            f"{row['sign_test_p']:>9.2e} {row['classification']:>7s} "
            f"{row['equivalence_bound_seconds']:>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/summary_comparison/preregistration_cell_bounds.csv"),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for arm in ARMS:
        rows.extend(collect(arm, args.outputs_root))

    write_csv(args.out, list(rows[0]), rows)
    print_report(rows)
    effects = sum(1 for r in rows if r["classification"] == "effect")
    print()
    print(f"{effects} effect cells, {len(rows) - effects} null cells")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
