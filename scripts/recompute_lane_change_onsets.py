"""Recompute onsets for completed runs after the direction-aware onset fix.

The confirmed-lane-change onset used to accept a command in either direction
inside the confirmation window. In the two-lane settings the ego starts in the
rightmost lane, where ``LANE_RIGHT`` is a no-op the simulator executes without
moving the vehicle, so a burst of those commands could date the onset up to a
full window before the ``LANE_LEFT`` command that actually caused the change.

This script re-derives the analysis tables for completed runs from their stored
``analysis/steps.csv``. No model is loaded and no simulation is run, so the
correction costs inference nothing. Results are written to a new directory per
run and the delivered ``analysis/`` tables are left untouched, keeping the
superseded evidence intact.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.constants import (
    LANE_CHANGE_ONSET_TARGET,
    SLOWDOWN_ONSET_TARGET,
)
from highway_transition_timing.io import read_csv_rows, write_csv_rows
from highway_transition_timing.pipeline import run_pipeline


def recompute_run(run_dir: Path, target: str, bootstrap: int, subdir: str) -> dict:
    steps_path = run_dir / "analysis" / "steps.csv"
    if not steps_path.exists():
        steps_path = run_dir / "evaluation" / "steps.csv"
    exposures_path = run_dir / "evaluation" / "exposures.csv"
    steps = read_csv_rows(steps_path)
    exposures = read_csv_rows(exposures_path) if exposures_path.exists() else []

    result = run_pipeline(
        steps,
        exposures,
        AnalysisConfig(bootstrap_samples=bootstrap),
        target,
    )
    out_dir = run_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in result.tables().items():
        write_csv_rows(out_dir / f"{name}.csv", rows)

    before = {
        (row["agent_condition"], row["exposure_id"], row.get("rollout_id", "r0")): row
        for row in read_csv_rows(run_dir / "analysis" / "episode_outcomes.csv")
        if row.get("analysis_target", target) == target
    }
    shifts: list[float] = []
    changed = 0
    for row in result.episode_outcomes:
        if row.get("analysis_target", target) != target:
            continue
        key = (row["agent_condition"], row["exposure_id"], row.get("rollout_id", "r0"))
        old = before.get(key)
        if old is None or old.get("response_latency") in ("", None):
            continue
        if row.get("response_latency") in ("", None):
            continue
        delta = float(row["response_latency"]) - float(old["response_latency"])
        shifts.append(delta)
        changed += delta != 0
    return {
        "run": run_dir.name,
        "episodes_compared": len(shifts),
        "episodes_changed": changed,
        "median_shift_steps": statistics.median(shifts) if shifts else 0.0,
        "mean_shift_steps": round(statistics.fmean(shifts), 4) if shifts else 0.0,
        "max_shift_steps": max(shifts, default=0.0),
        "min_shift_steps": min(shifts, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="completed run directories")
    parser.add_argument(
        "--target",
        default=LANE_CHANGE_ONSET_TARGET,
        choices=[LANE_CHANGE_ONSET_TARGET, SLOWDOWN_ONSET_TARGET],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--subdir", default="analysis_onset_fixed")
    parser.add_argument("--summary-out", default=None, help="optional summary CSV path")
    args = parser.parse_args()

    rows = []
    for raw in args.run_dirs:
        run_dir = Path(raw)
        if not run_dir.is_dir():
            print(f"skip (not a directory): {run_dir}", file=sys.stderr)
            continue
        row = recompute_run(run_dir, args.target, args.bootstrap_samples, args.subdir)
        rows.append(row)
        print(
            f"{row['run']}: changed {row['episodes_changed']}/{row['episodes_compared']}"
            f"  median {row['median_shift_steps']:+.1f}  mean {row['mean_shift_steps']:+.3f}"
            f"  range [{row['min_shift_steps']:+.0f}, {row['max_shift_steps']:+.0f}] steps"
        )

    if rows:
        total = sum(r["episodes_changed"] for r in rows)
        compared = sum(r["episodes_compared"] for r in rows)
        print(f"\ntotal: {total}/{compared} episodes changed across {len(rows)} runs")
    if args.summary_out and rows:
        write_csv_rows(Path(args.summary_out), rows)


if __name__ == "__main__":
    main()
