#!/usr/bin/env python3
"""Aggregate the checkpoint-gated 20-seed runs across seeds.

The per-seed pipeline already writes `analysis/outcome_summary.csv`,
`analysis/gap_summary.csv`, and `analysis/episode_outcomes.csv`. This script
treats the seed as the unit of replication: it summarises each seed first, then
reports the across-seed distribution of those seed-level statistics. Pooling
raw exposures across seeds would understate uncertainty, because the 36
exposures inside one seed share a single trained policy.

It reads only development-grid artefacts. The sealed held-out grid is never
touched, and no model is loaded.

Run from the repository root:

    python3 scripts/aggregate_selected_20seed.py

Outputs land in `outputs/<run_prefix without seed>_20seed/analysis/`.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

AGENTS = ("FD", "BAL", "SP")
GAP_LABELS = ("BAL - FD", "SP - BAL", "SP - FD")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345


@dataclass(frozen=True)
class Experiment:
    key: str
    run_prefix: str
    seeds: tuple[int, ...]
    analysis_target: str
    aggregate_dir: str


EXPERIMENTS = (
    Experiment(
        key="single_lane",
        run_prefix="single_lane_slow_front_selected_100k_seed",
        seeds=tuple(range(3000, 3020)),
        analysis_target="slowdown_onset",
        aggregate_dir="single_lane_slow_front_selected_100k_20seed",
    ),
    Experiment(
        key="twolane",
        run_prefix="multilane_open_lane_change_twolane_selected_100k_seed",
        seeds=tuple(range(4000, 4020)),
        analysis_target="lane_change_onset",
        aggregate_dir="multilane_open_lane_change_twolane_selected_100k_20seed",
    ),
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """Median and the two hinges, using the same convention as the pipeline."""
    ordered = sorted(values)
    n = len(ordered)
    median = statistics.median(ordered)
    if n < 2:
        return median, median, median
    lower = ordered[: n // 2]
    upper = ordered[(n + 1) // 2 :]
    return statistics.median(lower), median, statistics.median(upper)


def bootstrap_median_ci(
    values: Sequence[float],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile CI for the median, resampling whole seeds."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(draw))
    medians.sort()
    low = medians[int(0.025 * (samples - 1))]
    high = medians[int(0.975 * (samples - 1))]
    return low, high


def sign_test_p(positive: int, negative: int) -> float:
    """Two-sided exact sign test on the seed-level medians, ties dropped.

    This asks only whether the seeds agree on the direction of an effect, which
    is the weakest claim the 20-seed design can support. It does not assume the
    seed medians are symmetric or continuous, which matters here because the
    medians land on half policy steps.
    """
    n = positive + negative
    if n == 0:
        return float("nan")
    extreme = max(positive, negative)
    tail = sum(math.comb(n, i) for i in range(extreme, n + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def step_seconds(run_dir: Path) -> float:
    """Seconds per policy step, taken from the recorded policy frequency.

    The multi-lane analysis tables carry no `*_seconds` columns, so the
    conversion has to come from `training_runs.csv` rather than from a pair of
    step/second columns.
    """
    frequencies = {
        float(row["policy_frequency_hz"])
        for row in read_csv_rows(run_dir / "training_runs.csv")
    }
    if len(frequencies) != 1:
        raise SystemExit(f"Inconsistent policy frequency in {run_dir}: {frequencies}")
    frequency = frequencies.pop()
    if frequency <= 0:
        raise SystemExit(f"Invalid policy frequency in {run_dir}: {frequency}")
    return 1.0 / frequency


def seed_latencies(run_dir: Path, target: str) -> dict[str, list[float]]:
    """Valid-onset response latencies per agent, in policy steps."""
    per_agent: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    for row in read_csv_rows(run_dir / "analysis" / "episode_outcomes.csv"):
        if row["analysis_target"] != target:
            continue
        if row["episode_outcome"] != "valid_onset":
            continue
        latency = row["response_latency"]
        if latency == "":
            continue
        per_agent[row["agent_condition"]].append(float(latency))
    return per_agent


def seed_collisions(run_dir: Path, target: str) -> dict[str, tuple[int, int]]:
    per_agent: dict[str, list[int]] = {agent: [] for agent in AGENTS}
    for row in read_csv_rows(run_dir / "analysis" / "episode_outcomes.csv"):
        if row["analysis_target"] != target:
            continue
        per_agent[row["agent_condition"]].append(
            1 if row["collision_flag"].strip().lower() == "true" else 0
        )
    return {
        agent: (sum(flags), len(flags)) for agent, flags in per_agent.items()
    }


def selected_steps(run_dir: Path) -> dict[str, int]:
    steps: dict[str, int] = {}
    for row in read_csv_rows(run_dir / "model_selection.csv"):
        if row["selected"] == "True":
            steps[row["agent_condition"]] = int(row["checkpoint_step"])
    return steps


def across_seed_row(
    label: str,
    values: Sequence[float],
    unit: str,
    seconds_per_step: float,
    test_sign: bool,
) -> dict:
    q1, median, q3 = quartiles(values)
    low, high = bootstrap_median_ci(values)
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    row = {
        "label": label,
        "unit": unit,
        "n_seeds": len(values),
        "median_of_seed_medians": round(median, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(q3 - q1, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "bootstrap_ci_low": round(low, 4),
        "bootstrap_ci_high": round(high, 4),
        "n_seeds_positive": positive,
        "n_seeds_negative": negative,
        "n_seeds_zero": len(values) - positive - negative,
        # A latency is non-negative by construction, so its sign carries no
        # information; only the paired gaps get a directional test.
        "sign_test_p": (
            round(sign_test_p(positive, negative), 5) if test_sign else ""
        ),
    }
    row["median_seconds"] = round(median * seconds_per_step, 4)
    row["q1_seconds"] = round(q1 * seconds_per_step, 4)
    row["q3_seconds"] = round(q3 * seconds_per_step, 4)
    return row


def aggregate(experiment: Experiment, outputs_root: Path) -> dict:
    run_dirs = {
        seed: outputs_root / f"{experiment.run_prefix}{seed}"
        for seed in experiment.seeds
    }
    missing = [str(path) for path in run_dirs.values() if not path.is_dir()]
    if missing:
        raise SystemExit(f"Missing run directories:\n  " + "\n  ".join(missing))

    distinct_step_seconds = {step_seconds(path) for path in run_dirs.values()}
    if len(distinct_step_seconds) != 1:
        raise SystemExit(
            f"Seeds disagree on policy frequency: {distinct_step_seconds}"
        )
    seconds_per_step = distinct_step_seconds.pop()

    latency_rows: list[dict] = []
    gap_rows: list[dict] = []
    outcome_rows: list[dict] = []
    selection_rows: list[dict] = []

    latency_by_agent: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    gap_by_label: dict[str, list[float]] = {label: [] for label in GAP_LABELS}
    seeds_all_final: list[int] = []

    for seed in experiment.seeds:
        run_dir = run_dirs[seed]
        steps = selected_steps(run_dir)
        if set(steps.values()) == {100000}:
            seeds_all_final.append(seed)
        for agent in AGENTS:
            selection_rows.append(
                {
                    "seed": seed,
                    "agent_condition": agent,
                    "selected_checkpoint_step": steps[agent],
                    "walked_back": steps[agent] != 100000,
                }
            )

        latencies = seed_latencies(run_dir, experiment.analysis_target)
        collisions = seed_collisions(run_dir, experiment.analysis_target)
        for agent in AGENTS:
            values = latencies[agent]
            q1, median, q3 = quartiles(values) if values else (0.0, 0.0, 0.0)
            latency_by_agent[agent].append(median)
            collided, total = collisions[agent]
            latency_rows.append(
                {
                    "seed": seed,
                    "agent_condition": agent,
                    "n_valid_onsets": len(values),
                    "median_latency_steps": round(median, 4),
                    "q1_latency_steps": round(q1, 4),
                    "q3_latency_steps": round(q3, 4),
                    "median_latency_seconds": round(median * seconds_per_step, 4),
                    "collision_episodes": collided,
                    "n_episodes": total,
                    "collision_rate": round(collided / total, 4) if total else "",
                    "selected_checkpoint_step": steps[agent],
                }
            )

        for row in read_csv_rows(run_dir / "analysis" / "outcome_summary.csv"):
            if row["analysis_target"] != experiment.analysis_target:
                continue
            outcome_rows.append({"seed": seed, **row})

        for row in read_csv_rows(run_dir / "analysis" / "gap_summary.csv"):
            if row["analysis_target"] != experiment.analysis_target:
                continue
            label = row["gap_label"]
            median_gap = float(row["median_gap"])
            gap_by_label[label].append(median_gap)
            gap_rows.append(
                {
                    "seed": seed,
                    "gap_label": label,
                    "n_valid_pairs": int(row["n_valid_pairs"]),
                    "n_pairs": int(row["n_pairs"]),
                    "median_gap_steps": median_gap,
                    "q1_gap_steps": float(row["q1_gap"]),
                    "q3_gap_steps": float(row["q3_gap"]),
                    "median_gap_seconds": round(median_gap * seconds_per_step, 4),
                }
            )

    summary_rows = [
        across_seed_row(
            f"latency:{agent}",
            latency_by_agent[agent],
            "policy_steps",
            seconds_per_step,
            test_sign=False,
        )
        for agent in AGENTS
    ]
    summary_rows += [
        across_seed_row(
            f"gap:{label}",
            gap_by_label[label],
            "policy_steps",
            seconds_per_step,
            test_sign=True,
        )
        for label in GAP_LABELS
    ]

    # Sensitivity: restrict to seeds where no agent needed a checkpoint walk-back,
    # so every policy has the same 100,000 effective training steps.
    sensitivity_rows: list[dict] = []
    if seeds_all_final and len(seeds_all_final) < len(experiment.seeds):
        index = {seed: i for i, seed in enumerate(experiment.seeds)}
        keep = [index[seed] for seed in seeds_all_final]
        for agent in AGENTS:
            values = [latency_by_agent[agent][i] for i in keep]
            sensitivity_rows.append(
                across_seed_row(
                    f"latency:{agent}",
                    values,
                    "policy_steps",
                    seconds_per_step,
                    test_sign=False,
                )
            )
        for label in GAP_LABELS:
            values = [gap_by_label[label][i] for i in keep]
            sensitivity_rows.append(
                across_seed_row(
                    f"gap:{label}",
                    values,
                    "policy_steps",
                    seconds_per_step,
                    test_sign=True,
                )
            )

    out_dir = outputs_root / experiment.aggregate_dir / "analysis"
    write_csv(
        out_dir / "seed_level_latency.csv",
        [
            "seed",
            "agent_condition",
            "n_valid_onsets",
            "median_latency_steps",
            "q1_latency_steps",
            "q3_latency_steps",
            "median_latency_seconds",
            "collision_episodes",
            "n_episodes",
            "collision_rate",
            "selected_checkpoint_step",
        ],
        latency_rows,
    )
    write_csv(
        out_dir / "seed_level_gaps.csv",
        [
            "seed",
            "gap_label",
            "n_valid_pairs",
            "n_pairs",
            "median_gap_steps",
            "q1_gap_steps",
            "q3_gap_steps",
            "median_gap_seconds",
        ],
        gap_rows,
    )
    write_csv(
        out_dir / "seed_level_outcomes.csv",
        list(outcome_rows[0].keys()),
        outcome_rows,
    )
    write_csv(
        out_dir / "model_selection_summary.csv",
        ["seed", "agent_condition", "selected_checkpoint_step", "walked_back"],
        selection_rows,
    )
    summary_fields = list(summary_rows[0].keys())
    write_csv(out_dir / "across_seed_summary.csv", summary_fields, summary_rows)
    if sensitivity_rows:
        write_csv(
            out_dir / "across_seed_summary_final_checkpoint_only.csv",
            summary_fields,
            sensitivity_rows,
        )

    return {
        "experiment": experiment,
        "seconds_per_step": seconds_per_step,
        "summary_rows": summary_rows,
        "sensitivity_rows": sensitivity_rows,
        "seeds_all_final": seeds_all_final,
        "latency_rows": latency_rows,
        "outcome_rows": outcome_rows,
        "out_dir": out_dir,
    }


def print_report(result: dict) -> None:
    experiment: Experiment = result["experiment"]
    seconds = result["seconds_per_step"]
    print(f"\n=== {experiment.key} ({experiment.analysis_target}) ===")
    print(f"seeds: {len(experiment.seeds)}  seconds_per_step: {seconds}")

    valid = sum(int(row["valid_onset"]) for row in result["outcome_rows"])
    censored = sum(int(row["no_onset_censored"]) for row in result["outcome_rows"])
    terminal = sum(int(row["terminal_failure"]) for row in result["outcome_rows"])
    total = sum(int(row["n"]) for row in result["outcome_rows"])
    print(
        f"onsets: valid={valid}/{total} censored={censored} terminal_failure={terminal}"
    )

    collided = sum(row["collision_episodes"] for row in result["latency_rows"])
    episodes = sum(row["n_episodes"] for row in result["latency_rows"])
    print(f"episodes with a collision flag: {collided}/{episodes}")

    print(f"{'label':<18}{'median':>9}{'IQR':>18}{'CI':>18}{'sign':>10}{'p':>9}")
    for row in result["summary_rows"]:
        iqr = f"[{row['q1']:g}, {row['q3']:g}]"
        ci = f"[{row['bootstrap_ci_low']:g}, {row['bootstrap_ci_high']:g}]"
        sign = f"+{row['n_seeds_positive']}/-{row['n_seeds_negative']}"
        print(
            f"{row['label']:<18}{row['median_of_seed_medians']:>9g}"
            f"{iqr:>18}{ci:>18}{sign:>10}{row['sign_test_p'] if row['sign_test_p'] == '' else format(row['sign_test_p'], 'g'):>9}"
        )

    if result["sensitivity_rows"]:
        kept = len(result["seeds_all_final"])
        print(f"-- sensitivity: {kept} seeds with all agents at step 100000 --")
        for row in result["sensitivity_rows"]:
            iqr = f"[{row['q1']:g}, {row['q3']:g}]"
            print(
                f"{row['label']:<18}{row['median_of_seed_medians']:>9g}{iqr:>18}"
            )
    print(f"wrote: {result['out_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--experiment",
        choices=[experiment.key for experiment in EXPERIMENTS] + ["all"],
        default="all",
    )
    args = parser.parse_args()

    selected = [
        experiment
        for experiment in EXPERIMENTS
        if args.experiment in ("all", experiment.key)
    ]
    for experiment in selected:
        print_report(aggregate(experiment, args.outputs_root))


if __name__ == "__main__":
    main()
