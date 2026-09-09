#!/usr/bin/env python3
"""Validate and aggregate the one-time A/B/C held-out v2 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

AGENTS = ("FD", "BAL", "SP")
PAIRS = (("FD", "BAL", "BAL - FD"), ("BAL", "SP", "SP - BAL"), ("FD", "SP", "SP - FD"))
MIN_VALID_PAIRS = 24
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260909


@dataclass(frozen=True)
class Arm:
    name: str
    run_prefix: str
    seeds: tuple[int, ...]
    kind: str
    grid_file: str
    grid_sha256: str
    variants: tuple[str, ...]


ARMS = (
    Arm(
        "A",
        "single_lane_slow_front_fixed_100k_seed",
        tuple(range(3100, 3120)),
        "single",
        "arm_ab_grid.csv",
        "094a840d57782fe63eba6604c8ac36cc45e5b1d4f5aa72d05a2cba4fe5a18caf",
        ("original", "no-front", "matched-speed-front", "far-front"),
    ),
    Arm(
        "B",
        "multilane_twolane_open_fixed_100k_seed",
        tuple(range(4100, 4120)),
        "multi",
        "arm_ab_grid.csv",
        "094a840d57782fe63eba6604c8ac36cc45e5b1d4f5aa72d05a2cba4fe5a18caf",
        ("original", "no-front", "matched-speed-front", "far-front"),
    ),
    Arm(
        "C",
        "multilane_twolane_occupied_fixed_100k_seed",
        tuple(range(4200, 4220)),
        "multi",
        "arm_c_grid.csv",
        "065af8ea3de7bfa8f56695e81675b553ea3047158e9f694c746c7fe8b30cfdf9",
        ("original", "no-front", "matched-speed-front", "far-front", "open-target-lane"),
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def median_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def bootstrap_median_ci(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    draws = sorted(
        statistics.median(values[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return draws[int(0.025 * (BOOTSTRAP_SAMPLES - 1))], draws[
        int(0.975 * (BOOTSTRAP_SAMPLES - 1))
    ]


def sign_test_p(values: Sequence[float]) -> float:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        return 1.0
    extreme = max(positive, negative)
    tail = sum(math.comb(n, index) for index in range(extreme, n + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def expected_exposures(repo_root: Path, arm: Arm) -> set[str]:
    grid = repo_root / "experiments" / "heldout_v2" / arm.grid_file
    actual = file_sha256(grid)
    if actual != arm.grid_sha256:
        raise RuntimeError(
            f"{arm.name} grid checksum mismatch: expected {arm.grid_sha256}, got {actual}"
        )
    ids = [row["exposure_id"] for row in read_csv(grid)]
    if len(ids) != 36 or len(set(ids)) != 36:
        raise RuntimeError(f"{arm.name} grid must contain 36 unique exposures")
    return set(ids)


def validate_run(run_dir: Path, arm: Arm, expected_ids: set[str]) -> Path:
    for agent in AGENTS:
        model = run_dir / "models" / f"{agent}_main.zip"
        if not model.exists():
            raise FileNotFoundError(model)
    if not (run_dir / "training_runs.csv").exists():
        raise FileNotFoundError(run_dir / "training_runs.csv")

    heldout = run_dir / "rollout_counterfactual_heldout_v2"
    manifest_rows = read_csv(heldout / "evaluation_grid_manifest.csv")
    if len(manifest_rows) != 1:
        raise RuntimeError(f"Expected one grid manifest row: {run_dir}")
    manifest = manifest_rows[0]
    if manifest["evaluation_grid"] != "heldout-v2":
        raise RuntimeError(f"Wrong evaluation grid in {run_dir}: {manifest['evaluation_grid']}")
    if manifest["grid_sha256"] != arm.grid_sha256:
        raise RuntimeError(f"Wrong grid checksum in {run_dir}")
    if int(manifest["n_exposures"]) != len(expected_ids):
        raise RuntimeError(f"Wrong exposure count in {run_dir}")
    return heldout


def read_variant_outcomes(
    heldout: Path,
    arm: Arm,
    variant: str,
    expected_ids: set[str],
) -> dict[str, dict[str, tuple[float | None, bool]]]:
    if arm.kind == "single":
        path = heldout / variant / "analysis" / "episode_outcomes.csv"
        rows = [
            row
            for row in read_csv(path)
            if row["analysis_target"] == "slowdown_onset"
        ]
        latency_field = "response_latency_seconds"
        observed = lambda row: row["episode_outcome"] == "valid_onset"
    else:
        path = heldout / variant / "analysis" / "actual_lane_change_summary.csv"
        rows = read_csv(path)
        latency_field = "first_actual_lane_change_seconds"
        observed = lambda row: parse_bool(row["actual_lane_change_observed"])

    outcomes: dict[str, dict[str, tuple[float | None, bool]]] = {}
    for agent in AGENTS:
        agent_rows = [row for row in rows if row["agent_condition"] == agent]
        ids = [row["exposure_id"] for row in agent_rows]
        if len(ids) != 36 or set(ids) != expected_ids or len(set(ids)) != 36:
            raise RuntimeError(
                f"{arm.name} {variant} {agent} does not contain the sealed 36 exposures"
            )
        outcomes[agent] = {
            row["exposure_id"]: (
                float(row[latency_field]) if observed(row) else None,
                parse_bool(row["collision_flag"]),
            )
            for row in agent_rows
        }
    return outcomes


def summarize_across_seeds(label: str, values: Sequence[float]) -> dict:
    if not values:
        return {
            "label": label,
            "n_seeds": 0,
            "median": "",
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "sign_test_p": "",
        }
    low, high = bootstrap_median_ci(values)
    return {
        "label": label,
        "n_seeds": len(values),
        "median": round(statistics.median(values), 4),
        "bootstrap_ci_low": round(low, 4),
        "bootstrap_ci_high": round(high, 4),
        "n_positive": sum(value > 0 for value in values),
        "n_negative": sum(value < 0 for value in values),
        "n_zero": sum(value == 0 for value in values),
        "sign_test_p": round(sign_test_p(values), 6),
    }


def decision(summary: dict, direction: str) -> str:
    if summary["n_seeds"] != 20:
        return "non-estimable"
    low = float(summary["bootstrap_ci_low"])
    high = float(summary["bootstrap_ci_high"])
    supported = low > 0 if direction == ">0" else high < 0
    return "supported" if supported else "not supported"


def aggregate(repo_root: Path, outputs_root: Path, output_dir: Path) -> None:
    seed_latency_rows: list[dict] = []
    seed_gap_rows: list[dict] = []
    latency_values: dict[tuple[str, str, str], list[float]] = {}
    gap_values: dict[tuple[str, str], list[float]] = {}
    occurrence_values: dict[tuple[str, str, str], list[float]] = {}

    for arm in ARMS:
        expected_ids = expected_exposures(repo_root, arm)
        for seed in arm.seeds:
            run_dir = outputs_root / f"{arm.run_prefix}{seed}"
            heldout = validate_run(run_dir, arm, expected_ids)
            variant_outcomes = {
                variant: read_variant_outcomes(heldout, arm, variant, expected_ids)
                for variant in arm.variants
            }
            for variant, outcomes in variant_outcomes.items():
                for agent in AGENTS:
                    agent_values = outcomes[agent]
                    observed_latencies = [
                        latency for latency, _collision in agent_values.values()
                        if latency is not None
                    ]
                    median_latency = median_or_none(observed_latencies)
                    occurrence = len(observed_latencies) / len(expected_ids)
                    collisions = sum(collision for _latency, collision in agent_values.values())
                    seed_latency_rows.append(
                        {
                            "arm": arm.name,
                            "seed": seed,
                            "variant": variant,
                            "agent_condition": agent,
                            "n_observed": len(observed_latencies),
                            "n_censored": len(expected_ids) - len(observed_latencies),
                            "n_total": len(expected_ids),
                            "occurrence_rate": round(occurrence, 6),
                            "median_latency_seconds": (
                                round(median_latency, 4) if median_latency is not None else ""
                            ),
                            "collision_count": collisions,
                        }
                    )
                    occurrence_values.setdefault((arm.name, variant, agent), []).append(occurrence)
                    if median_latency is not None:
                        latency_values.setdefault((arm.name, variant, agent), []).append(median_latency)

            original = variant_outcomes["original"]
            for first, second, label in PAIRS:
                paired = [
                    original[second][exposure_id][0] - original[first][exposure_id][0]
                    for exposure_id in sorted(expected_ids)
                    if original[first][exposure_id][0] is not None
                    and original[second][exposure_id][0] is not None
                ]
                estimable = len(paired) >= MIN_VALID_PAIRS
                median_gap = median_or_none(paired)
                seed_gap_rows.append(
                    {
                        "arm": arm.name,
                        "seed": seed,
                        "gap_label": label,
                        "n_valid_pairs": len(paired),
                        "n_total_pairs": len(expected_ids),
                        "estimable": estimable,
                        "median_gap_seconds": (
                            round(median_gap, 4) if median_gap is not None else ""
                        ),
                    }
                )
                if estimable and median_gap is not None:
                    gap_values.setdefault((arm.name, label), []).append(median_gap)

    across_latency_rows: list[dict] = []
    for arm in ARMS:
        for variant in arm.variants:
            for agent in AGENTS:
                key = (arm.name, variant, agent)
                latency = summarize_across_seeds(
                    f"{arm.name}:{variant}:{agent}:conditional_latency_seconds",
                    latency_values.get(key, []),
                )
                occurrence = summarize_across_seeds(
                    f"{arm.name}:{variant}:{agent}:occurrence_rate",
                    occurrence_values[key],
                )
                across_latency_rows.extend([latency, occurrence])

    across_gap_rows = []
    for arm in ARMS:
        for _first, _second, label in PAIRS:
            summary = summarize_across_seeds(
                f"{arm.name}:{label}", gap_values.get((arm.name, label), [])
            )
            across_gap_rows.append({"arm": arm.name, "gap_label": label, **summary})

    summary_index = {
        (row["arm"], row["gap_label"]): row for row in across_gap_rows
    }
    hypotheses = [
        ("A primary", "A", "SP - FD", ">0", "FD earlier than SP"),
        ("C primary", "C", "SP - FD", "<0", "SP earlier than FD"),
        ("A supporting", "A", "BAL - FD", ">0", "FD earlier than BAL"),
        ("C supporting", "C", "SP - BAL", "<0", "SP earlier than BAL"),
    ]
    decision_rows = []
    for name, arm_name, label, direction, claim in hypotheses:
        summary = summary_index[(arm_name, label)]
        decision_rows.append(
            {
                "hypothesis": name,
                "arm": arm_name,
                "gap_label": label,
                "expected_direction": direction,
                "claim": claim,
                "n_estimable_seeds": summary["n_seeds"],
                "median_gap_seconds": summary["median"],
                "bootstrap_ci_low": summary["bootstrap_ci_low"],
                "bootstrap_ci_high": summary["bootstrap_ci_high"],
                "decision": decision(summary, direction),
            }
        )
    primary = [row for row in decision_rows if "primary" in row["hypothesis"]]
    reversal = "supported" if all(row["decision"] == "supported" for row in primary) else "not supported"

    write_csv(output_dir / "seed_level_latency.csv", seed_latency_rows)
    write_csv(output_dir / "seed_level_gaps.csv", seed_gap_rows)
    write_csv(output_dir / "across_seed_latency_and_occurrence.csv", across_latency_rows)
    write_csv(output_dir / "across_seed_gaps.csv", across_gap_rows)
    write_csv(output_dir / "confirmatory_decisions.csv", decision_rows)

    lines = [
        "# Held-Out V2 Synchronisation Summary",
        "",
        f"**Primary reversal: {reversal}.**",
        "",
        "| Test | Median gap (s) | 95% CI (s) | Decision |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in decision_rows:
        lines.append(
            f"| {row['hypothesis']}: {row['claim']} | {row['median_gap_seconds']} | "
            f"[{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}] | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "Arm B is a descriptive mechanistic control. An interval containing zero is not interpreted as equivalence.",
            "",
            "See the CSV tables in this directory for all seed-level denominators, censoring, collisions, counterfactual occurrence, and sign counts.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synchronisation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/heldout_v2_20seed/analysis"),
    )
    args = parser.parse_args()
    aggregate(args.repo_root.resolve(), args.outputs_root, args.out)
    print(f"Wrote held-out v2 aggregate to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
