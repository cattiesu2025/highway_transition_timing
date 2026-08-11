#!/usr/bin/env python3
"""Re-evaluate the completed two-lane policies on a distance-aligned grid.

The two-lane development grid starts the hazard at 140/180/220/260 m, but the
measured lane-change trigger sits at a remaining gap of 130-158 m. Three of the
four distance levels therefore never approach the decision region, which is a
candidate explanation for the two-lane null that has nothing to do with the
lane-change affordance. This script tests that directly: it re-evaluates the
already trained policies on the single-lane distance band, 90/120/150/180 m,
holding ego speeds, front speeds, geometry, rewards, and the analysis target
unchanged.

No model is trained and no reward constant is changed. The sealed held-out grid
is never loaded: this script builds its own diagnostic grid and never calls
`make_sealed_heldout_specs`.

The evaluation and analysis paths are the production ones. Only
`make_eval_specs` is replaced, so any difference in the result is attributable
to the grid and not to a re-implementation.

Run from the repository root:

    python3 scripts/diagnose_twolane_grid_placement.py --seeds 4000
    python3 scripts/diagnose_twolane_grid_placement.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aggregate_selected_20seed import (  # noqa: E402
    AGENTS,
    GAP_LABELS,
    bootstrap_median_ci,
    quartiles,
    sign_test_p,
    write_csv,
)

RUN_PREFIX = "multilane_open_lane_change_twolane_selected_100k_seed"
SEEDS = tuple(range(4000, 4020))

# The single-lane band, so the two grids span the same distances.
DIAGNOSTIC_FRONT_DISTANCES = (90.0, 120.0, 150.0, 180.0)
# Unchanged from the two-lane development grid.
DIAGNOSTIC_FRONT_SPEEDS = (10.0, 14.0, 18.0)
DIAGNOSTIC_EGO_SPEEDS = (26.0, 28.0, 30.0)
NUM_EXPOSURES = 36


def load_experiment_module():
    path = REPO_ROOT / "experiments" / "multilane_open_lane_change" / "run.py"
    spec = importlib.util.spec_from_file_location(
        "multilane_open_lane_change_grid_diagnostic", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_diagnostic_eval_specs(module, num_exposures: int, config) -> list:
    """Mirror `make_eval_specs`, changing only the distance levels."""
    specs = []
    for idx in range(num_exposures):
        specs.append(
            module.build_open_lane_spec(
                exposure_id=f"G{idx:04d}",
                exposure_seed=config.seed + idx,
                ego_lane=config.ego_lane,
                ego_speed=DIAGNOSTIC_EGO_SPEEDS[
                    (idx // (len(DIAGNOSTIC_FRONT_DISTANCES) * len(DIAGNOSTIC_FRONT_SPEEDS)))
                    % len(DIAGNOSTIC_EGO_SPEEDS)
                ],
                front_distance=DIAGNOSTIC_FRONT_DISTANCES[
                    idx % len(DIAGNOSTIC_FRONT_DISTANCES)
                ],
                front_speed=DIAGNOSTIC_FRONT_SPEEDS[
                    (idx // len(DIAGNOSTIC_FRONT_DISTANCES))
                    % len(DIAGNOSTIC_FRONT_SPEEDS)
                ],
                scenario_type="grid_placement_diagnostic_slow_front",
                include_front_vehicle=True,
            )
        )
    return specs


def config_for_run(module, run_dir: Path):
    """Rebuild the run's configuration from its recorded metadata."""
    with (run_dir / "training_runs.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    collision_penalty = row.get("reward_common_collision_penalty", "")
    return module.ExperimentConfig(
        evaluation_duration=int(float(row["evaluation_duration_seconds"])),
        seed=int(row["seed"]),
        policy_frequency=int(row["policy_frequency_hz"]),
        lanes_count=int(row["lanes_count"]),
        ego_lane=int(row["ego_lane"]),
        slow_down_penalty=float(row["reward_slow_down_penalty"]),
        lane_change_penalty=float(row["reward_lane_change_penalty"]),
        collision_risk_penalty=float(row["reward_collision_risk_penalty"]),
        collision_penalty=(
            float(collision_penalty) if collision_penalty not in (None, "") else None
        ),
        dqn_variant=row.get("dqn_variant", "double-dqn"),
    )


def evaluate_seed(module, run_dir: Path, bootstrap_samples: int) -> dict[str, Any]:
    config = config_for_run(module, run_dir)

    original = module.make_eval_specs
    module.make_eval_specs = lambda n, cfg: make_diagnostic_eval_specs(module, n, cfg)
    try:
        # evaluate_agents returns (step_rows, exposure_rows), in that order.
        steps, exposures = module.evaluate_agents(
            run_dir, list(AGENTS), NUM_EXPOSURES, config
        )
    finally:
        module.make_eval_specs = original

    from highway_transition_timing.config import AnalysisConfig
    from highway_transition_timing.pipeline import run_pipeline

    result = run_pipeline(
        steps,
        exposures,
        AnalysisConfig(bootstrap_samples=bootstrap_samples),
        module.LANE_CHANGE_ONSET_TARGET,
    )
    return {"config": config, "exposures": exposures, "result": result}


def seed_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    exposure_params = {
        row["exposure_id"]: (
            float(row["nearest_front_distance_at_exposure"]),
            float(row["relative_closing_speed_at_exposure"]),
        )
        for row in payload["exposures"]
    }
    frequency = float(payload["config"].policy_frequency)

    latency: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    trigger: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    valid = 0
    total = 0
    for row in result.episode_outcomes:
        total += 1
        if row["episode_outcome"] != "valid_onset":
            continue
        raw = row.get("response_latency", "")
        if raw in ("", None):
            continue
        valid += 1
        steps_latency = float(raw)
        agent = row["agent_condition"]
        latency[agent].append(steps_latency)
        gap, closing = exposure_params[row["exposure_id"]]
        trigger[agent].append(gap - closing * steps_latency / frequency)

    gaps = {
        row["gap_label"]: float(row["median_gap"]) for row in result.gap_summary
    }
    return {
        "latency": {a: statistics.median(v) if v else float("nan") for a, v in latency.items()},
        "trigger": {a: statistics.median(v) if v else float("nan") for a, v in trigger.items()},
        "gaps": gaps,
        "valid_onsets": valid,
        "episodes": total,
    }


def summarise(label: str, values: Sequence[float], test_sign: bool) -> dict[str, Any]:
    q1, median, q3 = quartiles(values)
    low, high = bootstrap_median_ci(values)
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    return {
        "label": label,
        "n_seeds": len(values),
        "median_of_seed_medians": round(median, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "bootstrap_ci_low": round(low, 4),
        "bootstrap_ci_high": round(high, 4),
        "n_seeds_positive": positive,
        "n_seeds_negative": negative,
        "sign_test_p": (
            round(sign_test_p(positive, negative), 5) if test_sign else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/multilane_twolane_grid_placement_diagnostic"),
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    module = load_experiment_module()
    print(
        f"diagnostic grid: distances {list(DIAGNOSTIC_FRONT_DISTANCES)} "
        f"front speeds {list(DIAGNOSTIC_FRONT_SPEEDS)} "
        f"ego speeds {list(DIAGNOSTIC_EGO_SPEEDS)}"
    )

    per_seed_rows: list[dict[str, Any]] = []
    latency_by_agent: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    trigger_by_agent: dict[str, list[float]] = {agent: [] for agent in AGENTS}
    gap_by_label: dict[str, list[float]] = {label: [] for label in GAP_LABELS}

    for seed in args.seeds:
        run_dir = args.outputs_root / f"{RUN_PREFIX}{seed}"
        started = time.time()
        stats = seed_statistics(
            evaluate_seed(module, run_dir, args.bootstrap_samples)
        )
        elapsed = time.time() - started
        for agent in AGENTS:
            latency_by_agent[agent].append(stats["latency"][agent])
            trigger_by_agent[agent].append(stats["trigger"][agent])
            per_seed_rows.append(
                {
                    "seed": seed,
                    "agent_condition": agent,
                    "median_latency_steps": round(stats["latency"][agent], 4),
                    "median_trigger_gap_m": round(stats["trigger"][agent], 2),
                }
            )
        for label in GAP_LABELS:
            gap_by_label[label].append(stats["gaps"][label])
        print(
            f"seed {seed}: valid {stats['valid_onsets']}/{stats['episodes']}  "
            f"latency med "
            + " ".join(f"{a}={stats['latency'][a]:.1f}" for a in AGENTS)
            + "  trigger med "
            + " ".join(f"{a}={stats['trigger'][a]:.0f}m" for a in AGENTS)
            + f"  ({elapsed:.0f}s)"
        )

    summary = [summarise(f"latency:{a}", latency_by_agent[a], False) for a in AGENTS]
    summary += [summarise(f"trigger_gap:{a}", trigger_by_agent[a], False) for a in AGENTS]
    summary += [summarise(f"gap:{g}", gap_by_label[g], True) for g in GAP_LABELS]

    analysis_dir = args.out / "analysis"
    write_csv(
        analysis_dir / "seed_level.csv",
        ["seed", "agent_condition", "median_latency_steps", "median_trigger_gap_m"],
        per_seed_rows,
    )
    write_csv(analysis_dir / "across_seed_summary.csv", list(summary[0].keys()), summary)

    print()
    print(f"{'label':<20}{'median':>10}{'IQR':>18}{'CI':>18}{'sign':>10}{'p':>9}")
    for row in summary:
        iqr = f"[{row['q1']:g}, {row['q3']:g}]"
        ci = f"[{row['bootstrap_ci_low']:g}, {row['bootstrap_ci_high']:g}]"
        sign = f"+{row['n_seeds_positive']}/-{row['n_seeds_negative']}"
        p = row["sign_test_p"]
        p_text = p if p == "" else format(p, "g")
        print(
            f"{row['label']:<20}{row['median_of_seed_medians']:>10g}{iqr:>18}"
            f"{ci:>18}{sign:>10}{p_text:>9}"
        )
    print(f"\nwrote: {analysis_dir}")


if __name__ == "__main__":
    main()
