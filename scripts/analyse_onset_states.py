#!/usr/bin/env python3
"""Compare the physical state at onset across agents, handling left censoring.

An onset that fires at `t = 0` tells us only that the agent's threshold is at or
beyond the scene's starting gap. The recorded value is then the starting gap
itself, so censored observations pile up on the four grid levels and drag the
estimate down. The pull is worse for the agent with the higher threshold, so
ignoring it compresses the contrast between agents.

Exposures are split three ways, per agent pair:

- both agents censored: the pair says nothing about which threshold is larger,
  so it is dropped;
- one agent censored: the censored agent's threshold is at least the starting
  gap, so when that already exceeds the other agent's observed value the
  direction is known even though the size is not. These pairs count towards the
  sign test only;
- neither censored: used for both direction and size.

Magnitudes therefore come from uncensored pairs and the sign test uses every
pair whose direction is determined. Pairing is always within one exposure and
one seed, which is what the matched-exposure design is for.

Run from the repository root:

    python3 scripts/analyse_onset_states.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aggregate_selected_20seed import (  # noqa: E402
    AGENTS,
    bootstrap_median_ci,
    quartiles,
    sign_test_p,
    write_csv,
)

PAIRS = (("FD", "BAL"), ("FD", "SP"), ("BAL", "SP"))
VARIABLES = ("gap", "ttc", "headway")
VEHICLE_LENGTH = 5.0


@dataclass(frozen=True)
class Experiment:
    key: str
    run_prefix: str
    seeds: tuple[int, ...]
    analysis_target: str


EXPERIMENTS = (
    Experiment(
        "single_lane",
        "single_lane_slow_front_selected_100k_seed",
        tuple(range(3000, 3020)),
        "slowdown_onset",
    ),
    Experiment(
        "twolane",
        "multilane_open_lane_change_twolane_selected_100k_seed",
        tuple(range(4000, 4020)),
        "lane_change_onset",
    ),
)


def onset_steps(run_dir: Path, target: str) -> dict[str, int]:
    """Onset step per episode, for episodes that produced a valid onset."""
    steps: dict[str, int] = {}
    with (run_dir / "analysis" / "episode_outcomes.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["analysis_target"] != target:
                continue
            if row["episode_outcome"] != "valid_onset":
                continue
            onset = row["target_onset_t"]
            if onset not in ("", "None"):
                steps[row["episode_id"]] = int(float(onset))
    return steps


def onset_states(run_dir: Path, target: str) -> dict[tuple[str, str], dict[str, Any]]:
    """State at onset, keyed by (exposure_id, agent)."""
    onsets = onset_steps(run_dir, target)
    states: dict[tuple[str, str], dict[str, Any]] = {}
    with (run_dir / "analysis" / "steps.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            episode = row["episode_id"]
            if episode not in onsets or int(row["t"]) != onsets[episode]:
                continue
            gap = row["nearest_front_distance"]
            front_speed = row["front_vehicle_speed"]
            if gap in ("", "inf", "None") or front_speed in ("", "inf", "None"):
                continue
            gap = float(gap)
            ego_speed = float(row["ego_speed"])
            closing = ego_speed - float(front_speed)
            net = gap - VEHICLE_LENGTH
            states[(row["exposure_id"], row["agent_condition"])] = {
                "gap": gap,
                "ttc": net / closing if closing > 0 else None,
                "headway": gap / ego_speed,
                "censored": onsets[episode] == 0,
            }
    return states


def collect(experiment: Experiment, outputs_root: Path) -> dict:
    """Per-seed paired differences, split by censoring status."""
    magnitudes: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    directions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    exposure_counts = {"both_censored": 0, "one_censored": 0, "uncensored": 0}

    for seed in experiment.seeds:
        run_dir = outputs_root / f"{experiment.run_prefix}{seed}"
        states = onset_states(run_dir, experiment.analysis_target)
        exposures = sorted({exposure for exposure, _ in states})

        for variable in VARIABLES:
            for a, b in PAIRS:
                sized: list[float] = []
                signs: list[int] = []
                for exposure in exposures:
                    left = states.get((exposure, a))
                    right = states.get((exposure, b))
                    if left is None or right is None:
                        continue
                    if left[variable] is None or right[variable] is None:
                        continue
                    difference = left[variable] - right[variable]
                    if not left["censored"] and not right["censored"]:
                        sized.append(difference)
                        signs.append(1 if difference > 0 else -1 if difference < 0 else 0)
                    elif left["censored"] and right["censored"]:
                        continue
                    else:
                        # The censored agent sits at or beyond the starting gap,
                        # so the direction is known when the difference already
                        # points away from the censored side.
                        censored_is_left = left["censored"]
                        if censored_is_left and difference > 0:
                            signs.append(1)
                        elif not censored_is_left and difference < 0:
                            signs.append(-1)
                if sized:
                    magnitudes[(variable, a, b)].append(statistics.median(sized))
                if signs:
                    directions[(variable, a, b)].append(
                        1 if statistics.median(signs) > 0 else -1
                        if statistics.median(signs) < 0
                        else 0
                    )

        for exposure in exposures:
            flags = [
                states[(exposure, agent)]["censored"]
                for agent in AGENTS
                if (exposure, agent) in states
            ]
            if all(flags):
                exposure_counts["both_censored"] += 1
            elif any(flags):
                exposure_counts["one_censored"] += 1
            else:
                exposure_counts["uncensored"] += 1

    return {
        "magnitudes": magnitudes,
        "directions": directions,
        "exposure_counts": exposure_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/onset_state_analysis")
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        collected = collect(experiment, args.outputs_root)
        counts = collected["exposure_counts"]
        total = sum(counts.values())
        print(
            f"\n=== {experiment.key} ({experiment.analysis_target}) ===\n"
            f"exposures: uncensored {counts['uncensored']}, one agent censored "
            f"{counts['one_censored']}, all censored {counts['both_censored']}, "
            f"total {total}"
        )
        print(
            f"{'variable':<9}{'contrast':<11}{'median':>9}{'IQR':>18}"
            f"{'CI':>18}{'n':>4}{'sign':>10}{'p':>9}"
        )
        for variable in VARIABLES:
            for a, b in PAIRS:
                sizes = collected["magnitudes"][(variable, a, b)]
                signs = collected["directions"][(variable, a, b)]
                q1, median, q3 = quartiles(sizes)
                low, high = bootstrap_median_ci(sizes)
                positive = sum(1 for value in signs if value > 0)
                negative = sum(1 for value in signs if value < 0)
                p = sign_test_p(positive, negative)
                rows.append(
                    {
                        "experiment": experiment.key,
                        "variable": variable,
                        "contrast": f"{a} - {b}",
                        "n_seeds_sized": len(sizes),
                        "median_difference": round(median, 4),
                        "q1": round(q1, 4),
                        "q3": round(q3, 4),
                        "bootstrap_ci_low": round(low, 4),
                        "bootstrap_ci_high": round(high, 4),
                        "n_seeds_signed": len(signs),
                        "n_seeds_positive": positive,
                        "n_seeds_negative": negative,
                        "sign_test_p": round(p, 5),
                    }
                )
                print(
                    f"{variable:<9}{a + ' - ' + b:<11}{median:>9.2f}"
                    f"{f'[{q1:.2f}, {q3:.2f}]':>18}{f'[{low:.2f}, {high:.2f}]':>18}"
                    f"{len(sizes):>4}{f'+{positive}/-{negative}':>10}{p:>9.4f}"
                )

    write_csv(args.out / "onset_state_contrasts.csv", list(rows[0].keys()), rows)
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
