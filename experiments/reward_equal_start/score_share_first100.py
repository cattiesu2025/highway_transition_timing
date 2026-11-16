#!/usr/bin/env python3
"""Unweighted speed and front-distance scores of BAL (0.70, 0.70) policies, first 100 evaluation steps.

Per episode: sum of the reward-component speed_score and front_distance_score over steps t < 100
(20 s at 5 Hz; each score is clipped to [0, 1] per step, so the maximum is 100). Then mean over the
36 development scenes of a seed. Arms and sources:
- A/B/C: the 20 main BAL runs per arm (seeds 3100-3119, 4100-4119, 4200-4219).
- D: the only (0.70, 0.70) Arm D policies, ES-BAL seeds 4200-4204 (5 seeds). As a check, the
  original Arm D F4 (0.56, 0.84) runs are summarised separately; they are not plotted.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SOURCES = {
    "A": "single_lane_slow_front_fixed_100k_seed*",
    "B": "multilane_twolane_open_fixed_100k_seed*",
    "C": "multilane_twolane_occupied_fixed_100k_seed*",
    "D": "reward_equal_start/first_log_runs/arm_c_distance-affine-v1_ES-BAL_100k_seed*",
}
CHECK_D_F4 = "arm_c_distance_figure_data/arm_c_distance-affine-v1_F4_100k_seed*"
EXPECTED_SEEDS = {"A": 20, "B": 20, "C": 20, "D": 5}
STEPS = 100


def seed_scores(steps_csv: Path) -> dict:
    csv.field_size_limit(10**9)
    episodes: dict[str, list[float]] = {}
    with steps_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["agent_condition"] != "BAL" or int(row["t"]) >= STEPS:
                continue
            parts = json.loads(row["reward_components"])
            e = episodes.setdefault(row["episode_id"], [0.0, 0.0, 0])
            e[0] += parts["speed_score"]; e[1] += parts["front_distance_score"]; e[2] += 1
    if len(episodes) != 36 or any(e[2] != STEPS for e in episodes.values()):
        raise ValueError(f"Expected 36 episodes of {STEPS} steps: {steps_csv}")
    return dict(episodes=36, S=mean(e[0] for e in episodes.values()), D=mean(e[1] for e in episodes.values()))


def collect(pattern: str, arm: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(OUT / pattern / "evaluation/steps.csv"))):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        rows.append(dict(arm=arm, seed=seed, **seed_scores(Path(path))))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "reward_equal_start/score_share/source_data")
    args = parser.parse_args(argv)
    rows = []
    for arm, pattern in SOURCES.items():
        found = collect(pattern, arm)
        if len(found) != EXPECTED_SEEDS[arm]:
            raise ValueError(f"Arm {arm}: expected {EXPECTED_SEEDS[arm]} seeds, found {len(found)}")
        rows += found
    check = collect(CHECK_D_F4, "D_F4")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("score_first100_by_seed.csv", rows), ("check_armD_F4_by_seed.csv", check)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    for arm in [*SOURCES, "D_F4"]:
        sel = [r for r in rows + check if r["arm"] == arm]
        S, D = mean(r["S"] for r in sel), mean(r["D"] for r in sel)
        print(f"{arm}: seeds {len(sel)}  S {S:.1f}  D {D:.1f}  speed share {S / (S + D):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
