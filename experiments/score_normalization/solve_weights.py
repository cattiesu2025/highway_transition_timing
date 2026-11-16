"""Solve SN1 equal-start weights from the stored calibration sets (see calibrate_*.py).

t=0: 2000 random-policy episodes per arm. First-log: first 4 episodes of 20 training seeds.
BAL is 0.7 / 0.6; SP and FD keep a 2:1 weight ratio and match BAL's weighted S + D, so the
weight-independent penalty term cancels. run.py freezes the first-log values at 6 decimals.
"""
import csv
import json
from pathlib import Path

import numpy as np

CAL = Path(__file__).resolve().parents[2] / "outputs/score_normalization/calibration"
BAL = 0.7 / 0.6


def solve(S: float, D: float) -> dict:
    spacing = BAL * (S + D) / (2 * S + D)
    speed = BAL * (S + D) / (S + 2 * D)
    return {"S": S, "D": D, "D_over_S": D / S, "BAL": [BAL, BAL],
            "SP": [2 * spacing, spacing], "FD": [speed, 2 * speed]}


def main() -> None:
    result = {"scores": "S=clip((v-10)/25,0,1), D=clip(d/200,0,1)", "t0": {}, "first_log": {}}
    for arm in "ABCD":
        data = np.load(CAL / f"reference_arm{arm}.npz")
        episodes = len(data["lengths"])
        result["t0"][arm] = solve(float(np.clip((data["v"] - 10) / 25, 0, 1).sum() / episodes),
                                  float(np.clip(data["d"] / 200, 0, 1).sum() / episodes))
        with (CAL / f"first_log_arm{arm}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        result["first_log"][arm] = solve(sum(float(r["S_new"]) for r in rows) / len(rows),
                                         sum(float(r["D_new"]) for r in rows) / len(rows))
    (CAL / "weights.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
