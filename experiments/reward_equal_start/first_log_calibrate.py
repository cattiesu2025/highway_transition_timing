#!/usr/bin/env python3
"""First-log equal start: SP/FD weights whose first logged training reward equals BAL's.

SB3 writes its first ``rollout/ep_rew_mean`` after ``FIRST_LOG_EPISODES`` episodes
(log_interval=4), long before learning_starts=1000, so those episodes are uniform-random
and identical for every weight pair of a seed. This script replays exactly those
episodes for each training seed on the real training stack, decomposes each episode
return into c + w_speed * S + w_spacing * D, and solves the 2:1 SP/FD weights so that
the mean first-log reward over the seeds of an arm equals BAL (0.70, 0.70).

Before solving it checks the replay against the logged first point of every existing
SP/BAL/FD run of those seeds (same step, same value under that run's weights).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRST_LOG_EPISODES = 4  # SB3 log_interval used by every training run here.
BASE_SEEDS = {"A": 3100, "B": 4100, "C": 4200, "D": 4200}
SEEDS = {arm: range(base, base + 5) for arm, base in BASE_SEEDS.items() if arm != "C"}
PROGRESS = [ROOT / "outputs/reward_equal_start/analysis/training_progress.csv",
            ROOT / "outputs/reward_equal_start/analysis_bd/training_progress.csv"]
OUT = ROOT / "outputs/reward_equal_start"
TOLERANCE = 1e-3  # Logged rewards carry 5-8 decimals.


def load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def first_logs() -> dict[tuple[str, int, str], dict[str, float]]:
    """Earliest logged point of every run: {(arm, seed, variant): step, reward, weights}."""
    first: dict[tuple[str, int, str], dict[str, float]] = {}
    for path in PROGRESS:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["arm"], int(row["seed"]), row["variant"])
                step = float(row["timesteps"])
                if key not in first or step < first[key]["step"]:
                    first[key] = {"step": step, "reward": float(row["episode_reward"]),
                                  "speed": float(row["speed_weight"]),
                                  "spacing": float(row["front_distance_weight"])}
    return first


def replay_arm(arm: str) -> list[dict]:
    scores = load("equal_start_random_scores", ROOT / "experiments/reward_equal_start/random_policy_scores.py")
    calibrate = scores.calibrate
    original = calibrate.training_config
    try:
        if arm == "B":
            calibrate.ARMS["B"] = ("multilane_open_lane_change", 4100)
            calibrate.RESETS_BEFORE_MODEL["B"] = False
        elif arm == "D":
            calibrate.RESETS_BEFORE_MODEL["D"] = False
            calibrate.training_config = scores.arm_d_config
            # The runner's own BAL is (0.56, 0.84); collect() defines c under (0.70, 0.70).
            runner = load("random_scores_arm_d", ROOT / "experiments/arm_c_distance_variant/run.py")
            runner.WEIGHTS["BAL"] = ("ES-BAL", 0.7, 0.7)
        rows = []
        for seed in SEEDS[arm]:
            rows += calibrate.collect(arm, seed, FIRST_LOG_EPISODES)
        return rows
    finally:
        calibrate.training_config = original


def check_against_logs(arm: str, rows: list[dict], first: dict, *, require: bool = True) -> list[dict]:
    """Compare replay with every logged run of these seeds; seeds never trained are skipped
    unless ``require`` (the original 5-seed calibration, where every run must exist)."""
    checks = []
    for seed in SEEDS[arm]:
        episodes = [r for r in rows if r["seed"] == seed]
        assert len(episodes) == FIRST_LOG_EPISODES, (arm, seed, len(episodes))
        step = sum(r["length"] for r in episodes)
        for variant in ("SP", "BAL", "FD"):
            if not require and (arm, seed, variant) not in first:
                continue
            logged = first[(arm, seed, variant)]
            predicted = sum(r["c"] + logged["speed"] * r["S"] + logged["spacing"] * r["D"]
                            for r in episodes) / FIRST_LOG_EPISODES
            checks.append({"arm": arm, "seed": seed, "variant": variant, "step": step,
                           "logged_step": logged["step"], "predicted": predicted,
                           "logged": logged["reward"]})
            if step != logged["step"] or abs(predicted - logged["reward"]) > TOLERANCE:
                raise SystemExit(f"Replay does not match the training log: {checks[-1]}")
    return checks


def solve(rows: list[dict]) -> dict:
    n = len(rows)
    S = sum(r["S"] for r in rows) / n
    D = sum(r["D"] for r in rows) / n
    c = sum(r["c"] for r in rows) / n
    scores = load("equal_start_random_scores", ROOT / "experiments/reward_equal_start/random_policy_scores.py")
    weights = scores.calibrate.solve_weights(S, D)
    target = {name: c + w["speed"] * S + w["spacing"] * D for name, w in weights.items()}
    return {"episodes": n, "S": S, "D": D, "c": c, "weights": weights, "first_log_mean": target}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=sorted(BASE_SEEDS), default=None)
    parser.add_argument("--seed-count", type=int, default=5,
                        help="Seeds per arm from its base seed; 5 reproduces the frozen calibration")
    parser.add_argument("--tag", default="", help="Suffix for output files, e.g. _20seed")
    args = parser.parse_args(argv)
    arms = args.arms or sorted(SEEDS)
    original = args.seed_count == 5 and not args.tag
    if not original:
        SEEDS.clear()
        SEEDS.update({arm: range(BASE_SEEDS[arm], BASE_SEEDS[arm] + args.seed_count) for arm in arms})
    first = first_logs()
    result = {"first_log_episodes": FIRST_LOG_EPISODES, "bal_weight": 0.7, "arms": {}}
    episodes, checks = [], []
    for arm in arms:
        rows = replay_arm(arm)
        checks += check_against_logs(arm, rows, first, require=original)
        result["arms"][arm] = {"seeds": list(SEEDS[arm]), **solve(rows)}
        episodes += rows
        print(arm, json.dumps(result["arms"][arm]["weights"]), flush=True)
    (OUT / f"first_log_weights{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, table in ((f"first_log_episodes{args.tag}.csv", episodes), (f"first_log_replay_check{args.tag}.csv", checks)):
        if not table:
            continue
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
