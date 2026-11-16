#!/usr/bin/env python3
"""Collect equal-start training curves for Arms B and D (25 new runs + 5 reused Arm B BAL).

Training-return curves only: reads each run's progress.csv after checking that
- the configured reward weights match the frozen manifest (training_runs.csv),
- training reached at least 99k steps,
- the training scene sequence matches the paired BAL run of the same arm and seed.
Model-selection status is recorded but does not exclude a run: a run whose
checkpoints all fail the eligibility gate still trained under the same protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_bd  # noqa: E402

B_BAL = "multilane_twolane_open_fixed_100k_seed{seed}"


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tasks(root: Path) -> list[tuple[dict, Path]]:
    keys = ("arm", "variant", "agent", "seed", "speed_weight", "front_distance_weight")
    new = [({k: job[k] for k in keys}, run_bd.run_directory(root, job)) for job in run_bd.jobs()]
    bal = [(dict(arm="B", variant="BAL", agent="BAL", seed=seed, speed_weight=0.7, front_distance_weight=0.7),
            root / B_BAL.format(seed=seed)) for seed in range(4100, 4105)]
    return new + bal


def scenes(path: Path, agent: str) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "agent_condition"}
            for row in rows(path / "training_scenarios.csv") if row["agent_condition"] == agent]


def selection_status(job: dict, directory: Path) -> tuple[str, str]:
    status_file = directory / "status.json"
    if status_file.exists():  # Arm D runner writes an explicit status.
        status = json.loads(status_file.read_text())
        return status["status"], status.get("error", "")
    selected = [r for r in rows(directory / "model_selection.csv")
                if r["agent_condition"] == job["agent"] and r["selected"].lower() == "true"]
    return ("complete", f"selected {selected[0]['checkpoint_step']}") if len(selected) == 1 else ("failed", "no selection")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/reward_equal_start/bd_runs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reward_equal_start/analysis_bd")
    args = parser.parse_args(argv)
    task_list = tasks(args.root)
    if len(task_list) != 30:
        raise ValueError("Expected 30 B/D runs")
    paired = {(job["arm"], job["seed"]): scenes(path, "BAL") for job, path in task_list if job["variant"] == "BAL"}
    progress, status = [], []
    for job, directory in task_list:
        meta = [r for r in rows(directory / "training_runs.csv") if r["agent_condition"] == job["agent"]]
        if len(meta) != 1:
            raise ValueError(f"Expected one training record for {job['agent']}: {directory}")
        meta = meta[0]
        for field, value in (("reward_speed_score", job["speed_weight"]),
                             ("reward_front_distance_score", job["front_distance_weight"])):
            if not math.isclose(float(meta[field]), value, abs_tol=1e-9):
                raise ValueError(f"{directory}: {field}={meta[field]} differs from manifest {value}")
        if int(meta["total_timesteps"]) != 100000 or meta["dqn_variant"] != "double-dqn":
            raise ValueError(f"Incompatible training protocol: {directory}")
        if job["variant"] != "BAL":
            current, base = scenes(directory, job["agent"]), paired[(job["arm"], job["seed"])]
            n = min(len(current), len(base))
            if not n or current[:n] != base[:n]:
                raise ValueError(f"Training scene sequence differs from paired BAL: {directory}")
        trace = []
        for row in rows(directory / "training_logs" / job["agent"] / "progress.csv"):
            if row.get("rollout/ep_rew_mean"):
                reward = float(row["rollout/ep_rew_mean"])
                if not math.isfinite(reward):
                    raise ValueError(f"Nonfinite training reward: {directory}")
                trace.append(dict(**job, timesteps=float(row["time/total_timesteps"]), episode_reward=reward))
        if not trace or trace[-1]["timesteps"] < 99000:
            raise ValueError(f"Incomplete training: {directory}")
        progress.extend(trace)
        state, detail = selection_status(job, directory)
        status.append(dict(**job, run=directory.name, final_timesteps=trace[-1]["timesteps"],
                           selection_status=state, detail=detail))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("training_progress.csv", progress), ("run_status.csv", status)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    failed = [s["run"] for s in status if s["selection_status"] != "complete"]
    print(f"Collected {len(status)} runs, {len(progress)} progress rows into {args.out}; selection failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
