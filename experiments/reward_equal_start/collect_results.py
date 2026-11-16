#!/usr/bin/env python3
"""Validate and collect SP/BAL/FD equal-start runs: 20 new plus 10 reused BAL."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The existing collector imports its own sibling ``run``; keep that lookup first.
sys.path.insert(0, str(ROOT / "experiments" / "reward_sensitivity"))
import collect as collector  # noqa: E402


def load_sweep():
    spec = importlib.util.spec_from_file_location("equal_start_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tasks(root: Path) -> list[tuple[dict, Path]]:
    sweep = load_sweep()
    new = [({k: job[k] for k in ("arm", "variant", "agent", "seed", "speed_weight", "front_distance_weight")},
            sweep.run_directory(root, job)) for job in sweep.jobs()]
    bal = [(dict(arm=arm, variant="BAL", agent="BAL", seed=seed, speed_weight=0.7, front_distance_weight=0.7),
            root / f"{collector.PREFIX[arm]}_100k_seed{seed}")
           for arm, base in sweep.BASE_SEEDS.items() for seed in range(base, base + 5)]
    return new + bal


def schedule(path: Path, agent: str) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "agent_condition"}
            for row in collector.rows(path) if row["agent_condition"] == agent]


def verify_pairing(task_list) -> None:
    """Every new run must use its paired BAL run's training scene sequence."""
    baseline = {(job["arm"], job["seed"]): schedule(path / "training_scenarios.csv", "BAL")
                for job, path in task_list if job["variant"] == "BAL"}
    for job, path in task_list:
        if job["variant"] == "BAL":
            continue
        current = schedule(path / "training_scenarios.csv", job["agent"])
        paired = baseline[(job["arm"], job["seed"])]
        n = min(len(current), len(paired))
        if not n or current[:n] != paired[:n]:
            raise ValueError(f"Training scene sequence differs from paired BAL: {path}")
        if json.loads((path / "sweep_job.json").read_text())["speed_weight"] != job["speed_weight"]:
            raise ValueError(f"Run manifest differs: {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "reward_equal_start" / "analysis")
    args = parser.parse_args(argv)
    task_list = tasks(args.root)
    verify_pairing(task_list)
    collector.collect(args.root, args.out, tasks=task_list)
    print(f"Collected {len(task_list)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
