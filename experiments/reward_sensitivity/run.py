"""Reproducible development-only reward sweep; print commands unless --execute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SAMPLING_SEED = 20261006
VARIANTS = {
    "R1": ("FD", 0.33, 0.51), "R2": ("FD", 0.42, 0.98),
    "R3": ("BAL", 0.74, 0.92), "R4": ("BAL", 0.88, 0.96),
    "R5": ("SP", 1.17, 0.76), "R6": ("SP", 1.03, 0.48),
}
BASE_SEEDS = {"A": 3100, "B": 4100, "C": 4200}


def reproduce_sample():
    rng = random.Random(SAMPLING_SEED)
    groups = {"FD": [], "BAL": [], "SP": []}
    while any(len(values) < 2 for values in groups.values()):
        speed, distance = (round(rng.uniform(0.25, 1.25), 2) for _ in range(2))
        ratio = distance / speed
        group = "FD" if ratio > 1.5 else "SP" if ratio < 2 / 3 else "BAL"
        if len(groups[group]) < 2:
            groups[group].append((group, speed, distance))
    return {f"R{i + 1}": row for i, row in enumerate(sum(groups.values(), []))}


def jobs():
    return [dict(index=i, arm=arm, variant=variant, agent=values[0],
                 speed_weight=values[1], front_distance_weight=values[2], seed=seed)
            for i, (arm, variant, values, seed) in enumerate(
                (arm, variant, values, base + offset)
                for arm, base in BASE_SEEDS.items()
                for variant, values in VARIANTS.items() for offset in range(5))]


def run_directory(output_root, job):
    return Path(output_root) / f"reward_sensitivity_arm{job['arm']}_{job['variant']}_100k_seed{job['seed']}"


def commands(job, output_root, phase):
    run_dir = run_directory(output_root, job)
    script = "single_lane_slow_front" if job["arm"] == "A" else "multilane_open_lane_change"
    entry = [sys.executable, str(ROOT / "experiments" / script / "run.py")]
    weights = ["--speed-weight", str(job["speed_weight"]), "--front-distance-weight", str(job["front_distance_weight"])]
    common = ["--agents", job["agent"], "--seed", str(job["seed"]), "--duration", "20",
              "--evaluation-duration", "120", "--num-exposures", "36", "--collision-risk-penalty", "3.0"]
    if phase == "train":
        args = entry + ["--out", str(run_dir)] + common + weights + [
            "--timesteps", "100000", "--checkpoint-every", "5000", "--learning-rate", "0.0005",
            "--training-scenario-profile", "stratified", "--near-matched-speed-delta", "2.0",
            "--bootstrap-samples", "500", "--verbose", "0", "--no-figures"]
        if job["arm"] != "A":
            args += ["--lanes-count", "2", "--ego-lane", "1", "--lane-change-penalty", "0.2", "--slow-down-penalty", "0.2"]
            if job["arm"] == "C":
                args += ["--target-lane-vehicle"]
        return [args]
    variants = ["original", "no-front", "matched-speed-front", "far-front"]
    if job["arm"] == "A":
        args = ["--run-dir", str(run_dir)] + common + weights + ["--counterfactual-variants"] + variants
        return [entry + ["rollout-counterfactual"] + args + ["--bootstrap-samples", "500", "--no-figures"],
                entry + ["counterfactual"] + args]
    return [[sys.executable, str(ROOT / "experiments" / script / "rollout_counterfactual.py"),
             "--run-dir", str(run_dir), "--agents", job["agent"], "--num-exposures", "36",
             "--evaluation-duration", "120", "--eval-grid", "development", "--variants"] + variants]


def main(argv=None, *, task_list=None):
    tasks = jobs() if task_list is None else task_list
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(len(tasks)))
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Write the task manifest as JSON.")
    args = parser.parse_args(argv)
    if reproduce_sample() != VARIANTS:
        raise RuntimeError("Random sample does not match frozen variants")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(tasks, indent=2) + "\n")
    if args.index is None:
        if args.execute:
            parser.error("--execute requires --index")
        return 0
    job = tasks[args.index]
    output_root = args.out_root.resolve()
    run_dir = run_directory(output_root, job)
    calls = commands(job, output_root, args.phase)
    if args.execute:
        if args.phase == "train":
            # mkdir is exclusive: no overwriting completed or partially completed jobs.
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "sweep_job.json").write_text(json.dumps(job, indent=2) + "\n")
        else:
            if json.loads((run_dir / "sweep_job.json").read_text()) != job:
                raise ValueError("Run manifest does not match requested task")
            if not (run_dir / "models" / f"{job['agent']}_main.zip").exists():
                raise FileNotFoundError("Selected model is missing")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for call in calls:
        print(shlex.join(call), flush=True)
        if args.execute:
            subprocess.run(call, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
