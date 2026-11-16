"""First-log equal start: 30 new jobs, Arms A/B/D x SP/FD x 5 paired seeds.

Weights from first_log_calibrate.py make the seed-mean of the first logged training
reward equal across SP, BAL and FD. BAL (0.70, 0.70) runs are reused, not retrained:
- A: single_lane_slow_front_fixed_100k_seed310[0-4]
- B: multilane_twolane_open_fixed_100k_seed410[0-4]
- D: arm_c_distance-affine-v1_ES-BAL_100k_seed420[0-4]
Arm C keeps its 2000-episode equal-start runs: its first-log means already agree
within 0.1 (SP 52.9, BAL 52.8, FD 52.8).

A and B use the same sweep training command as run.py and run_bd.py; D uses the
distance-variant runner with only its preference weights replaced. Earlier runners,
manifests and outputs are left unchanged. Print commands unless --execute.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate  # noqa: E402

ROOT = calibrate.ROOT
# Frozen at 6 decimals from outputs/reward_equal_start/first_log_weights.json.
WEIGHTS = {
    "A": {"SP": (1.020106, 0.510053), "FD": (0.430083, 0.860165)},
    "B": {"SP": (1.091429, 0.545715), "FD": (0.407622, 0.815244)},
    "D": {"SP": (1.103066, 0.551533), "FD": (0.404435, 0.808870)},
}
BASE_SEEDS = {"A": 3100, "B": 4100, "D": 4200}
MANIFEST = HERE / "jobs_first_log.json"


def jobs() -> list[dict]:
    rows = [
        dict(arm=arm, variant=variant, agent=variant, seed=seed,
             speed_weight=WEIGHTS[arm][variant][0], front_distance_weight=WEIGHTS[arm][variant][1])
        for arm in BASE_SEEDS
        for variant in ("SP", "FD")
        for seed in range(BASE_SEEDS[arm], BASE_SEEDS[arm] + 5)
    ]
    return [dict(index=i, **row) for i, row in enumerate(rows)]


def run_directory(root: Path, job: dict) -> Path:
    if job["arm"] == "D":
        # Named by the distance-variant runner from its variant label.
        return Path(root) / f"arm_c_distance-affine-v1_FL-{job['variant']}_100k_seed{job['seed']}"
    return Path(root) / f"reward_first_log_arm{job['arm']}_{job['variant']}_100k_seed{job['seed']}"


def run_sweep_arm(job: dict, root: Path, execute: bool) -> None:
    calls = calibrate.load_sweep().commands(job, root, "train", run_dir=run_directory(root, job))
    if execute:
        directory = run_directory(root, job)
        directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
        (directory / "sweep_job.json").write_text(json.dumps(job, indent=2) + "\n")
    for call in calls:
        print(shlex.join(call), flush=True)
        if execute:
            subprocess.run(call, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), check=True)


def run_arm_d(job: dict, root: Path, execute: bool, mode: str = "train") -> None:
    spec_path = ROOT / "experiments/arm_c_distance_variant/run.py"
    spec = importlib.util.spec_from_file_location("first_log_arm_d_runner", spec_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    # Replace only this agent's preference weights; everything else is the runner's own protocol.
    runner.WEIGHTS = {job["agent"]: (f"FL-{job['variant']}", job["speed_weight"], job["front_distance_weight"])}
    args = argparse.Namespace(agent=job["agent"], seed=job["seed"], mode=mode, out_root=root,
                              baseline_root=root, execute=execute)
    try:
        runner.run(args)
    finally:
        directory = run_directory(root, job)
        if mode != "train":
            directory = directory.with_name(f"{directory.name}_{mode}")
        if execute and directory.is_dir():
            (directory / "sweep_job.json").write_text(json.dumps(job, indent=2) + "\n")


def main(argv=None) -> int:
    tasks = jobs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(len(tasks)))
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Export the jobs; no execution required")
    parser.add_argument("--arm-d-mode", choices=("train", "smoke", "audit"), default="train",
                        help="Arm D only; non-train modes are local checks, never scientific runs")
    args = parser.parse_args(argv)
    if args.manifest:
        args.manifest.write_text(json.dumps(tasks, indent=2) + "\n")
    if args.index is None:
        if args.execute:
            parser.error("--execute requires --index")
        return 0
    job = tasks[args.index]
    root = args.out_root.resolve()
    if args.execute and json.loads(MANIFEST.read_text()) != tasks:
        raise ValueError("Frozen first-log manifest differs from the configured jobs")
    print(json.dumps(job), flush=True)
    if job["arm"] == "D":
        run_arm_d(job, root, args.execute, args.arm_d_mode)
    else:
        run_sweep_arm(job, root, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
