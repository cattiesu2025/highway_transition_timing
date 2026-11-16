"""Equal-start extension to Arms B and D: 25 new jobs. Print commands unless --execute.

- Arm B: SP/FD x seeds 4100-4104 through the same sweep training command as A/C.
  BAL (0.70, 0.70) is reused from multilane_twolane_open_fixed_100k_seed410[0-4].
- Arm D: SP/FD/BAL x seeds 4200-4204 through the distance-variant runner, with
  only its preference weights replaced. No (0.70, 0.70) Arm D model exists, so
  BAL is trained here as well.

The A/C runner (run.py) and its frozen manifest are left unchanged.
"""
from __future__ import annotations

import argparse
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
# Frozen at 6 decimals from outputs/reward_equal_start/equal_start_weights_BD.json,
# solved on 2000 random-policy episodes per arm (reference seed 9200).
WEIGHTS = {
    "B": {"SP": (1.101202, 0.550601), "FD": (0.404937, 0.809875)},
    "D": {"SP": (1.092444, 0.546222), "FD": (0.407339, 0.814678), "BAL": (0.7, 0.7)},
}
BASE_SEEDS = {"B": 4100, "D": 4200}
MANIFEST = HERE / "jobs_bd.json"


def jobs() -> list[dict]:
    rows = [
        dict(arm=arm, variant=variant, agent=variant, seed=seed,
             speed_weight=weights[0], front_distance_weight=weights[1])
        for arm, table in WEIGHTS.items()
        for variant, weights in table.items()
        for seed in range(BASE_SEEDS[arm], BASE_SEEDS[arm] + 5)
    ]
    return [dict(index=i, **row) for i, row in enumerate(rows)]


def run_directory(root: Path, job: dict) -> Path:
    if job["arm"] == "B":
        return Path(root) / f"reward_equal_start_armB_{job['variant']}_100k_seed{job['seed']}"
    # Named by the distance-variant runner from its variant label.
    return Path(root) / f"arm_c_distance-affine-v1_ES-{job['variant']}_100k_seed{job['seed']}"


def run_arm_b(job: dict, root: Path, execute: bool) -> None:
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
    import importlib.util
    spec = importlib.util.spec_from_file_location("equal_start_arm_d_runner", spec_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    # Replace only this agent's preference weights; everything else is the runner's own protocol.
    runner.WEIGHTS = {job["agent"]: (f"ES-{job['variant']}", job["speed_weight"], job["front_distance_weight"])}
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
        raise ValueError("Frozen B/D equal-start manifest differs from the configured jobs")
    print(json.dumps(job), flush=True)
    if job["arm"] == "B":
        run_arm_b(job, root, args.execute)
    else:
        run_arm_d(job, root, args.execute, args.arm_d_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
