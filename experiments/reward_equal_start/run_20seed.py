"""20-seed rerun of the t=0 (ES) and first-log (FL) equal-start designs. Print unless --execute.

Every model is trained fresh on 20 paired seeds per arm (A 3100-3119, B 4100-4119,
C/D 4200-4219); nothing from the 5-seed runs is reused.

- BAL (0.70, 0.70): one set per arm, shared by ES and FL (the same weights and seed
  give the same training run, as the 5-seed design already assumed).
- ES SP/FD, Arms A-D: t=0 weights, solved on the 2000-episode random-policy reference
  sets (seeds 9100/9200), which do not depend on training seeds. Same values as
  run.py (A/C) and run_bd.py (B/D).
- FL SP/FD, Arms A/B/D: recalibrated on the first 4 episodes of all 20 seeds
  (first_log_calibrate.py --seed-count 20 --tag _20seed).
- FL SP/FD, Arm C: optional block at the end. On 20 seeds Arm C's t=0 weights no longer
  equalise the first-log mean (SP 55.60, BAL 56.41, FD 57.04), unlike 5 seeds.

Arms A-C use the reward-sweep training command; Arm D uses the distance-variant runner
with only its preference weights replaced. Earlier runners and outputs are untouched.
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
SEEDS_PER_ARM = 20
BASE_SEEDS = {"A": 3100, "B": 4100, "C": 4200, "D": 4200}
BAL = (0.7, 0.7)
# t=0: frozen at 6 decimals from equal_start_weights.json (A/C) and _BD.json (B/D).
ES_WEIGHTS = {
    "A": {"SP": (1.068978, 0.534489), "FD": (0.414118, 0.828237)},
    "B": {"SP": (1.101202, 0.550601), "FD": (0.404937, 0.809875)},
    "C": {"SP": (1.078515, 0.539258), "FD": (0.411300, 0.822601)},
    "D": {"SP": (1.092444, 0.546222), "FD": (0.407339, 0.814678)},
}
# First-log: frozen at 6 decimals from outputs/reward_equal_start/first_log_weights_20seed.json.
FL_WEIGHTS = {
    "A": {"SP": (1.047241, 0.523620), "FD": (0.420887, 0.841774)},
    "B": {"SP": (1.099835, 0.549917), "FD": (0.405308, 0.810616)},
    "C": {"SP": (1.090537, 0.545269), "FD": (0.407871, 0.815742)},
    "D": {"SP": (1.106991, 0.553496), "FD": (0.403386, 0.806772)},
}
FL_ARMS = ("A", "B", "D")
OPTIONAL_FL_ARMS = ("C",)  # Last block; submitted only on request.
MANIFEST = HERE / "jobs_20seed.json"


def jobs() -> list[dict]:
    def block(design, arm, variant, weights):
        base = BASE_SEEDS[arm]
        return [dict(design=design, arm=arm, variant=variant, agent=variant, seed=seed,
                     speed_weight=weights[0], front_distance_weight=weights[1])
                for seed in range(base, base + SEEDS_PER_ARM)]

    rows = []
    for arm in BASE_SEEDS:
        rows += block("BAL", arm, "BAL", BAL)
    for arm in BASE_SEEDS:
        for variant in ("SP", "FD"):
            rows += block("ES", arm, variant, ES_WEIGHTS[arm][variant])
    for arm in FL_ARMS + OPTIONAL_FL_ARMS:
        for variant in ("SP", "FD"):
            rows += block("FL", arm, variant, FL_WEIGHTS[arm][variant])
    return [dict(index=i, **row) for i, row in enumerate(rows)]


def label(job: dict) -> str:
    return "BAL" if job["design"] == "BAL" else f"{job['design']}-{job['variant']}"


def run_directory(root: Path, job: dict) -> Path:
    if job["arm"] == "D":
        # Named by the distance-variant runner from its variant label.
        return Path(root) / f"arm_c_distance-affine-v1_R20-{label(job)}_100k_seed{job['seed']}"
    return Path(root) / f"reward_equal20_arm{job['arm']}_{label(job)}_100k_seed{job['seed']}"


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
    spec = importlib.util.spec_from_file_location("equal20_arm_d_runner", spec_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    # Replace only this agent's preference weights; everything else is the runner's own protocol.
    runner.WEIGHTS = {job["agent"]: (f"R20-{label(job)}", job["speed_weight"], job["front_distance_weight"])}
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
        raise ValueError("Frozen 20-seed manifest differs from the configured jobs")
    print(json.dumps(job), flush=True)
    if job["arm"] == "D":
        run_arm_d(job, root, args.execute, args.arm_d_mode)
    else:
        run_sweep_arm(job, root, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
