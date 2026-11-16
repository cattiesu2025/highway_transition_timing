"""Equal-start training sweep: 20 new jobs, A/C x SP/FD x 5 paired seeds.

BAL (0.70, 0.70) is reused from the existing F5/BAL condition records and is never
trained here. Print commands unless --execute.
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

BASE_SEEDS = {"A": 3100, "C": 4200}
# Frozen from outputs/reward_equal_start/equal_start_weights.json at 6 decimals.
WEIGHTS = {
    "A": {"SP": (1.068978, 0.534489), "FD": (0.414118, 0.828237)},
    "C": {"SP": (1.078515, 0.539258), "FD": (0.411300, 0.822601)},
}
MANIFEST = HERE / "jobs.json"


def jobs() -> list[dict]:
    rows = [
        dict(arm=arm, variant=variant, agent=variant, seed=seed,
             speed_weight=WEIGHTS[arm][variant][0], front_distance_weight=WEIGHTS[arm][variant][1])
        for arm, base in BASE_SEEDS.items()
        for variant in ("SP", "FD")
        for seed in range(base, base + 5)
    ]
    return [dict(index=i, **row) for i, row in enumerate(rows)]


def run_directory(root, job) -> Path:
    return Path(root) / f"reward_equal_start_arm{job['arm']}_{job['variant']}_100k_seed{job['seed']}"


def commands(job, root) -> list[list[str]]:
    return calibrate.load_sweep().commands(job, root, "train", run_dir=run_directory(root, job))


def main(argv=None) -> int:
    tasks = jobs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(len(tasks)))
    parser.add_argument("--out-root", type=Path, default=calibrate.ROOT / "outputs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Export the jobs; no execution required")
    args = parser.parse_args(argv)
    if args.manifest:
        args.manifest.write_text(json.dumps(tasks, indent=2) + "\n")
    if args.index is None:
        if args.execute:
            parser.error("--execute requires --index")
        return 0
    job = tasks[args.index]
    root = args.out_root.resolve()
    calls = commands(job, root)
    if args.execute:
        if json.loads(MANIFEST.read_text()) != tasks:
            raise ValueError("Frozen equal-start manifest differs from the configured jobs")
        directory = run_directory(root, job)
        directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
        (directory / "sweep_job.json").write_text(json.dumps(job, indent=2) + "\n")
    for call in calls:
        print(shlex.join(call), flush=True)
        if args.execute:
            subprocess.run(call, cwd=calibrate.ROOT,
                           env=dict(os.environ, PYTHONPATH=str(calibrate.ROOT / "src")), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
