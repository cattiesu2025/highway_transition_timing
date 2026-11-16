"""Score-normalisation pilot (SN1). Prints the job unless --execute.

Candidate reward scores, both tied to environment ranges instead of hand-set constants:

    speed_score          = clip((v - 10) / 25, 0, 1)   action target-speed range 10-35 m/s
    front_distance_score = clip(d / 200, 0, 1)         highway-env PERCEPTION_DISTANCE, 1 if no front

Maintained scores are clip((v - 20) / 15) and clip(d / 120). Both slopes shrink by exactly
0.6, so preference weights are scaled by 1/0.6 (BAL 0.70 -> 1.166667) to keep the fixed
collision, risk, lane-change and slow-down penalties at their maintained relative strength.
SP/FD keep the 2:1 weight ratio and are solved so the seed-mean first logged training
reward equals BAL's (first-log equal start, 20 training seeds per arm).

Only this process and the training subprocess it starts are adapted: ``install`` wraps
``highway_adapter.extract_diagnostics`` in memory. Shared source defaults are unchanged.
Arms A-C reuse the reward-sweep training command; Arm D reuses the distance-variant runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

VERSION = "SN1"
SPEED_RANGE = (10.0, 35.0)
DISTANCE_HORIZON = 200.0
PREFERENCE_SCALE = 1 / 0.6  # Both score slopes shrink by 0.6: 15/25 and 120/200.
BAL = (1.166667, 1.166667)
# Seed-mean S and D per episode over the first 4 episodes of 20 training seeds, new scores.
FIRST_LOG_SCORES = {
    "A": (54.389653, 65.519350),
    "B": (54.947834, 76.208518),
    "C": (55.740701, 59.547111),
    "D": (50.197628, 65.052158),
}
# Frozen at 6 decimals from outputs/score_normalization/calibration/first_log_weights.json.
WEIGHTS = {
    "A": {"SP": (1.605220, 0.802610), "FD": (0.754436, 1.508872)},
    "B": {"SP": (1.644409, 0.822205), "FD": (0.737906, 1.475812)},
    "C": {"SP": (1.572866, 0.786433), "FD": (0.769311, 1.538622)},
    "D": {"SP": (1.625388, 0.812694), "FD": (0.745738, 1.491477)},
}
# First two seeds of each arm's 20-seed block, paired with the maintained-score runs.
SEEDS = {"A": (3100, 3101), "B": (4100, 4101), "C": (4200, 4201), "D": (4200, 4201)}
VARIANTS = ("BAL", "SP", "FD")
MANIFEST = HERE / "jobs.json"
SMOKE = {"--timesteps": "1200", "--checkpoint-every": "600", "--num-exposures": "3",
         "--evaluation-duration": "4", "--bootstrap-samples": "0"}


def speed_score(speed: float) -> float:
    low, high = SPEED_RANGE
    return min(1.0, max(0.0, (float(speed) - low) / (high - low)))


def front_distance_score(distance: float) -> float:
    distance = float(distance)
    return 1.0 if math.isinf(distance) else min(1.0, max(0.0, distance / DISTANCE_HORIZON))


def install() -> None:
    """Replace the two scores in this process. Call before loading an experiment module."""
    from highway_transition_timing import highway_adapter as adapter

    if hasattr(adapter.extract_diagnostics, "maintained"):
        return
    maintained = adapter.extract_diagnostics

    def extract_diagnostics(env):
        result = maintained(env)
        result["speed_score"] = speed_score(result["ego_speed"])
        result["front_distance_score"] = front_distance_score(result["nearest_front_distance"])
        return result

    extract_diagnostics.maintained = maintained
    adapter.extract_diagnostics = extract_diagnostics


def uninstall() -> None:
    from highway_transition_timing import highway_adapter as adapter

    adapter.extract_diagnostics = getattr(adapter.extract_diagnostics, "maintained", adapter.extract_diagnostics)


def jobs() -> list[dict]:
    rows = []
    for arm, seeds in SEEDS.items():
        for variant in VARIANTS:
            speed, spacing = BAL if variant == "BAL" else WEIGHTS[arm][variant]
            rows += [dict(arm=arm, variant=variant, agent=variant, seed=seed,
                          speed_weight=speed, front_distance_weight=spacing) for seed in seeds]
    return [dict(index=i, **row) for i, row in enumerate(rows)]


def protocol(job: dict, smoke: bool) -> dict:
    return dict(experiment="score_normalization", version=VERSION, smoke=smoke, job=job,
                speed_score=f"clip((v - {SPEED_RANGE[0]:g}) / {SPEED_RANGE[1] - SPEED_RANGE[0]:g}, 0, 1)",
                front_distance_score=f"clip(d / {DISTANCE_HORIZON:g}, 0, 1); 1 if no front vehicle",
                maintained_scores="clip((v - 20) / 15, 0, 1); clip(d / 120, 0, 1)",
                preference_scale=PREFERENCE_SCALE, equal_start="first_log_20seed",
                penalties="maintained: collision 2, risk 3, arm-specific action costs")


def run_directory(root: Path, job: dict, smoke: bool = False) -> Path:
    if job["arm"] == "D":  # Named by the distance-variant runner from its variant label.
        name = f"arm_c_distance-affine-v1_{VERSION}-{job['variant']}_100k_seed{job['seed']}"
    else:
        name = f"score_norm_{VERSION}_arm{job['arm']}_{job['variant']}_100k_seed{job['seed']}"
    return Path(root) / (name + ("_smoke" if smoke else ""))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sweep_commands(job: dict, root: Path, smoke: bool) -> list[list[str]]:
    """The maintained training command, started through this file's ``entry``."""
    sweep = load("score_norm_reward_sweep", ROOT / "experiments/reward_sensitivity/run.py")
    calls = []
    for call in sweep.commands(job, root, "train", run_dir=run_directory(root, job, smoke)):
        call = [call[0], str(Path(__file__).resolve()), "entry", *call[1:]]
        if smoke:
            for flag, value in SMOKE.items():
                call[call.index(flag) + 1] = value
        calls.append(call)
    return calls


def run_sweep_arm(job: dict, root: Path, execute: bool, smoke: bool) -> None:
    calls = sweep_commands(job, root, smoke)
    if execute:
        directory = run_directory(root, job, smoke)
        directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
        (directory / "score_normalization.json").write_text(json.dumps(protocol(job, smoke), indent=2) + "\n")
    for call in calls:
        print(shlex.join(call), flush=True)
        if execute:
            subprocess.run(call, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), check=True)


def run_arm_d(job: dict, root: Path, execute: bool, smoke: bool) -> None:
    install()
    runner = load("score_norm_arm_d_runner", ROOT / "experiments/arm_c_distance_variant/run.py")
    # Replace only this agent's preference weights; everything else is the runner's own protocol.
    runner.WEIGHTS = {job["agent"]: (f"{VERSION}-{job['variant']}", job["speed_weight"], job["front_distance_weight"])}
    args = argparse.Namespace(agent=job["agent"], seed=job["seed"], mode="smoke" if smoke else "train",
                              out_root=root, baseline_root=root, execute=execute)
    try:
        runner.run(args)
    finally:
        directory = run_directory(root, job, smoke)
        if execute and directory.is_dir():
            (directory / "score_normalization.json").write_text(json.dumps(protocol(job, smoke), indent=2) + "\n")


def entry(argv: list[str]) -> None:
    """Run a maintained experiment script with the candidate scores installed."""
    install()
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["entry"]:
        entry(argv[1:])
        return 0
    tasks = jobs()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, choices=range(len(tasks)))
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Export the jobs; no execution required")
    parser.add_argument("--smoke", action="store_true", help="1200 training steps; a local check, never a scientific run")
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
        raise ValueError("Frozen manifest differs from the configured jobs")
    print(json.dumps(protocol(job, args.smoke)), flush=True)
    if job["arm"] == "D":
        run_arm_d(job, root, args.execute, args.smoke)
    else:
        run_sweep_arm(job, root, args.execute, args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
