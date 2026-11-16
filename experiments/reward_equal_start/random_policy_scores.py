#!/usr/bin/env python3
"""Random-policy (t=0) speed and front-distance scores for Arms B and D.

Extends the A/C reference set in ``calibrate.py`` to the other two arms without
changing it. Episodes come from the real training stack with learning never
starting, so actions are uniform random, exactly as before ``learning_starts``.

- Arm B: the selected-weight training command for B, parsed like A and C.
- Arm D: the distance-variant runner's own config and scene transform.

Both use reference seed 9200, the seed of the Arm C reference set. B, C and D
then see the same scene stream before their geometry differs. ``c`` is the
episode return minus 0.7 * (S + D) under BAL: the collision, risk, lane-change and
slow-down terms, whose coefficients are shared by SP, BAL and FD within an arm.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 9200


def load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibrate = load("equal_start_calibrate", ROOT / "experiments/reward_equal_start/calibrate.py")


def arm_d_config(_arm: str, seed: int):
    runner = load("random_scores_arm_d", ROOT / "experiments/arm_c_distance_variant/run.py")
    module = runner.load_base()
    config = runner.config_for(module, "BAL", seed)
    runner.install(module)
    return module, config


def collect(arm: str, episodes: int) -> list[dict]:
    if arm == "B":
        calibrate.ARMS["B"] = ("multilane_open_lane_change", SEED)
        calibrate.RESETS_BEFORE_MODEL["B"] = False
    elif arm == "D":
        # Arm D trains through the same multilane train_agents path as C.
        calibrate.RESETS_BEFORE_MODEL["D"] = False
        calibrate.training_config = arm_d_config
    rows = calibrate.collect(arm, SEED, episodes)
    return [{key: row[key] for key in ("arm", "seed", "episode", "scenario_type", "length", "S", "D", "c")}
            for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("B", "D"), required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reward_equal_start")
    args = parser.parse_args(argv)
    if args.episodes % calibrate.BLOCK_EPISODES:
        raise SystemExit(f"--episodes must be a multiple of {calibrate.BLOCK_EPISODES}")
    rows = collect(args.arm, args.episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"reference_episodes_arm{args.arm}.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    steps = sum(row["length"] for row in rows)
    print(json.dumps({"arm": args.arm, "seed": SEED, "episodes": len(rows),
                      "S_per_episode": sum(r["S"] for r in rows) / len(rows),
                      "D_per_episode": sum(r["D"] for r in rows) / len(rows),
                      "c_per_episode": sum(r["c"] for r in rows) / len(rows),
                      "S_per_step": sum(r["S"] for r in rows) / steps,
                      "D_per_step": sum(r["D"] for r in rows) / steps,
                      "mean_length": steps / len(rows), "output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
