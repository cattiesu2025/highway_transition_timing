#!/usr/bin/env python3
"""Collect onsets for the t = 0 equal-start weight set: 4 arms x SP/BAL/FD x 5 paired seeds (60 runs).

- A, C: SP/FD from run.py (outputs/reward_equal_start_arm{A,C}_*); BAL reused from the main runs.
- B: SP/FD from run_bd.py (pulled into outputs/reward_equal_start/bd_runs); BAL reused from the main runs.
- D: SP/FD/BAL from run_bd.py (ES-*); ES-BAL is the copy pulled with the first-log runs.

Weights, training length and scene pairing are checked as in collect_first_log.py, and the
onset definitions are that module's: A = ego slowdown latency, B/C/D = first physical lane change.
Only runs with a selected, eligible checkpoint contribute onsets.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect_first_log as fl  # noqa: E402
import run as run_ac  # noqa: E402
import run_bd  # noqa: E402

BD_PULLED = fl.OUTPUTS / "reward_equal_start/bd_runs"
BAL_DIRS = dict(fl.BAL_DIRS)  # A/B/C main runs; D ES-BAL pulled with the first-log runs


def tasks() -> list[tuple[dict, Path]]:
    out = [({k: j[k] for k in fl.KEYS}, run_ac.run_directory(fl.OUTPUTS, j)) for j in run_ac.jobs()]
    for j in run_bd.jobs():
        if j["variant"] == "BAL":
            continue  # D BAL is the shared ES-BAL entry below
        out.append(({k: j[k] for k in fl.KEYS}, run_bd.run_directory(BD_PULLED, j)))
    for arm, base in fl.SEEDS.items():
        for seed in range(base, base + 5):
            out.append((dict(arm=arm, variant="BAL", agent="BAL", seed=seed, speed_weight=0.7,
                             front_distance_weight=0.7), Path(str(BAL_DIRS[arm]).format(seed=seed))))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fl.OUTPUTS / "reward_equal_start/analysis_t0_onset")
    args = parser.parse_args(argv)
    task_list = tasks()
    if len(task_list) != 60:
        raise ValueError(f"Expected 60 runs, found {len(task_list)}")
    paired = {(j["arm"], j["seed"]): fl.scenes(d, "BAL") for j, d in task_list if j["variant"] == "BAL"}
    status, episodes, summary = [], [], []
    for job, directory in task_list:
        meta = [r for r in fl.rows(directory / "training_runs.csv") if r["agent_condition"] == job["agent"]]
        if len(meta) != 1:
            raise ValueError(f"Expected one training record: {directory}")
        for field, value in (("reward_speed_score", job["speed_weight"]),
                             ("reward_front_distance_score", job["front_distance_weight"])):
            if not math.isclose(float(meta[0][field]), value, abs_tol=1e-9):
                raise ValueError(f"{directory}: {field}={meta[0][field]} differs from {value}")
        if int(meta[0]["total_timesteps"]) != 100000 or meta[0]["dqn_variant"] != "double-dqn":
            raise ValueError(f"Incompatible training protocol: {directory}")
        if job["variant"] != "BAL":
            current, base = fl.scenes(directory, job["agent"]), paired[(job["arm"], job["seed"])]
            n = min(len(current), len(base))
            if not n or current[:n] != base[:n]:
                raise ValueError(f"Training scene sequence differs from paired BAL: {directory}")
        ok, detail = fl.selected(job, directory)
        status.append(dict(**job, run=directory.name, selection_status="complete" if ok else "failed", detail=detail))
        if ok:
            eps = fl.onsets(job, directory)
            episodes += eps
            valid = [float(e["onset_seconds"]) for e in eps if e["episode_outcome"] == "valid_onset"]
            summary.append(dict(**job, n_total=36, n_onset=len(valid),
                                n_collision=sum(e["collision"] for e in eps),
                                median_onset_seconds=median(valid) if valid else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("run_status.csv", status), ("episode_onsets.csv", episodes), ("seed_summary.csv", summary)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    failed = [s["run"] for s in status if s["selection_status"] != "complete"]
    print(f"{len(status)} runs, {len(summary)} with onsets; selection failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
