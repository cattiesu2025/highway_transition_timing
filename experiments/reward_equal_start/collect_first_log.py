#!/usr/bin/env python3
"""Collect the first-log equal-start comparison: 4 arms x SP/BAL/FD x 5 paired seeds (60 runs).

- A, B, D: SP/FD are the 30 first-log runs (run_first_log.py); BAL (0.70, 0.70) is reused.
- C: SP/FD are the 2000-episode equal-start runs (run.py); their first-log seed means
  already agree with BAL within 0.1, so C was not retrained.

For every run it checks the configured weights, >= 99k training steps and, for SP/FD,
that the training scene sequence equals the paired BAL run's. Training curves include
every run. Onsets come only from runs with a selected, eligible checkpoint:
- A: the latency of the first lane-change action-defined onset (episode_outcomes.csv),
- B/C/D: the first physical lane change (actual_lane_change_summary.csv),
matching experiments/reward_sensitivity/collect.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run as run_ac  # noqa: E402
import run_first_log  # noqa: E402

OUTPUTS = ROOT / "outputs"
PULLED = OUTPUTS / "reward_equal_start/first_log_runs"
BAL_DIRS = {
    "A": OUTPUTS / "single_lane_slow_front_fixed_100k_seed{seed}",
    "B": OUTPUTS / "multilane_twolane_open_fixed_100k_seed{seed}",
    "C": OUTPUTS / "multilane_twolane_occupied_fixed_100k_seed{seed}",
    "D": PULLED / "arm_c_distance-affine-v1_ES-BAL_100k_seed{seed}",
}
SEEDS = {"A": 3100, "B": 4100, "C": 4200, "D": 4200}
KEYS = ("arm", "variant", "agent", "seed", "speed_weight", "front_distance_weight")


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tasks() -> list[tuple[dict, Path]]:
    out = []
    for job in run_first_log.jobs():
        out.append(({k: job[k] for k in KEYS}, run_first_log.run_directory(PULLED, job)))
    for job in run_ac.jobs():
        if job["arm"] == "C":
            out.append(({k: job[k] for k in KEYS}, run_ac.run_directory(OUTPUTS, job)))
    for arm, base in SEEDS.items():
        for seed in range(base, base + 5):
            out.append((dict(arm=arm, variant="BAL", agent="BAL", seed=seed, speed_weight=0.7,
                             front_distance_weight=0.7), Path(str(BAL_DIRS[arm]).format(seed=seed))))
    return out


def scenes(directory: Path, agent: str) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "agent_condition"}
            for r in rows(directory / "training_scenarios.csv") if r["agent_condition"] == agent]


def selected(job: dict, directory: Path) -> tuple[bool, str]:
    status = directory / "status.json"
    if status.exists():
        state = json.loads(status.read_text())
        if state["status"] != "complete":
            return False, state.get("error", state["status"])
    chosen = [r for r in rows(directory / "model_selection.csv") if r["agent_condition"] == job["agent"]
              and r["selected"].lower() == "true" and r["eligible"].lower() == "true"]
    return (True, f"selected {chosen[0]['checkpoint_step']}") if len(chosen) == 1 else (False, "no eligible checkpoint")


def onsets(job: dict, directory: Path) -> list[dict]:
    episodes = [r for r in rows(directory / "analysis/episode_outcomes.csv") if r["agent_condition"] == job["agent"]]
    if len(episodes) != 36:
        raise ValueError(f"Expected 36 evaluation episodes: {directory}")
    physical = {}
    if job["arm"] != "A":
        physical = {r["episode_id"]: r for r in rows(directory / "analysis/actual_lane_change_summary.csv")
                    if r["agent_condition"] == job["agent"]}
        if len(physical) != 36:
            raise ValueError(f"Expected 36 physical lane-change records: {directory}")
    out = []
    for row in episodes:
        collision = row["collision_flag"].lower() == "true"
        if job["arm"] == "A":
            outcome, latency = row["episode_outcome"], row["response_latency_seconds"]
        else:
            actual = physical[row["episode_id"]]
            observed = actual["actual_lane_change_observed"].lower() == "true"
            latency = actual["first_actual_lane_change_seconds"] if observed else ""
            outcome = ("valid_onset" if observed else "terminal_failure"
                       if collision or row["episode_outcome"] == "terminal_failure" else "no_onset_censored")
        out.append(dict(**job, episode_id=row["episode_id"], exposure_id=row["exposure_id"],
                        episode_outcome=outcome, collision=collision,
                        onset_seconds=latency if outcome == "valid_onset" else ""))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUTS / "reward_equal_start/analysis_first_log")
    args = parser.parse_args(argv)
    task_list = tasks()
    if len(task_list) != 60:
        raise ValueError(f"Expected 60 runs, found {len(task_list)}")
    paired = {(j["arm"], j["seed"]): scenes(d, "BAL") for j, d in task_list if j["variant"] == "BAL"}
    progress, status, episodes, summary = [], [], [], []
    for job, directory in task_list:
        meta = [r for r in rows(directory / "training_runs.csv") if r["agent_condition"] == job["agent"]]
        if len(meta) != 1:
            raise ValueError(f"Expected one training record: {directory}")
        for field, value in (("reward_speed_score", job["speed_weight"]),
                             ("reward_front_distance_score", job["front_distance_weight"])):
            if not math.isclose(float(meta[0][field]), value, abs_tol=1e-9):
                raise ValueError(f"{directory}: {field}={meta[0][field]} differs from {value}")
        if int(meta[0]["total_timesteps"]) != 100000 or meta[0]["dqn_variant"] != "double-dqn":
            raise ValueError(f"Incompatible training protocol: {directory}")
        if job["variant"] != "BAL":
            current, base = scenes(directory, job["agent"]), paired[(job["arm"], job["seed"])]
            n = min(len(current), len(base))
            if not n or current[:n] != base[:n]:
                raise ValueError(f"Training scene sequence differs from paired BAL: {directory}")
        trace = [dict(**job, timesteps=float(r["time/total_timesteps"]), episode_reward=float(r["rollout/ep_rew_mean"]))
                 for r in rows(directory / "training_logs" / job["agent"] / "progress.csv") if r.get("rollout/ep_rew_mean")]
        if not trace or trace[-1]["timesteps"] < 99000 or not all(math.isfinite(t["episode_reward"]) for t in trace):
            raise ValueError(f"Incomplete or nonfinite training: {directory}")
        progress += trace
        ok, detail = selected(job, directory)
        status.append(dict(**job, run=directory.name, final_timesteps=trace[-1]["timesteps"],
                           first_log_timesteps=trace[0]["timesteps"], first_log_reward=trace[0]["episode_reward"],
                           selection_status="complete" if ok else "failed", detail=detail))
        if ok:
            eps = onsets(job, directory)
            episodes += eps
            valid = [float(e["onset_seconds"]) for e in eps if e["episode_outcome"] == "valid_onset"]
            summary.append(dict(**job, n_total=36, n_onset=len(valid),
                                n_censored=sum(e["episode_outcome"] == "no_onset_censored" for e in eps),
                                n_failure=sum(e["episode_outcome"] == "terminal_failure" for e in eps),
                                n_collision=sum(e["collision"] for e in eps),
                                median_onset_seconds=median(valid) if valid else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("training_progress.csv", progress), ("run_status.csv", status),
                        ("episode_onsets.csv", episodes), ("seed_summary.csv", summary)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    failed = [s["run"] for s in status if s["selection_status"] != "complete"]
    print(f"{len(status)} runs, {len(progress)} progress rows, {len(summary)} with onsets; selection failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
