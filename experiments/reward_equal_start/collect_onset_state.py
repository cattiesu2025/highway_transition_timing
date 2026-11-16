#!/usr/bin/env python3
"""Ego speed and lead gap at the onset instant, for one equal-start weight set (same runs and
episodes as plot_onset_4arm.R):
  t0:        collect_onsets_t0.py runs  -> outputs/reward_equal_start/analysis_t0_onset
  first_log: collect_first_log.py runs  -> outputs/reward_equal_start/analysis_first_log
Usage: collect_onset_state.py <t0|first_log>

Alignment: every value is the state at the onset time itself (t_seconds == onset_seconds).
- A (onset = slowdown, decision-time onset): steps row t = target_onset_t.
- B/C/D (onset = first physical lane change, post-step time): the state after step
  t* = first_actual_lane_change_t, i.e. steps row t* + 1.

Speed: ego_speed of that row (m/s).
Gap: bumper-to-bumper longitudinal gap to the lead vehicle in ego's ORIGINAL lane:
  centre distance - (L_ego + L_lead) / 2, with highway-env Vehicle.LENGTH = 5.0 m.
- A: ego never leaves its lane, so it is nearest_front_distance of that row - 5.
- B/C/D: at row t* + 1 ego's lane index has already flipped, so nearest_front_distance there
  refers to the target lane. The original-lane lead's position is advanced one policy step
  from row t*: x_lead = x_ego(t*) + d(t*) + v_lead(t*) * dt. The constant-speed step is
  checked on the preceding step (t* - 1 -> t*, same lead id) and the max error is printed.
Episodes with no lead in the original lane get an empty gap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect_first_log as fl  # noqa: E402
import collect_onsets_t0 as t0  # noqa: E402

SETS = {"t0": (t0.tasks, "analysis_t0_onset"), "first_log": (fl.tasks, "analysis_first_log")}

VEHICLE_LENGTH = 5.0  # highway_env.vehicle.kinematics.Vehicle.LENGTH, ego and traffic alike


def steps_path(directory: Path) -> Path:
    for p in (directory / "analysis/steps.csv", directory / "evaluation/steps.csv"):
        if p.exists():
            return p
    raise FileNotFoundError(f"No steps.csv: {directory}")


def fnum(value: str) -> float:
    return float(value) if value not in ("", None) else math.inf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set", choices=sorted(SETS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    tasks, analysis = SETS[args.set]
    args.out = args.out or fl.OUTPUTS / "reward_equal_start" / analysis
    out, max_step_err, n_runs = [], 0.0, 0
    for job, directory in tasks():
        ok, _ = fl.selected(job, directory)
        if not ok:
            continue
        n_runs += 1
        agent = job["agent"]
        steps: dict[str, dict[int, dict]] = {}
        for r in fl.rows(steps_path(directory)):
            if r["agent_condition"] == agent and r["evaluation_policy_mode"] == "deterministic":
                steps.setdefault(r["episode_id"], {})[int(r["t"])] = r
        outcomes = {r["episode_id"]: r for r in fl.rows(directory / "analysis/episode_outcomes.csv")
                    if r["agent_condition"] == agent}
        lane = {}
        if job["arm"] != "A":
            lane = {r["episode_id"]: r for r in fl.rows(directory / "analysis/actual_lane_change_summary.csv")
                    if r["agent_condition"] == agent}
        for ep in fl.onsets(job, directory):
            if ep["episode_outcome"] != "valid_onset":
                continue
            eid, onset_s = ep["episode_id"], float(ep["onset_seconds"])
            s = steps[eid]
            if job["arm"] == "A":
                t_on = int(outcomes[eid]["target_onset_t"])
                row = s[t_on]
                d = fnum(row["nearest_front_distance"])
                lead_centre = d
            else:
                t_star = int(lane[eid]["first_actual_lane_change_t"])
                pre, row = s[t_star], s[t_star + 1]
                if int(pre["ego_lane"]) != int(lane[eid]["initial_lane"]) or int(row["ego_lane"]) == int(pre["ego_lane"]):
                    raise ValueError(f"Unexpected lane sequence at onset: {directory} {eid}")
                dt = float(pre["policy_step_seconds"])
                d = fnum(pre["nearest_front_distance"])
                if math.isfinite(d):
                    x_pre = json.loads(pre["ego_position"])[0]
                    x_now = json.loads(row["ego_position"])[0]
                    v_lead = float(pre["front_vehicle_speed"])
                    lead_centre = x_pre + d + v_lead * dt - x_now
                    prev = s.get(t_star - 1)
                    if prev and prev["nearest_front_vehicle_id"] == pre["nearest_front_vehicle_id"]:
                        pred = json.loads(prev["ego_position"])[0] + fnum(prev["nearest_front_distance"]) \
                            + float(prev["front_vehicle_speed"]) * dt
                        max_step_err = max(max_step_err, abs(pred - (x_pre + d)))
                else:
                    lead_centre = math.inf
            if not math.isclose(float(row["t_seconds"]), onset_s, abs_tol=1e-6):
                raise ValueError(f"Row time {row['t_seconds']} != onset {onset_s}: {directory} {eid}")
            gap = lead_centre - VEHICLE_LENGTH
            out.append(dict(arm=job["arm"], variant=job["variant"], seed=job["seed"], episode_id=eid,
                            onset_seconds=onset_s, onset_speed=float(row["ego_speed"]),
                            onset_gap_centre=lead_centre if math.isfinite(lead_centre) else "",
                            onset_gap=gap if math.isfinite(gap) else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "episode_onset_state.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    no_lead = sum(r["onset_gap"] == "" for r in out)
    print(f"{n_runs} runs, {len(out)} onset episodes -> {path}; no original-lane lead: {no_lead}; "
          f"max one-step lead extrapolation error: {max_step_err:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
