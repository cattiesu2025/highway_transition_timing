#!/usr/bin/env python3
"""Extract weight-independent behaviour for selected weights F1/F4/F7 in arms A/B/C/D.

Each policy's first 100 evaluation steps (the 20 s training horizon) on original
development scenes give mean speed and mean front gap capped at the 120 m scoring
horizon, where no front vehicle counts as 120 m. Seeds combine newly trained
selected-validation runs with the five reused fixed-sum runs. The logged reward
is re-derived from the expected weights, so a run with other weights fails.
Arm D uses the evaluation steps copied from Katana. FD/4209 and BAL/4205 failed
checkpoint selection under its protocol, so they have no evaluation by design.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/learning_vs_behaviour/source_data/behaviour_by_seed.csv")
ARM_D_GLOB = "outputs/arm_c_distance_figure_data/arm_c_distance-affine-v1_F[147]_100k_seed*"
ARM_D_PATTERN = re.compile(r"arm_c_distance-affine-v1_(F[147])_100k_seed(\d+)$")
ARM_D_EXPECTED_MISSING = {("FD", 4209), ("BAL", 4205)}
AGENT = {"F1": "FD", "F4": "BAL", "F7": "SP"}
WEIGHTS = {"FD": (0.28, 1.12), "BAL": (0.56, 0.84), "SP": (0.98, 0.42)}
PATTERN = re.compile(r"(?:reward_selected_validation|reward_fixed_sum)_arm([ABC])_(F[147])_100k_seed(\d+)$")
COLUMNS = ["episode_id", "agent_condition", "t", "ego_speed", "front_distance_score",
           "reward_total", "reward_components", "collision_flag"]


def summarise(path: str, arm: str, variant: str, seed: int) -> dict:
    agent = AGENT[variant]
    steps = pd.read_csv(f"{path}/evaluation/steps.csv", usecols=COLUMNS)
    steps = steps[(steps.agent_condition == agent) & (steps.t < 100)]
    if steps.empty:
        raise ValueError(f"no steps for {agent} in {path}")
    parts = steps.reward_components.map(json.loads)
    speed_weight, front_weight = WEIGHTS[agent]
    action_cost = 0.0 if arm == "A" else 0.2
    expected = (speed_weight * parts.map(lambda x: x["speed_score"])
                + front_weight * parts.map(lambda x: x["front_distance_score"])
                + 2.0 * parts.map(lambda x: x["collision_penalty"])
                + 3.0 * parts.map(lambda x: x["collision_risk_penalty"])
                - action_cost * parts.map(lambda x: abs(x["slow_down_penalty"]) + abs(x["lane_change_penalty"])))
    if not np.allclose(steps.reward_total, expected, atol=1e-3):
        raise ValueError(f"logged reward does not match {agent} weights in {path}")
    scores = steps.assign(gap=steps.front_distance_score * 120,
                          S=parts.map(lambda x: x["speed_score"]),
                          D=parts.map(lambda x: x["front_distance_score"]))
    # Per-step reward scores (post-step, as rewarded) and the share of steps in
    # each clipped region, where the score cannot change with speed or gap.
    score_summary = {"speed_score_per_step": scores.S.mean(), "front_score_per_step": scores.D.mean(),
                     "share_speed_score_zero": (scores.S <= 0).mean(),
                     "share_front_score_capped": (scores.D >= 1).mean()}
    episodes = scores.groupby("episode_id").agg(
        speed=("ego_speed", "mean"), gap=("gap", "mean"), steps=("t", "size"),
        collided=("collision_flag", "any"))
    return {"arm": arm, "agent": agent, "seed": seed, "source": Path(path).name,
            "n_episodes": len(episodes), "mean_episode_steps": episodes.steps.mean(),
            "collision_rate": episodes.collided.mean(), "speed_ms": episodes.speed.mean(),
            "front_gap_m": episodes.gap.mean(), **score_summary}


def main() -> None:
    rows = []
    for path in sorted(glob.glob("outputs/reward_*_arm[ABC]_F[147]_100k_seed*")):
        match = PATTERN.search(Path(path).name)
        if match and Path(path, "evaluation/steps.csv").exists():
            rows.append(summarise(path, match[1], match[2], int(match[3])))
    for path in sorted(glob.glob(ARM_D_GLOB)):
        match = ARM_D_PATTERN.search(Path(path).name)
        if match and Path(path, "evaluation/steps.csv").exists():
            rows.append(summarise(path, "D", match[1], int(match[2])))
    table = pd.DataFrame(rows).sort_values(["arm", "agent", "seed"])
    duplicated = table.duplicated(["arm", "agent", "seed"])
    if duplicated.any():
        raise ValueError(f"duplicate seeds: {table[duplicated]}")
    arm_d = set(zip(table[table.arm == "D"].agent, table[table.arm == "D"].seed))
    expected = {(agent, seed) for agent in AGENT.values() for seed in range(4200, 4220)}
    if expected - arm_d != ARM_D_EXPECTED_MISSING or arm_d - expected:
        raise ValueError(f"unexpected Arm D coverage: {sorted(expected - arm_d)}")
    table.to_csv(OUT, index=False)
    print(table.groupby(["arm", "agent"]).agg(
        seeds=("seed", "nunique"), episodes=("n_episodes", "median"), steps=("mean_episode_steps", "median"),
        collisions=("collision_rate", "max"), speed=("speed_ms", "median"), gap=("front_gap_m", "median"),
        S=("speed_score_per_step", "median"), D=("front_score_per_step", "median"),
        S_zero=("share_speed_score_zero", "median"), D_cap=("share_front_score_capped", "median"),
        speed_lo=("speed_ms", "min"), speed_hi=("speed_ms", "max"), gap_lo=("front_gap_m", "min"), gap_hi=("front_gap_m", "max")).round(2).to_string())


if __name__ == "__main__":
    main()
