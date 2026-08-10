#!/usr/bin/env python3
"""Generate non-visual root-cause diagnostics for seed 4016 / BAL.

This script deliberately avoids the sealed held-out grid. It produces two CSVs:

1. Q-values for all 20 BAL models on an identical synthetic risk-state grid.
2. Exact simulator branches two seconds before each seed4016/BAL collision,
   comparing the learned policy with five forced first actions.

Plotting is intentionally left to the R-only figure script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from highway_transition_timing.double_dqn import DoubleDQN


SEEDS = tuple(range(4000, 4020))
FAILED_SEED = 4016
ACTIONS = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTIONS)}


def load_experiment_module(repo_root: Path):
    path = (
        repo_root
        / "experiments"
        / "multilane_open_lane_change"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location(
        "multilane_open_lane_change_diagnostic_run",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_dir(outputs_root: Path, seed: int) -> Path:
    return outputs_root / f"multilane_open_lane_change_duration20_100k_seed{seed}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def parse_q_values(value: str) -> np.ndarray:
    values = np.fromstring(value.strip().strip("[]"), sep=",", dtype=float)
    if values.shape != (5,):
        raise RuntimeError(f"Expected five Q-values, found {value!r}")
    return values


def model_path(outputs_root: Path, seed: int) -> Path:
    path = run_dir(outputs_root, seed) / "models" / "BAL_main.zip"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_synthetic_observations(experiment, config) -> tuple[np.ndarray, list[dict[str, float]]]:
    env = experiment.make_env("BAL", config, training=False)
    observations: list[np.ndarray] = []
    states: list[dict[str, float]] = []
    try:
        state_index = 0
        for centre_distance in range(10, 201, 10):
            for closing_speed in range(0, 19, 2):
                ego_speed = 28.0
                front_speed = ego_speed - float(closing_speed)
                env.reset(seed=FAILED_SEED)
                state = experiment.OpenLaneSpec(
                    exposure_id=f"synthetic_{state_index:04d}",
                    exposure_seed=FAILED_SEED,
                    ego_speed=ego_speed,
                    front_distance=float(centre_distance),
                    front_speed=front_speed,
                    ego_lane=1,
                )
                observation = experiment.apply_open_lane_scene(env, state)
                observations.append(np.asarray(observation, dtype=np.float32))
                net_gap = max(float(centre_distance) - 5.0, 0.0)
                ttc = (
                    net_gap / float(closing_speed)
                    if closing_speed > 0 and net_gap > 0
                    else math.inf
                )
                states.append(
                    {
                        "state_index": state_index,
                        "ego_speed": ego_speed,
                        "front_speed": front_speed,
                        "front_centre_distance": float(centre_distance),
                        "front_net_gap": net_gap,
                        "closing_speed": float(closing_speed),
                        "ttc_seconds": ttc,
                    }
                )
                state_index += 1
    finally:
        env.close()
    return np.stack(observations), states


def generate_q_surface(
    outputs_root: Path,
    experiment,
    config,
) -> list[dict[str, Any]]:
    observations, states = build_synthetic_observations(experiment, config)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"Q surface: loading BAL seed {seed}", flush=True)
        model = DoubleDQN.load(model_path(outputs_root, seed))
        observation_tensor, _ = model.policy.obs_to_tensor(observations)
        with torch.no_grad():
            q_values = model.q_net(observation_tensor).cpu().numpy()
        for state, values in zip(states, q_values, strict=True):
            argmax_index = int(np.argmax(values))
            lane_q = float(max(values[0], values[2]))
            non_lane_q = float(max(values[1], values[3], values[4]))
            row: dict[str, Any] = {
                "seed": seed,
                "seed_status": (
                    "seed 4016 (collision)" if seed == FAILED_SEED else "19 safe seeds"
                ),
                **state,
                "q_argmax_action": ACTIONS[argmax_index],
                "lane_change_q_advantage": lane_q - non_lane_q,
            }
            for action, value in zip(ACTIONS, values, strict=True):
                row[f"q_{action.lower()}"] = float(value)
            rows.append(row)
    return rows


def collided_bal_trajectories(outputs_root: Path) -> dict[str, list[dict[str, str]]]:
    summary_path = (
        run_dir(outputs_root, FAILED_SEED)
        / "analysis"
        / "actual_lane_change_summary.csv"
    )
    collided_ids = {
        row["exposure_id"]
        for row in read_csv_rows(summary_path)
        if row["agent_condition"] == "BAL" and parse_bool(row["collision_flag"])
    }
    if len(collided_ids) != 21:
        raise RuntimeError(f"Expected 21 collided BAL exposures, found {len(collided_ids)}")

    steps_path = run_dir(outputs_root, FAILED_SEED) / "evaluation" / "steps.csv"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with steps_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["agent_condition"] == "BAL" and row["exposure_id"] in collided_ids:
                grouped[row["exposure_id"]].append(row)
    for exposure_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["t"]))
        if not parse_bool(rows[-1]["collision_flag"]):
            raise RuntimeError(f"Trajectory does not end in collision: {exposure_id}")
    return dict(grouped)


def select_branch_index(rows: list[dict[str, str]], lead_seconds: float = 2.0) -> int:
    terminal_seconds = float(rows[-1]["t_seconds"])
    target_seconds = max(0.0, terminal_seconds - lead_seconds)
    candidates = [
        index
        for index, row in enumerate(rows)
        if float(row["t_seconds"]) <= target_seconds + 1e-9
    ]
    if not candidates:
        return 0
    return max(candidates, key=lambda index: float(rows[index]["t_seconds"]))


def replay_to_branch(
    experiment,
    config,
    spec,
    prefix_rows: list[dict[str, str]],
):
    env = experiment.make_env("BAL", config, training=False)
    observation, _ = env.reset(seed=FAILED_SEED)
    observation = experiment.apply_open_lane_scene(env, spec)
    for row in prefix_rows:
        action_index = ACTION_TO_INDEX[row["action"]]
        observation, _reward, terminated, truncated, _info = env.step(action_index)
        if terminated or truncated:
            env.close()
            raise RuntimeError(
                f"Replay terminated before branch in {row['exposure_id']} at t={row['t']}"
            )
    return env, observation


def generate_forced_action_branches(
    outputs_root: Path,
    experiment,
    config,
) -> list[dict[str, Any]]:
    model = DoubleDQN.load(model_path(outputs_root, FAILED_SEED))
    trajectories = collided_bal_trajectories(outputs_root)
    eval_specs = {
        spec.exposure_id: spec for spec in experiment.make_eval_specs(36, config)
    }
    branch_labels = ("POLICY",) + ACTIONS
    max_steps = 50
    gamma = float(model.gamma)
    result_rows: list[dict[str, Any]] = []

    for exposure_id in sorted(trajectories):
        trajectory = trajectories[exposure_id]
        branch_index = select_branch_index(trajectory)
        branch_row = trajectory[branch_index]
        prefix_rows = trajectory[:branch_index]
        spec = eval_specs[exposure_id]
        recorded_q = parse_q_values(branch_row["q_values_or_action_scores"])

        for branch_label in branch_labels:
            env, observation = replay_to_branch(
                experiment,
                config,
                spec,
                prefix_rows,
            )
            try:
                branch_diagnostics = experiment.extract_diagnostics(env)
                speed_error = abs(
                    float(branch_diagnostics["ego_speed"])
                    - float(branch_row["ego_speed"])
                )
                distance_error = abs(
                    float(branch_diagnostics["nearest_front_distance"])
                    - float(branch_row["nearest_front_distance"])
                )
                observation_tensor, _ = model.policy.obs_to_tensor(observation)
                with torch.no_grad():
                    branch_q = model.q_net(observation_tensor).cpu().numpy()[0]
                q_error = float(np.max(np.abs(branch_q - recorded_q)))
                if speed_error > 1e-4 or distance_error > 1e-4 or q_error > 1e-4:
                    raise RuntimeError(
                        "Replay mismatch for "
                        f"{exposure_id}: speed={speed_error}, distance={distance_error}, "
                        f"Q={q_error}"
                    )

                if branch_label == "POLICY":
                    first_action_index = int(
                        np.asarray(model.predict(observation, deterministic=True)[0]).item()
                    )
                else:
                    first_action_index = ACTION_TO_INDEX[branch_label]
                first_action = ACTIONS[first_action_index]
                first_action_q = float(branch_q[first_action_index])

                discounted_return = 0.0
                undiscounted_return = 0.0
                collision = False
                lane_change_observed = False
                lane_change_latency_steps: int | None = None
                terminated = False
                truncated = False
                first_reward = math.nan
                steps_completed = 0

                for step_offset in range(max_steps):
                    if step_offset == 0:
                        action_index = first_action_index
                    else:
                        action_index = int(
                            np.asarray(
                                model.predict(observation, deterministic=True)[0]
                            ).item()
                        )
                    observation, reward, terminated, truncated, _info = env.step(
                        action_index
                    )
                    reward_value = float(reward)
                    if step_offset == 0:
                        first_reward = reward_value
                    discounted_return += (gamma**step_offset) * reward_value
                    undiscounted_return += reward_value
                    steps_completed = step_offset + 1

                    diagnostics = experiment.extract_diagnostics(env)
                    if int(diagnostics["ego_lane"]) != int(spec.ego_lane):
                        lane_change_observed = True
                        if lane_change_latency_steps is None:
                            lane_change_latency_steps = steps_completed
                    collision = collision or bool(diagnostics["collision_flag"])
                    if terminated or truncated:
                        break

                result_rows.append(
                    {
                        "exposure_id": exposure_id,
                        "branch_t": int(branch_row["t"]),
                        "branch_t_seconds": float(branch_row["t_seconds"]),
                        "original_collision_t_seconds": float(
                            trajectory[-1]["t_seconds"]
                        ),
                        "seconds_before_original_collision": (
                            float(trajectory[-1]["t_seconds"])
                            - float(branch_row["t_seconds"])
                        ),
                        "branch_ego_speed": float(branch_row["ego_speed"]),
                        "branch_front_speed": float(
                            branch_row["front_vehicle_speed"]
                        ),
                        "branch_front_distance": float(
                            branch_row["nearest_front_distance"]
                        ),
                        "branch_closing_speed": (
                            float(branch_row["ego_speed"])
                            - float(branch_row["front_vehicle_speed"])
                        ),
                        "recorded_policy_action": branch_row["action"],
                        "branch_label": branch_label,
                        "executed_first_action": first_action,
                        "first_action_q": first_action_q,
                        "first_reward": first_reward,
                        "discounted_return_10s": discounted_return,
                        "undiscounted_return_10s": undiscounted_return,
                        "steps_completed": steps_completed,
                        "collision": collision,
                        "lane_change_observed": lane_change_observed,
                        "lane_change_latency_seconds": (
                            lane_change_latency_steps * config.policy_step_seconds
                            if lane_change_latency_steps is not None
                            else ""
                        ),
                        "terminated": terminated,
                        "truncated": truncated,
                        "replay_speed_abs_error": speed_error,
                        "replay_front_distance_abs_error": distance_error,
                        "replay_q_max_abs_error": q_error,
                    }
                )
            finally:
                env.close()

    policy_rows = [row for row in result_rows if row["branch_label"] == "POLICY"]
    if len(policy_rows) != 21 or not all(row["collision"] for row in policy_rows):
        raise RuntimeError(
            "The exact-replay POLICY branch did not reproduce all 21 collisions"
        )
    return result_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--out",
        default=(
            "outputs/multilane_open_lane_change_duration20_100k_20seed/"
            "figures/diagnostics/root_cause/source_data"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    outputs_root = (repo_root / args.outputs_root).resolve()
    output_dir = (repo_root / args.out).resolve()
    experiment = load_experiment_module(repo_root)
    config = experiment.ExperimentConfig(
        seed=FAILED_SEED,
        duration=20,
        evaluation_duration=120,
        lanes_count=2,
        ego_lane=1,
    )

    q_rows = generate_q_surface(outputs_root, experiment, config)
    write_csv_rows(output_dir / "bal_q_risk_surface_20seed.csv", q_rows)

    branch_rows = generate_forced_action_branches(
        outputs_root,
        experiment,
        config,
    )
    write_csv_rows(output_dir / "seed4016_bal_forced_action_branches.csv", branch_rows)
    print(f"Wrote root-cause source data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
