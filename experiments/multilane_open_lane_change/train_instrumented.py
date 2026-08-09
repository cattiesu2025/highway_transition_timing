#!/usr/bin/env python3
"""Instrumented BAL retraining for the duration-20 seed failure diagnosis.

The maintained training configuration is unchanged. This diagnostic adds only
read-only callbacks around learning: checkpoints, replay-buffer snapshots and
summaries, cumulative action counts, fixed development risk-surface Q-values,
and a lightweight 36-scene development evaluation at each checkpoint.

The sealed held-out grid is never loaded by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


ACTIONS = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTIONS)}


def load_experiment_module(repo_root: Path):
    path = repo_root / "experiments" / "multilane_open_lane_change" / "run.py"
    spec = importlib.util.spec_from_file_location(
        "multilane_open_lane_change_instrumented_run",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class InstrumentedBALCallback(BaseCallback):
    """Persist development-only diagnostics without changing reward or updates."""

    def __init__(
        self,
        experiment,
        config,
        output_dir: Path,
        checkpoint_every: int,
        development_exposures: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if development_exposures <= 0:
            raise ValueError("development_exposures must be positive")
        self.experiment = experiment
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_every = checkpoint_every
        self.development_exposures = development_exposures
        self.next_checkpoint = checkpoint_every
        self.last_snapshot_step = -1
        self.cumulative_action_counts: Counter[str] = Counter()
        self.q_surface_rows: list[dict[str, Any]] = []
        self.action_count_rows: list[dict[str, Any]] = []
        self.replay_summary_rows: list[dict[str, Any]] = []
        self.development_episode_rows: list[dict[str, Any]] = []
        self.development_summary_rows: list[dict[str, Any]] = []
        self.risk_observations: np.ndarray | None = None
        self.risk_states: list[dict[str, float]] = []

        self.instrumentation_dir = output_dir / "instrumentation"
        self.checkpoint_dir = self.instrumentation_dir / "checkpoints"
        self.replay_dir = self.instrumentation_dir / "replay_buffers"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self) -> None:
        rng_state = capture_rng_state()
        try:
            self.risk_observations, self.risk_states = self._make_risk_grid()
        finally:
            restore_rng_state(rng_state)

    def _on_step(self) -> bool:
        # SB3 calls callbacks after env.step() but before it stores/trains on
        # the current transition. Snapshot step N at callback N+1 so the model,
        # replay buffer and cumulative action counts all include exactly N
        # completed transitions and gradient opportunities.
        completed_steps = self.num_timesteps - int(self.training_env.num_envs)
        total_target = int(getattr(self.model, "_total_timesteps", 0))
        while completed_steps >= self.next_checkpoint:
            checkpoint_step = self.next_checkpoint
            self.next_checkpoint += self.checkpoint_every
            if checkpoint_step < total_target:
                self._snapshot(checkpoint_step)

        actions = np.asarray(self.locals.get("actions", []), dtype=int).reshape(-1)
        for action_index in actions:
            self.cumulative_action_counts[ACTIONS[int(action_index)]] += 1
        return True

    def _on_training_end(self) -> None:
        self._snapshot(int(self.num_timesteps), force=True)

    def _snapshot(self, step: int, force: bool = False) -> None:
        if step == self.last_snapshot_step and not force:
            return
        rng_state = capture_rng_state()
        try:
            print(f"Writing instrumented BAL checkpoint at step {step}", flush=True)
            self.model.save(self.checkpoint_dir / f"BAL_step_{step:06d}.zip")
            self.model.save_replay_buffer(
                self.replay_dir / f"BAL_step_{step:06d}.pkl"
            )
            self._record_action_counts(step)
            self._record_q_surface(step)
            self._record_replay_summary(step)
            self._record_development_evaluation(step)
            self._write_tables()
            self.last_snapshot_step = step
        finally:
            restore_rng_state(rng_state)

    def _make_risk_grid(self) -> tuple[np.ndarray, list[dict[str, float]]]:
        env = self.experiment.make_env("BAL", self.config, training=False)
        observations: list[np.ndarray] = []
        states: list[dict[str, float]] = []
        try:
            state_index = 0
            for centre_distance in range(10, 201, 10):
                for closing_speed in range(0, 19, 2):
                    ego_speed = 28.0
                    front_speed = ego_speed - float(closing_speed)
                    env.reset(seed=self.config.seed)
                    spec = self.experiment.OpenLaneSpec(
                        exposure_id=f"risk_{state_index:04d}",
                        exposure_seed=self.config.seed,
                        ego_speed=ego_speed,
                        front_distance=float(centre_distance),
                        front_speed=front_speed,
                        ego_lane=self.config.ego_lane,
                    )
                    observation = self.experiment.apply_open_lane_scene(env, spec)
                    observations.append(np.asarray(observation, dtype=np.float32))
                    net_gap = max(float(centre_distance) - 5.0, 0.0)
                    states.append(
                        {
                            "state_index": state_index,
                            "ego_speed": ego_speed,
                            "front_speed": front_speed,
                            "front_centre_distance": float(centre_distance),
                            "front_net_gap": net_gap,
                            "closing_speed": float(closing_speed),
                            "ttc_seconds": (
                                net_gap / float(closing_speed)
                                if closing_speed > 0 and net_gap > 0
                                else math.inf
                            ),
                        }
                    )
                    state_index += 1
        finally:
            env.close()
        return np.stack(observations), states

    def _record_q_surface(self, step: int) -> None:
        if self.risk_observations is None:
            raise RuntimeError("Risk grid was not initialized")
        observation_tensor, _ = self.model.policy.obs_to_tensor(
            self.risk_observations
        )
        with torch.no_grad():
            q_values = self.model.q_net(observation_tensor).cpu().numpy()
        for state, values in zip(self.risk_states, q_values, strict=True):
            argmax_index = int(np.argmax(values))
            row: dict[str, Any] = {
                "checkpoint_step": step,
                "seed": self.config.seed,
                **state,
                "q_argmax_action": ACTIONS[argmax_index],
                "lane_change_q_advantage": float(
                    max(values[0], values[2])
                    - max(values[1], values[3], values[4])
                ),
            }
            for action, value in zip(ACTIONS, values, strict=True):
                row[f"q_{action.lower()}"] = float(value)
            self.q_surface_rows.append(row)

    def _record_action_counts(self, step: int) -> None:
        total = sum(self.cumulative_action_counts.values())
        for action in ACTIONS:
            count = self.cumulative_action_counts[action]
            self.action_count_rows.append(
                {
                    "checkpoint_step": step,
                    "seed": self.config.seed,
                    "action": action,
                    "cumulative_count": count,
                    "cumulative_fraction": count / total if total else 0.0,
                    "cumulative_total": total,
                }
            )

    def _valid_replay_arrays(self) -> tuple[np.ndarray, ...]:
        replay_buffer = self.model.replay_buffer
        size = int(replay_buffer.size())
        if size <= 0:
            raise RuntimeError("Replay buffer is empty at checkpoint")
        valid = slice(None) if replay_buffer.full else slice(0, size)
        observations = np.asarray(replay_buffer.observations[valid])
        actions = np.asarray(replay_buffer.actions[valid])
        rewards = np.asarray(replay_buffer.rewards[valid])
        dones = np.asarray(replay_buffer.dones[valid])
        observations = observations.reshape((-1,) + observations.shape[2:])
        return (
            observations,
            actions.reshape(-1),
            rewards.reshape(-1),
            dones.reshape(-1),
        )

    def _record_replay_summary(self, step: int) -> None:
        observations, actions, rewards, dones = self._valid_replay_arrays()
        if observations.ndim != 3 or observations.shape[1:] != (4, 5):
            raise RuntimeError(
                f"Unexpected replay observation shape: {observations.shape}"
            )
        front_slot = observations[:, 1, :]
        front_present = front_slot[:, 0] > 0.5
        front_ahead = front_present & (front_slot[:, 1] > 0)
        closing_front = front_ahead & (front_slot[:, 3] < 0)
        near_closing_front = closing_front & (front_slot[:, 1] <= 0.5)

        row: dict[str, Any] = {
            "checkpoint_step": step,
            "seed": self.config.seed,
            "replay_size": len(actions),
            "replay_capacity": int(self.model.replay_buffer.buffer_size),
            "replay_full": bool(self.model.replay_buffer.full),
            "replay_position": int(self.model.replay_buffer.pos),
            "reward_mean": float(np.mean(rewards)),
            "reward_q1": float(np.quantile(rewards, 0.25)),
            "reward_median": float(np.median(rewards)),
            "reward_q3": float(np.quantile(rewards, 0.75)),
            "done_fraction": float(np.mean(dones > 0.5)),
            "front_slot_present_fraction": float(np.mean(front_present)),
            "front_ahead_fraction": float(np.mean(front_ahead)),
            "closing_front_fraction": float(np.mean(closing_front)),
            "near_closing_front_fraction": float(np.mean(near_closing_front)),
            "front_x_normalized_median_when_ahead": (
                float(np.median(front_slot[front_ahead, 1]))
                if np.any(front_ahead)
                else ""
            ),
            "front_vx_normalized_median_when_closing": (
                float(np.median(front_slot[closing_front, 3]))
                if np.any(closing_front)
                else ""
            ),
        }
        for action_index, action in enumerate(ACTIONS):
            count = int(np.sum(actions == action_index))
            row[f"action_{action.lower()}_count"] = count
            row[f"action_{action.lower()}_fraction"] = count / len(actions)
        self.replay_summary_rows.append(row)

    def _record_development_evaluation(self, step: int) -> None:
        specs = self.experiment.make_eval_specs(
            self.development_exposures,
            self.config,
        )
        rows: list[dict[str, Any]] = []
        env = self.experiment.make_env("BAL", self.config, training=False)
        try:
            for spec in specs:
                observation, _ = env.reset(seed=spec.exposure_seed)
                observation = self.experiment.apply_open_lane_scene(env, spec)
                initial_lane = int(spec.ego_lane)
                first_lane_action_seconds: float | str = ""
                physical_lane_change_seconds: float | str = ""
                collision = False
                action_counts: Counter[str] = Counter()
                steps_completed = 0

                for decision_t in range(self.config.evaluation_max_policy_steps):
                    action_index = int(
                        np.asarray(
                            self.model.predict(observation, deterministic=True)[0]
                        ).item()
                    )
                    action_name = ACTIONS[action_index]
                    action_counts[action_name] += 1
                    if (
                        first_lane_action_seconds == ""
                        and action_name in {"LANE_LEFT", "LANE_RIGHT"}
                    ):
                        first_lane_action_seconds = (
                            decision_t * self.config.policy_step_seconds
                        )

                    observation, _reward, terminated, truncated, _info = env.step(
                        action_index
                    )
                    steps_completed = decision_t + 1
                    diagnostics = self.experiment.extract_diagnostics(env)
                    collision = bool(diagnostics["collision_flag"])
                    if int(diagnostics["ego_lane"]) != initial_lane:
                        physical_lane_change_seconds = (
                            steps_completed * self.config.policy_step_seconds
                        )
                        break
                    if collision or terminated or truncated:
                        break

                if physical_lane_change_seconds != "":
                    outcome = "physical_lane_change"
                elif collision:
                    outcome = "collision_before_change"
                else:
                    outcome = "no_change_by_horizon"
                rows.append(
                    {
                        "checkpoint_step": step,
                        "seed": self.config.seed,
                        "agent_condition": "BAL",
                        "exposure_id": spec.exposure_id,
                        "ego_speed": spec.ego_speed,
                        "front_distance": spec.front_distance,
                        "front_speed": spec.front_speed,
                        "first_lane_action_seconds": first_lane_action_seconds,
                        "physical_lane_change_seconds": physical_lane_change_seconds,
                        "collision": collision,
                        "outcome": outcome,
                        "steps_completed": steps_completed,
                        **{
                            f"action_{action.lower()}_count": action_counts[action]
                            for action in ACTIONS
                        },
                    }
                )
        finally:
            env.close()

        self.development_episode_rows.extend(rows)
        physical_times = [
            float(row["physical_lane_change_seconds"])
            for row in rows
            if row["physical_lane_change_seconds"] != ""
        ]
        self.development_summary_rows.append(
            {
                "checkpoint_step": step,
                "seed": self.config.seed,
                "agent_condition": "BAL",
                "n_exposures": len(rows),
                "physical_lane_changes": sum(
                    row["outcome"] == "physical_lane_change" for row in rows
                ),
                "collisions_before_change": sum(
                    row["outcome"] == "collision_before_change" for row in rows
                ),
                "no_change_by_horizon": sum(
                    row["outcome"] == "no_change_by_horizon" for row in rows
                ),
                "median_physical_lane_change_seconds": (
                    float(np.median(physical_times)) if physical_times else ""
                ),
            }
        )

    def _write_tables(self) -> None:
        write_csv_rows = self.experiment.write_csv_rows
        write_csv_rows(
            self.instrumentation_dir / "q_risk_surface.csv",
            self.q_surface_rows,
        )
        write_csv_rows(
            self.instrumentation_dir / "cumulative_action_counts.csv",
            self.action_count_rows,
        )
        write_csv_rows(
            self.instrumentation_dir / "replay_buffer_summary.csv",
            self.replay_summary_rows,
        )
        write_csv_rows(
            self.instrumentation_dir / "development_checkpoint_episodes.csv",
            self.development_episode_rows,
        )
        write_csv_rows(
            self.instrumentation_dir / "development_checkpoint_summary.csv",
            self.development_summary_rows,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--num-exposures", type=int, default=36)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--evaluation-duration", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    experiment = load_experiment_module(repo_root)
    experiment.configure_headless_runtime()
    output_dir = Path(args.out)
    existing_entries = (
        {path.name for path in output_dir.iterdir()}
        if output_dir.exists()
        else set()
    )
    unexpected_entries = existing_entries - {"training_scenario_audit"}
    if unexpected_entries:
        raise RuntimeError(
            "Refusing to overwrite diagnostic directory containing: "
            f"{sorted(unexpected_entries)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = experiment.ExperimentConfig(
        duration=args.duration,
        evaluation_duration=args.evaluation_duration,
        seed=args.seed,
        lanes_count=4,
        ego_lane=1,
        policy_frequency=5,
        simulation_frequency=15,
        observation_normalize=True,
        observation_absolute=False,
        slow_down_penalty=0.2,
        lane_change_penalty=0.2,
        collision_risk_penalty=3.0,
        collision_penalty=None,
        dqn_variant="double-dqn",
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=32,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        n_steps=1,
        target_update_interval=250,
        exploration_initial_eps=1.0,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        training_scenario_profile="stratified",
        no_front_train_fraction=0.2,
        near_matched_speed_delta=2.0,
    )

    experiment.write_csv_rows(
        output_dir / "instrumentation_config.csv",
        [
            {
                "experiment": "multilane_duration20_instrumented_bal",
                "seed": args.seed,
                "agent_condition": "BAL",
                "timesteps": args.timesteps,
                "checkpoint_every": args.checkpoint_every,
                "development_exposures_per_checkpoint": args.num_exposures,
                "training_duration_seconds": args.duration,
                "evaluation_duration_seconds": args.evaluation_duration,
                "heldout_opened": False,
                "diagnostics_change_training_objective": False,
            }
        ],
    )

    def callback_factory(agent: str, _model, _training_env):
        if agent != "BAL":
            raise RuntimeError(f"Instrumented run supports BAL only, got {agent}")
        return InstrumentedBALCallback(
            experiment=experiment,
            config=config,
            output_dir=output_dir,
            checkpoint_every=args.checkpoint_every,
            development_exposures=args.num_exposures,
            verbose=args.verbose,
        )

    experiment.train_agents(
        output_dir=output_dir,
        agents=["BAL"],
        total_timesteps=args.timesteps,
        config=config,
        verbose=args.verbose,
        callback_factory=callback_factory,
    )
    steps, exposures = experiment.evaluate_agents(
        output_dir,
        ["BAL"],
        args.num_exposures,
        config,
    )
    result = experiment.write_analysis(
        output_dir,
        steps,
        exposures,
        config,
        bootstrap_samples=args.bootstrap_samples,
        figures=False,
    )
    experiment.print_summary(result, output_dir)
    print(f"Completed instrumented BAL run: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
