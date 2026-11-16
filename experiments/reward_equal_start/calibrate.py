#!/usr/bin/env python3
"""Solve SP/FD reward weights whose t=0 return exactly equals BAL's, per arm.

Before ``learning_starts`` SB3 samples actions uniformly, independent of the
network, so t=0 behaviour is one well-defined policy. Episode return is linear
in the two preference weights: R = c + w_s*S + w_d*D, with c independent of them.
On a shared reference set of random-policy episodes, equality is therefore exact
algebra, and c cancels:

    SP (w_s = 2 w_d): w_d = w_bal (S + D) / (2S + D)
    FD (w_d = 2 w_s): w_s = w_bal (S + D) / (S + 2D)

``replay`` re-runs an existing training seed and checks the decomposition against
that run's real monitor.csv. ``reference`` builds the set and solves the weights.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BAL_WEIGHT = 0.7
BLOCK_EPISODES = 20  # STRATIFIED_TRAINING_BLOCK length in both arms.
ARMS = {
    "A": ("single_lane_slow_front", 9100),
    "C": ("multilane_open_lane_change", 9200),
}
# Arm A's train_agents resets before building the model; Arm C's does not. The
# extra reset consumes a scene, so copying it to C would shift the scene stream.
RESETS_BEFORE_MODEL = {"A": True, "C": False}


def solve_weights(S: float, D: float, bal: float = BAL_WEIGHT) -> dict[str, dict[str, float]]:
    """Return speed/spacing weight pairs with the same weighted sum as BAL."""
    if S <= 0 or D <= 0:
        raise ValueError(f"S and D must be positive, got S={S}, D={D}")
    sp_spacing = bal * (S + D) / (2 * S + D)
    fd_speed = bal * (S + D) / (S + 2 * D)
    return {
        "BAL": {"speed": bal, "spacing": bal},
        "SP": {"speed": 2 * sp_spacing, "spacing": sp_spacing},
        "FD": {"speed": fd_speed, "spacing": 2 * fd_speed},
    }


def episode_return(row: dict[str, float], speed: float, spacing: float) -> float:
    return row["c"] + speed * row["S"] + spacing * row["D"]


def load_arm(arm: str):
    folder = ARMS[arm][0]
    name = f"equal_start_{folder}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sweep():
    name = "equal_start_reward_sweep"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "experiments" / "reward_sensitivity" / "run.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def training_config(arm: str, seed: int):
    """Parse the real training command so settings cannot drift from training."""
    job = {"arm": arm, "variant": "BAL", "agent": "BAL", "seed": seed,
           "speed_weight": BAL_WEIGHT, "front_distance_weight": BAL_WEIGHT}
    command = load_sweep().commands(job, ROOT / "tmp", "train", run_dir=ROOT / "tmp" / "unused")[0]
    module = load_arm(arm)
    return module, module.config_from_args(module.build_parser().parse_args(command[2:]))


def reward_layer(env):
    while not hasattr(env, "last_reward_components"):
        env = env.env
    return env


def collect(arm: str, seed: int, episodes: int) -> list[dict[str, float]]:
    """Run the training stack with warm-up never ending; decompose each episode."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    module, config = training_config(arm, seed)
    module.configure_headless_runtime()
    env = Monitor(module.make_env("BAL", config, training=True))
    if RESETS_BEFORE_MODEL[arm]:
        env.reset(seed=config.seed)
    never = 10**12
    model = module.dqn_class_for_variant(config.dqn_variant)(
        "MlpPolicy", env, learning_rate=config.learning_rate, buffer_size=1_000,
        learning_starts=never, batch_size=config.batch_size, gamma=config.gamma,
        train_freq=config.train_freq, gradient_steps=config.gradient_steps,
        n_steps=config.n_steps, target_update_interval=config.target_update_interval,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        seed=config.seed, verbose=0)
    layer = reward_layer(env)
    rows: list[dict[str, float]] = []

    class Decompose(BaseCallback):
        S = D = R = 0.0
        length = 0
        scenario_type = None

        def _on_step(self) -> bool:
            # The VecEnv auto-resets on done, so by the time this runs the wrapper
            # already holds the next episode's scene. Latch each label up front.
            if self.scenario_type is None:
                self.scenario_type = layer.env.last_training_exposure.scenario_type
            parts = layer.last_reward_components
            self.S += parts["speed_score"]
            self.D += parts["front_distance_score"]
            self.R += float(self.locals["rewards"][0])
            self.length += 1
            if self.locals["dones"][0]:
                rows.append({
                    "arm": arm, "seed": seed, "episode": len(rows),
                    "scenario_type": self.scenario_type,
                    "length": self.length, "S": self.S, "D": self.D,
                    "c": self.R - BAL_WEIGHT * (self.S + self.D),
                })
                self.S = self.D = self.R = 0.0
                self.length = 0
                self.scenario_type = layer.env.last_training_exposure.scenario_type
                return len(rows) < episodes
            return True

    model.learn(total_timesteps=never, callback=Decompose(), log_interval=None, progress_bar=False)
    env.close()
    return rows


def replay(arm: str, seed: int, variant: str, speed: float, spacing: float,
           monitor: Path, episodes: int, tolerance: float = 1e-3) -> None:
    with monitor.open() as handle:
        next(handle)
        real = list(csv.DictReader(handle))
    rows = collect(arm, seed, episodes)
    for row, actual in zip(rows, real):
        predicted = episode_return(row, speed, spacing)
        print(arm, seed, variant, row["episode"], row["length"], int(float(actual["l"])),
              round(predicted, 4), round(float(actual["r"]), 4))
        if int(float(actual["l"])) != row["length"] or abs(predicted - float(actual["r"])) > tolerance:
            raise SystemExit(f"Replay mismatch at episode {row['episode']}: {row} vs {actual}")
    print(f"replay ok: {len(rows)} episodes match {monitor}")


def reference(out: Path, episodes: int) -> None:
    if episodes % BLOCK_EPISODES:
        raise SystemExit(f"--episodes must be a multiple of {BLOCK_EPISODES} to keep scenario strata balanced")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"episodes_per_arm": episodes, "bal_weight": BAL_WEIGHT, "arms": {}}
    for arm, (_folder, seed) in ARMS.items():
        rows = collect(arm, seed, episodes)
        with (out / f"reference_episodes_arm{arm}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        n = len(rows)
        S = sum(r["S"] for r in rows) / n
        D = sum(r["D"] for r in rows) / n
        c = sum(r["c"] for r in rows) / n
        weights = solve_weights(S, D)
        returns = {name: c + w["speed"] * S + w["spacing"] * D for name, w in weights.items()}
        bal = [episode_return(r, BAL_WEIGHT, BAL_WEIGHT) for r in rows]
        mean = sum(bal) / n
        se = (sum((x - mean) ** 2 for x in bal) / (n - 1) / n) ** 0.5
        summary["arms"][arm] = dict(
            reference_seed=seed, S=S, D=D, c=c, weights=weights, t0_return=returns,
            bal_episode_standard_error=se, mean_length=sum(r["length"] for r in rows) / n)
    (out / "equal_start_weights.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--arm", choices=list(ARMS), required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--speed", type=float, required=True)
    r.add_argument("--spacing", type=float, required=True)
    r.add_argument("--episodes", type=int, default=8)
    s = sub.add_parser("reference")
    s.add_argument("--out", type=Path, default=ROOT / "outputs" / "reward_equal_start")
    s.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == "replay":
        prefix = "reward_fixed_sum" if args.variant.startswith("F") else "reward_sensitivity"
        run = ROOT / "outputs" / f"{prefix}_arm{args.arm}_{args.variant}_100k_seed{args.seed}"
        monitors = sorted(run.glob("training_logs/*/monitor.csv"))
        if len(monitors) != 1:
            raise SystemExit(f"Expected one monitor.csv under {run}, found {len(monitors)}")
        replay(args.arm, args.seed, args.variant, args.speed, args.spacing, monitors[0], args.episodes)
    else:
        reference(args.out, args.episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
