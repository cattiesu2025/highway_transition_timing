#!/usr/bin/env python3
"""Locate each policy's decision boundary on a shared synthetic distance sweep.

SUPERSEDED. This probe is retained as a record of an approach that did not work.
It reported SP crossing 18.1 m nearer than FD in the two-lane setting, with 16
of 20 seeds in that direction. The uncensored behavioural analysis in
`analyse_onset_states.py` gives 5.2 m with an 11/9 sign split, so the probe
disagrees with the rollouts in direction, not only in size. A single-step argmax
over Q values is not the same construct as a completion-confirmed onset, and the
synthetic state is very likely off the policy's visited distribution. Do not use
its numbers for any behavioural claim.

Trigger distances read off rollouts are censored by the evaluation grid: a scene
that starts at 90 m cannot record a trigger at 120 m, so the observed value is
pinned to the grid level. This script avoids that entirely. It places a front
vehicle at each distance on a common sweep, queries the policy once, and reads
the decision margin

    single lane  Q(SLOWER)    - max(Q(IDLE), Q(FASTER))
    two lanes    Q(LANE_LEFT) - max(Q of every other action)

The distance where the margin crosses zero is the policy's own threshold,
measured on the same sweep for both experiments and independent of which
scenario grid was used to evaluate it.

Nothing is trained and nothing is written to a maintained run directory. The
sealed held-out grid is never loaded.

Run from the repository root:

    python3 scripts/probe_decision_boundary.py --seeds 3000 4000
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aggregate_selected_20seed import AGENTS, quartiles, write_csv  # noqa: E402

# A common physical sweep, covering both grids and both training ranges.
SWEEP_DISTANCES = tuple(float(d) for d in range(60, 205, 5))
PROBE_EGO_SPEED = 28.0
PROBE_FRONT_SPEED = 14.0

EXPERIMENTS = {
    "single_lane": {
        "module_path": REPO_ROOT / "experiments" / "single_lane_slow_front" / "run.py",
        "run_prefix": "single_lane_slow_front_selected_100k_seed",
        "seeds": tuple(range(3000, 3020)),
        # index -> name, from the longitudinal-only action set
        "target_action": 0,  # SLOWER
        "action_names": ("SLOWER", "IDLE", "FASTER"),
    },
    "twolane": {
        "module_path": REPO_ROOT
        / "experiments"
        / "multilane_open_lane_change"
        / "run.py",
        "run_prefix": "multilane_open_lane_change_twolane_selected_100k_seed",
        "seeds": tuple(range(4000, 4020)),
        "target_action": 0,  # LANE_LEFT
        "action_names": ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"),
    },
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config_for_run(module, run_dir: Path, key: str):
    with (run_dir / "training_runs.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    kwargs: dict[str, Any] = {
        "seed": int(row["seed"]),
        "policy_frequency": int(row["policy_frequency_hz"]),
        "evaluation_duration": int(float(row["evaluation_duration_seconds"])),
    }
    if key == "twolane":
        kwargs["lanes_count"] = int(row["lanes_count"])
        kwargs["ego_lane"] = int(row["ego_lane"])
    return module.ExperimentConfig(**kwargs)


def build_spec(module, key: str, config, distance: float):
    common = dict(
        exposure_id="PROBE",
        exposure_seed=config.seed,
        ego_speed=PROBE_EGO_SPEED,
        front_distance=distance,
        front_speed=PROBE_FRONT_SPEED,
        include_front_vehicle=True,
    )
    if key == "twolane":
        return module.build_open_lane_spec(
            scenario_type="decision_boundary_probe",
            ego_lane=config.ego_lane,
            **common,
        )
    return module.build_single_lane_spec(
        scenario_type="decision_boundary_probe",
        **common,
    )


def apply_scene(module, key: str, env, spec):
    if key == "twolane":
        return module.apply_open_lane_scene(env, spec)
    return module.apply_single_lane_scene(env, spec)


def margins_for_model(module, key: str, config, agent: str, model) -> list[float]:
    import torch

    margins: list[float] = []
    target = EXPERIMENTS[key]["target_action"]
    for distance in SWEEP_DISTANCES:
        env = module.make_env(agent, config, training=False)
        env.reset(seed=config.seed)
        observation = apply_scene(module, key, env, build_spec(module, key, config, distance))
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            q = model.q_net(obs_tensor).detach().cpu().numpy()[0]
        others = [float(v) for i, v in enumerate(q) if i != target]
        margins.append(float(q[target]) - max(others))
        env.close()
    return margins


def crossing_distance(margins: Sequence[float]) -> float | str:
    """Largest distance at which the margin turns positive, linearly interpolated.

    The sweep runs from near to far, so we look for the last index where the
    margin is still positive and interpolate against the next negative one.
    """
    for index in range(len(margins) - 1):
        near, far = margins[index], margins[index + 1]
        if near > 0.0 >= far:
            d_near, d_far = SWEEP_DISTANCES[index], SWEEP_DISTANCES[index + 1]
            if near == far:
                return d_near
            return d_near + (d_far - d_near) * (near / (near - far))
    if margins[0] <= 0.0:
        return "never_positive"
    return "always_positive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/decision_boundary_probe")
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    print(
        f"sweep {SWEEP_DISTANCES[0]:.0f}-{SWEEP_DISTANCES[-1]:.0f} m "
        f"step {SWEEP_DISTANCES[1] - SWEEP_DISTANCES[0]:.0f} m, "
        f"ego {PROBE_EGO_SPEED} m/s, front {PROBE_FRONT_SPEED} m/s"
    )

    rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for key, spec in EXPERIMENTS.items():
        module = load_module(spec["module_path"], f"probe_{key}")
        seeds = [s for s in spec["seeds"] if args.seeds is None or s in args.seeds]
        crossings: dict[str, list[float]] = {agent: [] for agent in AGENTS}
        for seed in seeds:
            run_dir = args.outputs_root / f"{spec['run_prefix']}{seed}"
            config = config_for_run(module, run_dir, key)
            dqn_class = module.dqn_class_for_variant(config.dqn_variant)
            for agent in AGENTS:
                model = dqn_class.load(run_dir / "models" / f"{agent}_main.zip")
                margins = margins_for_model(module, key, config, agent, model)
                cross = crossing_distance(margins)
                if isinstance(cross, float):
                    crossings[agent].append(cross)
                rows.append(
                    {
                        "experiment": key,
                        "seed": seed,
                        "agent_condition": agent,
                        "crossing_distance_m": (
                            round(cross, 2) if isinstance(cross, float) else cross
                        ),
                    }
                )
                for distance, margin in zip(SWEEP_DISTANCES, margins):
                    margin_rows.append(
                        {
                            "experiment": key,
                            "seed": seed,
                            "agent_condition": agent,
                            "front_distance_m": distance,
                            "decision_margin": round(margin, 6),
                        }
                    )
            print(f"  {key} seed {seed}: " + " ".join(
                f"{a}={rows[-3 + i]['crossing_distance_m']}" for i, a in enumerate(AGENTS)
            ))
        for agent in AGENTS:
            values = crossings[agent]
            if not values:
                continue
            q1, median, q3 = quartiles(values)
            summary.append(
                {
                    "experiment": key,
                    "agent_condition": agent,
                    "n_seeds_with_crossing": len(values),
                    "median_crossing_m": round(median, 2),
                    "q1_m": round(q1, 2),
                    "q3_m": round(q3, 2),
                    "iqr_width_m": round(q3 - q1, 2),
                }
            )

    out_dir = args.out
    write_csv(
        out_dir / "crossing_per_model.csv",
        ["experiment", "seed", "agent_condition", "crossing_distance_m"],
        rows,
    )
    write_csv(
        out_dir / "decision_margin_sweep.csv",
        ["experiment", "seed", "agent_condition", "front_distance_m", "decision_margin"],
        margin_rows,
    )
    write_csv(out_dir / "summary.csv", list(summary[0].keys()), summary)

    print()
    print(f"{'experiment':<12}{'agent':<6}{'n':>4}{'median(m)':>12}{'IQR':>20}{'width':>8}")
    for row in summary:
        iqr = f"[{row['q1_m']:.1f}, {row['q3_m']:.1f}]"
        print(
            f"{row['experiment']:<12}{row['agent_condition']:<6}"
            f"{row['n_seeds_with_crossing']:>4}{row['median_crossing_m']:>12.1f}"
            f"{iqr:>20}{row['iqr_width_m']:>8.1f}"
        )
    print(f"\nwrote: {out_dir}")


if __name__ == "__main__":
    main()
