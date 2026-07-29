"""Run full-episode front-vehicle counterfactuals for trained multi-lane agents."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any


AGENTS = ("FD", "BAL", "SP")
VARIANTS = ("original", "no-front", "matched-speed-front", "far-front")
LANE_CHANGE_ACTIONS = {"LANE_LEFT", "LANE_RIGHT"}


def load_experiment_module():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("multilane_open_lane_change_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed training run.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: RUN_DIR/rollout_counterfactual_open_lane).",
    )
    parser.add_argument("--num-exposures", type=int, default=36)
    parser.add_argument("--evaluation-duration", type=int, default=120)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=["original", "no-front"],
    )
    return parser


def parse_bool(value: str, default: bool) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def config_from_training_run(
    module,
    run_dir: Path,
    evaluation_duration: int,
):
    path = run_dir / "training_runs.csv"
    if not path.exists():
        return module.ExperimentConfig(evaluation_duration=evaluation_duration)
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    collision_penalty = row.get("reward_common_collision_penalty", "")
    return module.ExperimentConfig(
        evaluation_duration=evaluation_duration,
        seed=int(row.get("seed", 0)),
        observation_normalize=parse_bool(
            row.get("observation_normalize", ""),
            True,
        ),
        observation_absolute=parse_bool(
            row.get("observation_absolute", ""),
            False,
        ),
        slow_down_penalty=float(row.get("reward_slow_down_penalty", 0.2)),
        collision_risk_penalty=float(
            row.get("reward_collision_risk_penalty", 3.0)
        ),
        collision_penalty=(
            float(collision_penalty) if collision_penalty not in (None, "") else None
        ),
        dqn_variant=row.get("dqn_variant", "double-dqn"),
    )


def counterfactual_settings(
    spec,
    variant: str,
) -> dict[str, float | bool | None]:
    if variant == "original":
        return {
            "include_front_vehicle": True,
            "front_distance": spec.front_distance,
            "front_speed": spec.front_speed,
        }
    if variant == "no-front":
        return {
            "include_front_vehicle": False,
            "front_distance": None,
            "front_speed": None,
        }
    if variant == "matched-speed-front":
        return {
            "include_front_vehicle": True,
            "front_distance": spec.front_distance,
            "front_speed": spec.ego_speed,
        }
    if variant == "far-front":
        return {
            "include_front_vehicle": True,
            "front_distance": max(360.0, float(spec.front_distance)),
            "front_speed": spec.front_speed,
        }
    raise ValueError(f"Unsupported counterfactual variant: {variant}")


def rollout_variant(
    module,
    models: Mapping[str, Any],
    agents: Sequence[str],
    specs: Sequence[Any],
    config,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []

    for spec in specs:
        exposure_recorded = False
        settings = counterfactual_settings(spec, variant)
        for agent in agents:
            env = module.make_env(agent, config, training=False)
            obs, _info = env.reset(seed=spec.exposure_seed)
            obs = module.apply_open_lane_scene(
                env,
                spec,
                include_front_vehicle=bool(settings["include_front_vehicle"]),
                front_distance=settings["front_distance"],
                front_speed=settings["front_speed"],
            )
            if not exposure_recorded:
                exposure_row = module.exposure_row_from_spec(env, spec, config)
                exposure_row.update(
                    {
                        "counterfactual_variant": variant,
                        "include_front_vehicle": settings["include_front_vehicle"],
                        "configured_front_distance": module.finite_or_blank(
                            settings["front_distance"]
                        ),
                        "configured_front_speed": module.finite_or_blank(
                            settings["front_speed"]
                        ),
                    }
                )
                exposure_rows.append(exposure_row)
                exposure_recorded = True

            episode_id = f"{agent}_{variant}_{spec.exposure_id}_r0"
            done = False
            t = 0
            lane_change_count = 0
            while not done and t < config.evaluation_duration:
                action, q_values = module.predict_action_and_scores(
                    models[agent],
                    obs,
                    deterministic=True,
                )
                action_name = module.action_name_from_env(env, int(action))
                if action_name in LANE_CHANGE_ACTIONS:
                    lane_change_count += 1

                pre_step = module.extract_diagnostics(env)
                next_obs, reward, terminated, truncated, _info = env.step(int(action))
                post_step = module.extract_diagnostics(env)
                done = bool(terminated or truncated)
                collision = bool(post_step["collision_flag"])
                row = module.step_row_from_diagnostics(
                    diagnostics=pre_step,
                    post_step_diagnostics=post_step,
                    agent_condition=agent,
                    policy_id=f"{agent}_main",
                    episode_id=episode_id,
                    exposure_id=spec.exposure_id,
                    exposure_seed=spec.exposure_seed,
                    rollout_id="r0",
                    rollout_seed=0,
                    t=t,
                    action_name=action_name,
                    action_scores=q_values,
                    reward_total=reward,
                    reward_components=getattr(env, "last_reward_components", {}),
                    collision_flag=collision,
                    done=done,
                    termination_reason=(
                        "collision" if collision else ("duration" if done else "")
                    ),
                    lane_change_count=lane_change_count,
                    exposure_t=spec.exposure_t,
                )
                row.update(
                    {
                        "post_ego_lane": post_step["ego_lane"],
                        "lane_delta_after_step": (
                            int(post_step["ego_lane"]) - int(pre_step["ego_lane"])
                        ),
                        "actual_lane_changed_after_step": (
                            int(post_step["ego_lane"]) != int(pre_step["ego_lane"])
                        ),
                        "counterfactual_variant": variant,
                        "include_front_vehicle": settings["include_front_vehicle"],
                    }
                )
                step_rows.append(row)
                obs = next_obs
                t += 1
            env.close()

    return step_rows, exposure_rows


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_median(values: Sequence[int]) -> float | str:
    return median(values) if values else ""


def optional_mean(values: Sequence[int]) -> float | str:
    return round(mean(values), 6) if values else ""


def summarize_variant(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    steps_by_agent: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        steps_by_agent.setdefault(str(row["agent_condition"]), []).append(row)

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault(str(row["agent_condition"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for agent, rows in sorted(grouped.items()):
        first_action = [
            value
            for row in rows
            if (value := optional_int(row["first_lane_change_action_t"])) is not None
        ]
        first_actual = [
            value
            for row in rows
            if (value := optional_int(row["first_actual_lane_change_t"])) is not None
        ]
        action_counts = Counter(
            str(row["action"]) for row in steps_by_agent.get(agent, [])
        )
        summary.append(
            {
                "counterfactual_variant": variant,
                "agent_condition": agent,
                "n_episodes": len(rows),
                "lane_change_action_episode_count": len(first_action),
                "actual_lane_change_episode_count": len(first_actual),
                "collision_episode_count": sum(
                    str(row["collision_flag"]).lower() == "true" for row in rows
                ),
                "median_first_lane_change_action_t": optional_median(first_action),
                "mean_first_lane_change_action_t": optional_mean(first_action),
                "median_first_actual_lane_change_t": optional_median(first_actual),
                "mean_first_actual_lane_change_t": optional_mean(first_actual),
                "action_counts": module_json_dumps(dict(sorted(action_counts.items()))),
            }
        )
    return summary


def module_json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def paired_variant_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    baseline: str = "original",
    counterfactual: str = "no-front",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (
            str(row["agent_condition"]),
            str(row["exposure_id"]),
            str(row["counterfactual_variant"]),
        ): row
        for row in episode_rows
    }
    agents = sorted({key[0] for key in indexed})
    exposures = sorted({key[1] for key in indexed})
    details: list[dict[str, Any]] = []

    for agent in agents:
        for exposure_id in exposures:
            original = indexed.get((agent, exposure_id, baseline))
            no_front = indexed.get((agent, exposure_id, counterfactual))
            if original is None or no_front is None:
                continue
            original_t = optional_int(original["first_actual_lane_change_t"])
            no_front_t = optional_int(no_front["first_actual_lane_change_t"])
            details.append(
                {
                    "agent_condition": agent,
                    "exposure_id": exposure_id,
                    "baseline_variant": baseline,
                    "counterfactual_variant": counterfactual,
                    "baseline_first_actual_lane_change_t": (
                        original_t if original_t is not None else ""
                    ),
                    "counterfactual_first_actual_lane_change_t": (
                        no_front_t if no_front_t is not None else ""
                    ),
                    "latency_delta_counterfactual_minus_baseline": (
                        no_front_t - original_t
                        if original_t is not None and no_front_t is not None
                        else ""
                    ),
                    "baseline_actual_lane_change": original_t is not None,
                    "counterfactual_actual_lane_change": no_front_t is not None,
                }
            )

    summaries: list[dict[str, Any]] = []
    for agent in agents:
        rows = [row for row in details if row["agent_condition"] == agent]
        deltas = [
            int(row["latency_delta_counterfactual_minus_baseline"])
            for row in rows
            if row["latency_delta_counterfactual_minus_baseline"] != ""
        ]
        summaries.append(
            {
                "agent_condition": agent,
                "baseline_variant": baseline,
                "counterfactual_variant": counterfactual,
                "n_pairs": len(rows),
                "baseline_actual_lane_change_count": sum(
                    bool(row["baseline_actual_lane_change"]) for row in rows
                ),
                "counterfactual_actual_lane_change_count": sum(
                    bool(row["counterfactual_actual_lane_change"]) for row in rows
                ),
                "both_actual_lane_change_count": len(deltas),
                "baseline_only_count": sum(
                    bool(row["baseline_actual_lane_change"])
                    and not bool(row["counterfactual_actual_lane_change"])
                    for row in rows
                ),
                "counterfactual_only_count": sum(
                    not bool(row["baseline_actual_lane_change"])
                    and bool(row["counterfactual_actual_lane_change"])
                    for row in rows
                ),
                "neither_count": sum(
                    not bool(row["baseline_actual_lane_change"])
                    and not bool(row["counterfactual_actual_lane_change"])
                    for row in rows
                ),
                "median_latency_delta_counterfactual_minus_baseline": (
                    optional_median(deltas)
                ),
                "mean_latency_delta_counterfactual_minus_baseline": (
                    optional_mean(deltas)
                ),
            }
        )
    return details, summaries


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module = load_experiment_module()
    module.configure_headless_runtime()

    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.out)
        if args.out is not None
        else run_dir / "rollout_counterfactual_open_lane"
    )
    config = config_from_training_run(
        module,
        run_dir,
        evaluation_duration=args.evaluation_duration,
    )
    module.require_highway_deps(include_training=True)
    dqn_class = module.dqn_class_for_variant(config.dqn_variant)
    models = {
        agent: dqn_class.load(run_dir / "models" / f"{agent}_main.zip")
        for agent in args.agents
    }
    specs = module.make_eval_specs(args.num_exposures, config)

    all_episode_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    for variant in args.variants:
        steps, exposures = rollout_variant(
            module,
            models,
            args.agents,
            specs,
            config,
            variant,
        )
        variant_dir = output_dir / variant
        module.write_csv_rows(variant_dir / "evaluation" / "steps.csv", steps)
        module.write_csv_rows(variant_dir / "evaluation" / "exposures.csv", exposures)
        episodes = module.summarize_actual_lane_changes(steps)
        for row in episodes:
            row["counterfactual_variant"] = variant
        module.write_csv_rows(
            variant_dir / "analysis" / "actual_lane_change_summary.csv",
            episodes,
        )
        all_episode_rows.extend(episodes)
        all_summary_rows.extend(summarize_variant(episodes, steps, variant))

    module.write_csv_rows(output_dir / "counterfactual_rollout_summary.csv", all_summary_rows)
    module.write_csv_rows(output_dir / "counterfactual_episode_summary.csv", all_episode_rows)

    if {"original", "no-front"}.issubset(args.variants):
        paired_rows, paired_summary = paired_variant_rows(all_episode_rows)
        module.write_csv_rows(output_dir / "paired_original_no_front.csv", paired_rows)
        module.write_csv_rows(
            output_dir / "paired_original_no_front_summary.csv",
            paired_summary,
        )
        for row in paired_summary:
            print(
                f"{row['agent_condition']}: "
                f"original={row['baseline_actual_lane_change_count']}/{row['n_pairs']}, "
                f"no-front={row['counterfactual_actual_lane_change_count']}/{row['n_pairs']}, "
                "median_delta="
                f"{row['median_latency_delta_counterfactual_minus_baseline']}"
            )
    print(f"Wrote full-rollout counterfactual analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
