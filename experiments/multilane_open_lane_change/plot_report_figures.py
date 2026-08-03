"""Generate report-facing figures for the controlled multi-lane experiment."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt


AGENTS = ("FD", "BAL", "SP")
PANEL_AGENTS = ("BAL", "FD", "SP")
VARIANTS = ("original", "no-front", "matched-speed-front", "far-front")
DEFAULT_NUM_EXPOSURES = 36
VARIANT_LABELS = {
    "original": "original",
    "no-front": "no front",
    "matched-speed-front": "matched\nspeed",
    "far-front": "far front",
}
ROLLOUT_VARIANT_LABELS = {
    **VARIANT_LABELS,
    "original": "slow front",
}
AGENT_COLORS = {
    "FD": "#3B7C8F",
    "BAL": "#626B75",
    "SP": "#C77C2B",
}
COUNTERFACTUAL_COLORS = {
    "original": "#355F7A",
    "no-front": "#A7A9AC",
    "matched-speed-front": "#8A84B8",
    "far-front": "#C88B62",
}
COUNTERFACTUAL_LINESTYLES = {
    "original": "-",
    "no-front": "--",
    "matched-speed-front": "-.",
    "far-front": ":",
}
ACTION_COLORS = {
    "lane_change": "#4C78A8",
    "FASTER": "#E58A2A",
    "IDLE": "#8A8A8A",
    "SLOWER": "#5BA04D",
}
ACTION_ORDER = ("lane_change", "FASTER", "IDLE", "SLOWER")
LANE_CHANGE_ACTIONS = {"LANE_LEFT", "LANE_RIGHT"}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.frameon": False,
        }
    )


def load_experiment_module():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("multilane_open_lane_change_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_all(fig: mpl.figure.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_actual_lane_change_latencies(path: Path) -> dict[str, list[float]]:
    values = {agent: [] for agent in AGENTS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            agent = row["agent_condition"]
            if agent not in values:
                continue
            latency = row["first_actual_lane_change_t"]
            if latency != "":
                values[agent].append(float(latency))
    return values


def plot_lane_change_latency(run_dir: Path) -> None:
    latencies = read_actual_lane_change_latencies(
        run_dir / "analysis" / "actual_lane_change_summary.csv"
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    for idx, agent in enumerate(AGENTS):
        values = latencies[agent]
        color = AGENT_COLORS[agent]
        jitter = [
            (rank - (len(values) - 1) / 2.0) * 0.018 for rank, _ in enumerate(values)
        ]
        ax.scatter(
            [idx + offset for offset in jitter],
            values,
            s=25,
            color=color,
            alpha=0.72,
            linewidth=0,
            zorder=3,
        )
        med = median(values)
        q1 = quantile(values, 0.25)
        q3 = quantile(values, 0.75)
        ax.vlines(idx, q1, q3, color="#222222", linewidth=2.0, zorder=4)
        ax.hlines(med, idx - 0.24, idx + 0.24, color="#222222", linewidth=2.4, zorder=5)
        ax.text(
            idx + 0.28,
            max(med + 1.0, 1.6),
            f"med {med:g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222",
        )

    ax.set_title(
        "First lane-change latency in open-lane slow-front scenes",
        fontsize=11,
        pad=8,
    )
    ax.set_ylabel("First actual lane-change latency (policy steps)")
    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels(AGENTS)
    ax.set_ylim(-1, 38)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    exposure_counts = [len(latencies[agent]) for agent in AGENTS]
    exposure_count_label = (
        str(exposure_counts[0])
        if len(set(exposure_counts)) == 1
        else "/".join(str(value) for value in exposure_counts)
    )
    ax.text(
        0.01,
        0.98,
        f"{exposure_count_label} matched exposures per agent",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    save_all(fig, run_dir / "analysis" / "report_lane_change_latency_by_agent")
    plt.close(fig)


def counterfactual_settings(spec: Any, variant: str) -> dict[str, float | bool | None]:
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


def grouped_action(action_name: str) -> str:
    if action_name in LANE_CHANGE_ACTIONS:
        return "lane_change"
    return action_name


def evaluate_initial_action_counterfactuals(
    run_dir: Path,
    output_dir: Path,
    num_exposures: int = DEFAULT_NUM_EXPOSURES,
) -> list[dict[str, Any]]:
    module = load_experiment_module()
    config = module.ExperimentConfig()
    dqn_class = module.dqn_class_for_variant(config.dqn_variant)
    models = {
        agent: dqn_class.load(run_dir / "models" / f"{agent}_main.zip")
        for agent in AGENTS
    }
    rows: list[dict[str, Any]] = []
    specs = module.make_eval_specs(num_exposures, config)

    for spec in specs:
        for variant in VARIANTS:
            settings = counterfactual_settings(spec, variant)
            for agent in AGENTS:
                env = module.make_env(agent, config, training=False)
                obs, _info = env.reset(seed=spec.exposure_seed)
                obs = module.apply_open_lane_scene(
                    env,
                    spec,
                    include_front_vehicle=bool(settings["include_front_vehicle"]),
                    front_distance=settings["front_distance"],
                    front_speed=settings["front_speed"],
                )
                diagnostics = module.extract_diagnostics(env)
                action, q_values = module.predict_action_and_scores(
                    models[agent],
                    obs,
                    deterministic=True,
                )
                action_name = module.action_name_from_env(env, int(action))
                rows.append(
                    {
                        "exposure_id": spec.exposure_id,
                        "exposure_seed": spec.exposure_seed,
                        "agent_condition": agent,
                        "policy_id": f"{agent}_main",
                        "counterfactual_variant": variant,
                        "include_front_vehicle": settings["include_front_vehicle"],
                        "configured_front_distance": module.finite_or_blank(
                            settings["front_distance"]
                        ),
                        "configured_front_speed": module.finite_or_blank(
                            settings["front_speed"]
                        ),
                        "ego_speed": round(float(diagnostics["ego_speed"]), 6),
                        "nearest_front_distance": module.finite_or_blank(
                            diagnostics["nearest_front_distance"]
                        ),
                        "front_vehicle_speed": module.finite_or_blank(
                            diagnostics["front_vehicle_speed"]
                        ),
                        "action": action_name,
                        "action_group": grouped_action(action_name),
                        "is_lane_change_action": action_name in LANE_CHANGE_ACTIONS,
                        "q_values_or_action_scores": [
                            round(float(value), 6) for value in q_values
                        ],
                    }
                )
                env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "counterfactual_initial_actions.csv", rows)
    summary = summarize_initial_action_rows(rows)
    write_csv(output_dir / "counterfactual_action_summary.csv", summary)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_initial_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["agent_condition"]), str(row["counterfactual_variant"])),
            [],
        ).append(row)

    summary: list[dict[str, Any]] = []
    for (agent, variant), group in sorted(grouped.items()):
        counts = Counter(str(row["action_group"]) for row in group)
        lane_change = counts.get("lane_change", 0)
        summary.append(
            {
                "agent_condition": agent,
                "counterfactual_variant": variant,
                "n": len(group),
                "lane_change_action_count": lane_change,
                "lane_change_action_rate": round(lane_change / len(group), 6)
                if group
                else "",
                "action_counts": json.dumps(dict(sorted(counts.items()))),
            }
        )
    return summary


def read_initial_action_summary(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    action_counts: dict[tuple[str, str], dict[str, int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["agent_condition"], row["counterfactual_variant"])
            action_counts[key] = {
                str(action): int(count)
                for action, count in json.loads(row["action_counts"]).items()
            }
    return action_counts


def initial_action_summary_matches_exposure_count(
    path: Path,
    expected_n: int,
) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and all(int(row["n"]) == expected_n for row in rows)


def plot_initial_action_counterfactual_rates(run_dir: Path) -> None:
    cf_dir = run_dir / "counterfactual_open_lane"
    summary_path = cf_dir / "counterfactual_action_summary.csv"
    if not initial_action_summary_matches_exposure_count(
        summary_path,
        DEFAULT_NUM_EXPOSURES,
    ):
        evaluate_initial_action_counterfactuals(
            run_dir,
            cf_dir,
            num_exposures=DEFAULT_NUM_EXPOSURES,
        )

    action_counts = read_initial_action_summary(summary_path)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)

    for ax, agent in zip(axes, PANEL_AGENTS, strict=True):
        bottoms = [0.0 for _ in VARIANTS]
        for action in ACTION_ORDER:
            rates = []
            for variant in VARIANTS:
                counts = action_counts[(agent, variant)]
                total = sum(counts.values())
                rates.append(counts.get(action, 0) / total if total else 0.0)
            ax.bar(
                range(len(VARIANTS)),
                rates,
                bottom=bottoms,
                color=ACTION_COLORS[action],
                width=0.78,
                label=action,
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms = [bottom + rate for bottom, rate in zip(bottoms, rates, strict=True)]

        ax.set_title(agent, fontsize=10, pad=5)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([VARIANT_LABELS[variant] for variant in VARIANTS], fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, alpha=0.75)
        ax.tick_params(axis="y", labelsize=7)

    axes[0].set_ylabel("Initial action rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.91, 0.52),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Initial action under open-lane front-vehicle counterfactuals", fontsize=11, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.24, wspace=0.15)
    save_all(fig, run_dir / "analysis" / "report_initial_action_counterfactual_rates")
    plt.close(fig)


def summarize_rollout_counterfactual_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["agent_condition"], row["counterfactual_variant"])
        grouped.setdefault(key, []).append(row)

    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        use_seconds = all(
            "first_actual_lane_change_seconds" in row for row in group
        )
        event_key = (
            "first_actual_lane_change_seconds"
            if use_seconds
            else "first_actual_lane_change_t"
        )
        observed = [
            float(row[event_key])
            for row in group
            if row[event_key] != ""
        ]
        horizon_values = (
            [
                float(row["recording_horizon_seconds"])
                for row in group
                if row.get("recording_horizon_seconds", "") != ""
            ]
            if use_seconds
            else [
                float(row["terminal_t"]) + 1.0
                for row in group
                if row.get("terminal_t", "") != ""
            ]
        )
        summary[key] = {
            "n": len(group),
            "observed": observed,
            "n_observed": len(observed),
            "n_censored": len(group) - len(observed),
            "median": median(observed) if observed else None,
            "time_unit": "seconds" if use_seconds else "policy_steps",
            "horizon": max(horizon_values) if horizon_values else 120.0,
        }
    return summary


def read_rollout_counterfactual_summary(
    run_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]] | None:
    source_path = (
        run_dir
        / "rollout_counterfactual_open_lane"
        / "counterfactual_episode_summary.csv"
    )
    if not source_path.exists():
        return None
    with source_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_rollout_counterfactual_rows(rows)
    if not all((agent, variant) in summary for agent in AGENTS for variant in VARIANTS):
        return None
    return summary


def cumulative_incidence_points(
    observed: list[float],
    n_episodes: int,
    horizon: float,
) -> tuple[list[float], list[float]]:
    ordered = sorted(float(value) for value in observed)
    x_values = [0.0, *ordered, float(horizon)]
    y_values = [
        0.0,
        *[
            index / n_episodes
            for index in range(1, len(ordered) + 1)
        ],
        len(ordered) / n_episodes,
    ]
    return x_values, y_values


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#222222",
    )


def plot_rollout_counterfactual_summary_grid(run_dir: Path) -> bool:
    summary = read_rollout_counterfactual_summary(run_dir)
    if summary is None:
        return False

    fig = plt.figure(figsize=(7.2, 3.25))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.82, 1.55],
        left=0.085,
        right=0.985,
        top=0.80,
        bottom=0.23,
        wspace=0.34,
    )
    ax_rate = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])

    rate_matrix = [
        [
            summary[(agent, variant)]["n_observed"]
            / summary[(agent, variant)]["n"]
            for agent in AGENTS
        ]
        for variant in VARIANTS
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "lane_change_rate",
        ["#F4F5F6", "#AEBBD0", "#355F7A"],
    )
    ax_rate.imshow(
        rate_matrix,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index, variant in enumerate(VARIANTS):
        for column_index, agent in enumerate(AGENTS):
            cell = summary[(agent, variant)]
            rate = cell["n_observed"] / cell["n"]
            ax_rate.text(
                column_index,
                row_index,
                f"{cell['n_observed']}/{cell['n']}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color="white" if rate >= 0.65 else "#27313A",
            )
    ax_rate.set_xticks(range(len(AGENTS)))
    ax_rate.set_xticklabels(AGENTS, fontsize=7.5)
    ax_rate.set_yticks(range(len(VARIANTS)))
    ax_rate.set_yticklabels(
        [ROLLOUT_VARIANT_LABELS[variant] for variant in VARIANTS],
        fontsize=7.3,
    )
    ax_rate.tick_params(length=0)
    for spine in ax_rate.spines.values():
        spine.set_visible(False)
    ax_rate.set_title("Lane-change occurrence", fontsize=9.5, pad=8)
    add_panel_label(ax_rate, "a")

    timing_variants = ("original", "matched-speed-front", "far-front")
    time_unit = str(summary[(AGENTS[0], VARIANTS[0])]["time_unit"])
    horizon = max(
        float(summary[(agent, variant)]["horizon"])
        for agent in AGENTS
        for variant in VARIANTS
    )
    y_positions = {
        "original": 2.0,
        "matched-speed-front": 1.0,
        "far-front": 0.0,
    }
    offsets = {"FD": 0.20, "BAL": 0.0, "SP": -0.20}
    markers = {"FD": "o", "BAL": "s", "SP": "D"}
    for variant in timing_variants:
        base_y = y_positions[variant]
        for agent in AGENTS:
            observed = [
                float(value)
                for value in summary[(agent, variant)]["observed"]
            ]
            y_value = base_y + offsets[agent]
            color = AGENT_COLORS[agent]
            if not observed:
                ax_time.text(
                    0.02 * horizon,
                    y_value,
                    "no event",
                    ha="left",
                    va="center",
                    fontsize=6.8,
                    color=color,
                )
                continue
            med = float(median(observed))
            q1 = quantile(observed, 0.25)
            q3 = quantile(observed, 0.75)
            ax_time.hlines(
                y_value,
                q1,
                q3,
                color=color,
                linewidth=2.4,
                alpha=0.72,
                zorder=2,
            )
            ax_time.scatter(
                [med],
                [y_value],
                s=38,
                marker=markers[agent],
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                label=agent if variant == "original" else "_nolegend_",
            )
            ax_time.text(
                med + 0.02 * horizon,
                y_value,
                f"{med:g}",
                ha="left",
                va="center",
                fontsize=6.8,
                color=color,
            )

    ax_time.set_yticks([2.0, 1.0, 0.0])
    ax_time.set_yticklabels(
        [
            ROLLOUT_VARIANT_LABELS["original"],
            ROLLOUT_VARIANT_LABELS["matched-speed-front"],
            ROLLOUT_VARIANT_LABELS["far-front"],
        ],
        fontsize=7.5,
    )
    ax_time.set_xlim(0, horizon)
    ax_time.set_ylim(-0.55, 2.55)
    ax_time.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    ax_time.set_xlabel(
        "First actual lane-change latency (seconds)"
        if time_unit == "seconds"
        else "First actual lane-change latency (policy steps)",
        fontsize=8,
    )
    ax_time.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax_time.set_axisbelow(True)
    ax_time.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    no_front_cells = [summary[(agent, "no-front")] for agent in AGENTS]
    no_front_label = (
        f"No front: {no_front_cells[0]['n_observed']}/{no_front_cells[0]['n']} "
        "events for every agent"
        if len(
            {
                (cell["n_observed"], cell["n"])
                for cell in no_front_cells
            }
        )
        == 1
        else "No front: see occurrence matrix"
    )
    ax_time.text(
        0.02,
        0.02,
        no_front_label,
        transform=ax_time.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#666666",
    )
    ax_time.set_title("Timing among observed events", fontsize=9.5, pad=8)
    add_panel_label(ax_time, "b")

    fig.suptitle(
        "Counterfactual lane-change occurrence and timing",
        fontsize=11,
        y=0.97,
    )
    exposure_counts = {
        int(summary[(agent, variant)]["n"])
        for agent in AGENTS
        for variant in VARIANTS
    }
    exposure_count_label = (
        str(next(iter(exposure_counts)))
        if len(exposure_counts) == 1
        else "varying"
    )
    fig.text(
        0.5,
        0.04,
        (
            f"n={exposure_count_label} matched exposures per agent and condition. "
            "Points show conditional medians; horizontal ranges show IQR."
        ),
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#555555",
    )
    save_all(
        fig,
        run_dir / "analysis" / "report_rollout_counterfactual_lane_change_response",
    )
    plt.close(fig)
    return True


def plot_rollout_counterfactual_cumulative_incidence(run_dir: Path) -> bool:
    summary = read_rollout_counterfactual_summary(run_dir)
    if summary is None:
        return False

    horizon = max(
        float(summary[(agent, variant)]["horizon"])
        for agent in AGENTS
        for variant in VARIANTS
    )
    time_unit = str(summary[(AGENTS[0], VARIANTS[0])]["time_unit"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True, sharey=True)
    for ax, agent in zip(axes, AGENTS, strict=True):
        for variant in VARIANTS:
            cell = summary[(agent, variant)]
            x_values, y_values = cumulative_incidence_points(
                list(cell["observed"]),
                int(cell["n"]),
                horizon,
            )
            ax.step(
                x_values,
                y_values,
                where="post",
                color=COUNTERFACTUAL_COLORS[variant],
                linestyle=COUNTERFACTUAL_LINESTYLES[variant],
                linewidth=2.0 if variant == "original" else 1.7,
                label=(
                    f"{ROLLOUT_VARIANT_LABELS[variant].replace(chr(10), ' ')} "
                    f"({cell['n_observed']}/{cell['n']})"
                    if agent == AGENTS[0]
                    else "_nolegend_"
                ),
                zorder=3,
            )

        ax.set_title(agent, fontsize=9.5, pad=6)
        ax.set_xlim(0, horizon)
        ax.set_ylim(-0.035, 1.035)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cumulative lane-change incidence", fontsize=8)
    axes[1].set_xlabel(
        "Seconds after exposure"
        if time_unit == "seconds"
        else "Policy steps after exposure",
        fontsize=8,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        fontsize=6.8,
        handlelength=2.4,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.suptitle(
        "Lane-change incidence under front-vehicle counterfactuals",
        fontsize=11,
        y=0.98,
    )
    exposure_counts = {
        int(summary[(agent, variant)]["n"])
        for agent in AGENTS
        for variant in VARIANTS
    }
    exposure_count_label = (
        str(next(iter(exposure_counts)))
        if len(exposure_counts) == 1
        else "varying"
    )
    fig.text(
        0.5,
        0.025,
        (
            "Actual lane-index changes; "
            f"n={exposure_count_label} matched exposures per condition. "
            "Episodes without an event are right-censored at "
            f"{horizon:g} {'seconds' if time_unit == 'seconds' else 'steps'}."
        ),
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.70,
        bottom=0.22,
        wspace=0.14,
    )
    save_all(
        fig,
        run_dir / "analysis" / "report_rollout_counterfactual_cumulative_incidence",
    )
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/multilane_open_lane_change_mixed_20k_smoke",
        help="Trained multi-lane run directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_matplotlib()
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    plot_lane_change_latency(run_dir)
    plot_initial_action_counterfactual_rates(run_dir)
    plot_rollout_counterfactual_summary_grid(run_dir)
    plot_rollout_counterfactual_cumulative_incidence(run_dir)


if __name__ == "__main__":
    main()
