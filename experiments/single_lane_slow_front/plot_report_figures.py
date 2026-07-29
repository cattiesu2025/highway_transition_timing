"""Generate report-facing figures for the single-lane slow-front experiment."""

from __future__ import annotations

import argparse
import csv
import os
import json
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt


AGENTS = ("FD", "BAL", "SP")
PANEL_AGENTS = ("BAL", "FD", "SP")
AGENT_COLORS = {
    "FD": "#3B7C8F",
    "BAL": "#626B75",
    "SP": "#C77C2B",
}
ACTION_COLORS = {
    "FASTER": "#E58A2A",
    "IDLE": "#8A8A8A",
    "SLOWER": "#5BA04D",
}
ACTION_ORDER = ("FASTER", "IDLE", "SLOWER")
VARIANTS = ("original", "no-front", "matched-speed-front", "far-front")
VARIANT_LABELS = {
    "original": "original",
    "no-front": "no front",
    "matched-speed-front": "matched\nspeed",
    "far-front": "far front",
}
ROLLOUT_VARIANT_LABELS = {
    "original": "slow front",
    "no-front": "no front",
    "matched-speed-front": "matched\nspeed",
    "far-front": "far front",
}


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
        }
    )


def read_episode_latencies(path: Path) -> dict[str, list[float]]:
    latencies = {agent: [] for agent in AGENTS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            agent = row["agent_condition"]
            if agent not in latencies:
                continue
            if row["episode_outcome"] != "valid_onset":
                continue
            if row["response_latency"] == "":
                continue
            latencies[agent].append(float(row["response_latency"]))
    return latencies


def read_exposure_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_rollout_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rollout_outcome_rows(
    rows: list[dict[str, str]],
    variant: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped = {agent: [] for agent in AGENTS}
    for row in rows:
        agent = row["agent_condition"]
        if agent in grouped:
            grouped[agent].append(row)

    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for agent, group in grouped.items():
        observed = [
            float(row["response_latency"])
            for row in group
            if row["episode_outcome"] == "valid_onset"
            and row["response_latency"] != ""
        ]
        summary[(agent, variant)] = {
            "n": len(group),
            "observed": observed,
            "n_observed": len(observed),
            "n_without_onset": len(group) - len(observed),
            "median": median(observed) if observed else None,
        }
    return summary


def read_rollout_counterfactual_episode_summary(
    run_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]] | None:
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    rollout_dir = run_dir / "rollout_counterfactual_front_vehicle"
    for variant in VARIANTS:
        source_path = rollout_dir / variant / "analysis" / "episode_outcomes.csv"
        if not source_path.exists():
            return None
        with source_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary.update(summarize_rollout_outcome_rows(rows, variant))

    if not all(
        (agent, variant) in summary
        for agent in AGENTS
        for variant in VARIANTS
    ):
        return None
    return summary


def save_all(fig: mpl.figure.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_latency_by_agent(run_dir: Path) -> None:
    latencies = read_episode_latencies(run_dir / "analysis" / "episode_outcomes.csv")
    exposure_count = read_exposure_count(run_dir / "evaluation" / "exposures.csv")
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
        label_y = max(med + 1.0, 1.6)
        ax.text(
            idx + 0.28,
            label_y,
            f"med {med:g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222",
        )

    ax.set_title(
        "Slowdown onset latency in matched slow-front scenes",
        fontsize=11,
        pad=8,
    )
    ax.set_ylabel("Slowdown onset latency (policy steps)")
    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels(AGENTS)
    ax.set_ylim(-1, 29)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.text(
        0.01,
        0.98,
        f"{exposure_count} matched exposures per agent",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    save_all(fig, run_dir / "analysis" / "report_slowdown_latency_by_agent")
    plt.close(fig)


def plot_counterfactual_valid_onsets(run_dir: Path) -> None:
    rows = read_rollout_summary(
        run_dir
        / "rollout_counterfactual_front_vehicle"
        / "counterfactual_rollout_summary.csv"
    )
    variants = ("original", "no-front", "far-front")
    values: dict[tuple[str, str], int] = {}
    totals: dict[tuple[str, str], int] = {}
    latencies: dict[tuple[str, str], str] = {}
    for row in rows:
        if row["record_type"] != "agent_outcome":
            continue
        key = (row["counterfactual_variant"], row["agent_condition"])
        values[key] = int(row["valid_onset_count"])
        totals[key] = int(row["n"])
        latencies[key] = row["median_response_latency"]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bar_width = 0.22
    centers = list(range(len(variants)))
    offsets = {"FD": -bar_width, "BAL": 0.0, "SP": bar_width}
    max_total = max(totals.values())

    for agent in AGENTS:
        xs = [center + offsets[agent] for center in centers]
        ys = [values[(variant, agent)] for variant in variants]
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=AGENT_COLORS[agent],
            label=agent,
            alpha=0.88,
        )
        for x_value, y_value, variant in zip(xs, ys, variants, strict=True):
            total = totals[(variant, agent)]
            label = f"{y_value}/{total}"
            text_y = y_value - max(total * 0.05, 0.8) if y_value else max_total * 0.025
            text_color = "white" if y_value else "#222222"
            ax.text(
                x_value,
                text_y,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold" if y_value else "normal",
            )

    ax.set_title("Full-episode counterfactual check", fontsize=11, pad=8)
    ax.set_ylabel("Episodes with stable slowdown onset")
    ax.set_xticks(centers)
    ax.set_xticklabels(["slow front", "no front", "far front"])
    ax.set_ylim(0, max_total * 1.15)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.text(
        0.01,
        0.98,
        "No-front control rules out unconditional slowdown",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    save_all(fig, run_dir / "analysis" / "report_counterfactual_valid_onsets")
    plt.close(fig)


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
    summary = read_rollout_counterfactual_episode_summary(run_dir)
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
        "slowdown_rate",
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
    ax_rate.set_title("Stable-slowdown occurrence", fontsize=9.5, pad=8)
    add_panel_label(ax_rate, "a")

    timing_variants = ("original", "matched-speed-front", "far-front")
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
            if not observed:
                continue
            med = float(median(observed))
            q1 = quantile(observed, 0.25)
            q3 = quantile(observed, 0.75)
            y_value = base_y + offsets[agent]
            color = AGENT_COLORS[agent]
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
                med + 2.2,
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
    ax_time.set_xlim(-5, 120)
    ax_time.set_ylim(-0.55, 2.55)
    ax_time.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax_time.set_xlabel("First stable-slowdown latency (policy steps)", fontsize=8)
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
        "Counterfactual slowdown occurrence and timing",
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
        run_dir / "analysis" / "report_rollout_counterfactual_slowdown_response",
    )
    plt.close(fig)
    return True


def plot_initial_action_counterfactual_rates(run_dir: Path) -> None:
    action_counts = read_initial_action_summary(
        run_dir
        / "counterfactual_front_vehicle"
        / "counterfactual_action_summary.csv"
    )
    sample_sizes = {sum(counts.values()) for counts in action_counts.values()}
    sample_size_label = (
        str(next(iter(sample_sizes))) if len(sample_sizes) == 1 else "varied"
    )
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
    fig.suptitle("Initial action under front-vehicle counterfactuals", fontsize=11, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.24, wspace=0.15)
    fig.text(
        0.48,
        0.02,
        f"n={sample_size_label} matched exposures per agent and condition",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )

    save_all(fig, run_dir / "analysis" / "report_initial_action_counterfactual_rates")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/single_lane_slow_front_mixed_20k_smoke",
        help="Trained single-lane run directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_matplotlib()
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    plot_latency_by_agent(run_dir)
    plot_counterfactual_valid_onsets(run_dir)
    plot_rollout_counterfactual_summary_grid(run_dir)
    plot_initial_action_counterfactual_rates(run_dir)


if __name__ == "__main__":
    main()
