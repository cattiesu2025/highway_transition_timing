"""Generate a compact FD/BAL/SP training diagnostics figure."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt


AGENTS = ("FD", "BAL", "SP")
AGENT_COLORS = {
    "FD": "#3B7C8F",
    "BAL": "#626B75",
    "SP": "#C77C2B",
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


def read_training_metric(
    run_dir: Path,
    agent: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    path = run_dir / "training_logs" / agent / "progress.csv"
    xs: list[float] = []
    ys: list[float] = []
    if not path.exists():
        return xs, ys
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestep = row.get("time/total_timesteps", "")
            value = row.get(metric, "")
            if not timestep or not value:
                continue
            xs.append(float(timestep))
            ys.append(float(value))
    return xs, ys


def plot_run_panel(
    ax: mpl.axes.Axes,
    run_dir: Path,
    title: str,
    metric: str,
) -> None:
    for agent in AGENTS:
        xs, ys = read_training_metric(run_dir, agent, metric)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=AGENT_COLORS[agent],
            linewidth=1.9,
            label=agent,
        )
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            color=AGENT_COLORS[agent],
            s=18,
            linewidth=0,
            zorder=3,
        )
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xlabel("Training timesteps")
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda value, _position: (
                f"{value / 1000:g}K" if abs(value) >= 1000 else f"{value:g}"
            )
        )
    )


def save_all(fig: mpl.figure.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-run-dir",
        default="outputs/single_lane_slow_front_controls_100k",
    )
    parser.add_argument(
        "--multi-run-dir",
        default="outputs/multilane_open_lane_change_mixed_100k",
    )
    parser.add_argument(
        "--out",
        default="outputs/report_figures/training_diagnostics",
        help="Output path without extension.",
    )
    parser.add_argument(
        "--metric",
        default="rollout/ep_rew_mean",
        help="Progress CSV metric to plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
    plot_run_panel(
        axes[0],
        Path(args.single_run_dir),
        "Single-lane slowdown agents",
        args.metric,
    )
    plot_run_panel(
        axes[1],
        Path(args.multi_run_dir),
        "Multi-lane lane-change agents",
        args.metric,
    )
    axes[0].set_ylabel("Mean episode reward")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.suptitle(
        "Training diagnostics for FD / BAL / SP agents",
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        (
            "Diagnostic only: FD, BAL, and SP optimize different reward weights, "
            "so absolute returns are not interpreted as direct performance rankings."
        ),
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.74, bottom=0.2, wspace=0.16)
    save_all(fig, Path(args.out))
    plt.close(fig)


if __name__ == "__main__":
    main()
