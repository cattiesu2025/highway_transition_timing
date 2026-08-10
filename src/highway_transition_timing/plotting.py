"""Optional matplotlib figures for the analysis outputs."""

from __future__ import annotations

import os
import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import (
    NO_ONSET_CENSORED,
    TERMINAL_FAILURE,
    VALID_ONSET,
    VALID_PAIR,
)
from .utils import get_float, get_int, get_str


def write_optional_figures(
    output_dir: str | Path,
    gaps: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Write figures if matplotlib is installed; otherwise return an empty list."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / ".cache"))

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        return []

    written = [
        plot_paired_timing_gaps(gaps, out / "paired_timing_gaps.png"),
        plot_outcome_stacked_bar(outcomes, out / "outcome_stacked_bar.png"),
    ]
    return [path for path in written if path is not None]


def write_training_convergence_figures(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Write reward, length, and loss convergence figures for a training run."""

    root = Path(run_dir)
    out = Path(output_dir) if output_dir is not None else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / ".cache"))

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        return []

    rows_by_agent = read_training_progress_by_agent(root)
    if not rows_by_agent:
        return []
    path = plot_training_convergence(
        rows_by_agent,
        out / "training_convergence.png",
    )
    return [path] if path is not None else []


def write_counterfactual_figures(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Write compact front-vehicle counterfactual summary figures."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / ".cache"))

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        return []

    paths = [
        plot_counterfactual_action_rates(
            rows,
            out / "counterfactual_action_rates.png",
        ),
        plot_counterfactual_q_advantage(
            rows,
            out / "counterfactual_lane_change_q_advantage.png",
        ),
    ]
    return [path for path in paths if path is not None]


def read_training_progress_by_agent(
    run_dir: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    root = Path(run_dir)
    logs_dir = root / "training_logs"
    if not logs_dir.exists():
        return {}
    rows_by_agent: dict[str, list[dict[str, Any]]] = {}
    for agent_dir in sorted(path for path in logs_dir.iterdir() if path.is_dir()):
        progress_path = agent_dir / "progress.csv"
        if not progress_path.exists():
            continue
        with progress_path.open(newline="", encoding="utf-8") as handle:
            rows_by_agent[agent_dir.name] = list(csv.DictReader(handle))
    return rows_by_agent


def plot_training_convergence(
    rows_by_agent: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    if not rows_by_agent:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    metric_specs = [
        ("rollout/ep_rew_mean", "mean episode reward"),
        ("rollout/ep_len_mean", "mean episode length"),
        ("train/loss", "training loss"),
    ]
    colors = {"FD": "#4C78A8", "BAL": "#54A24B", "SP": "#F58518"}
    for ax, (metric, ylabel) in zip(axes, metric_specs, strict=True):
        for agent, rows in rows_by_agent.items():
            points = [
                (get_float(row, "time/total_timesteps"), get_float(row, metric))
                for row in rows
                if get_str(row, "time/total_timesteps") != ""
                and get_str(row, metric) != ""
            ]
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            ax.plot(
                xs,
                ys,
                label=agent,
                linewidth=1.8,
                color=colors.get(agent),
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("training timesteps")
    axes[0].set_title("DQN Training Convergence")
    axes[0].legend(frameon=False, ncol=max(1, len(rows_by_agent)))
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_counterfactual_action_rates(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    if not rows:
        return None

    agents = sorted({get_str(row, "agent_condition") for row in rows})
    variants = ordered_unique(get_str(row, "counterfactual_variant") for row in rows)
    categories = ["lane_change", "FASTER", "IDLE", "SLOWER"]
    colors = {
        "lane_change": "#4C78A8",
        "FASTER": "#F58518",
        "IDLE": "#8C8C8C",
        "SLOWER": "#54A24B",
    }
    fig, axes = plt.subplots(
        1,
        len(agents),
        figsize=(3.6 * len(agents), 3.9),
        sharey=True,
    )
    if len(agents) == 1:
        axes = [axes]
    for ax, agent in zip(axes, agents, strict=True):
        bottoms = [0.0] * len(variants)
        agent_rows = [row for row in rows if get_str(row, "agent_condition") == agent]
        totals = Counter(get_str(row, "counterfactual_variant") for row in agent_rows)
        grouped = defaultdict(Counter)
        for row in agent_rows:
            variant = get_str(row, "counterfactual_variant")
            action = get_str(row, "action")
            category = (
                "lane_change" if action in {"LANE_LEFT", "LANE_RIGHT"} else action
            )
            grouped[variant][category] += 1
        for category in categories:
            values = [
                grouped[variant][category] / totals[variant]
                if totals[variant]
                else 0.0
                for variant in variants
            ]
            ax.bar(
                variants,
                values,
                bottom=bottoms,
                color=colors[category],
                label=category,
            )
            bottoms = [
                bottom + value for bottom, value in zip(bottoms, values, strict=True)
            ]
        ax.set_title(agent)
        ax.set_ylim(0, 1)
        ax.set_xticks(
            range(len(variants)),
            [counterfactual_variant_label(variant) for variant in variants],
        )
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("initial action rate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Initial Action Under Front-Vehicle Counterfactuals", y=1.02)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_counterfactual_q_advantage(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    if not rows:
        return None

    agents = sorted({get_str(row, "agent_condition") for row in rows})
    variants = ordered_unique(get_str(row, "counterfactual_variant") for row in rows)
    x_base = list(range(len(variants)))
    width = 0.22 if len(agents) > 1 else 0.55
    offsets = [
        (index - (len(agents) - 1) / 2) * width for index in range(len(agents))
    ]
    colors = {"FD": "#4C78A8", "BAL": "#54A24B", "SP": "#F58518"}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    for agent, offset in zip(agents, offsets, strict=True):
        means: list[float] = []
        for variant in variants:
            values = [
                get_float(row, "lane_change_q_advantage")
                for row in rows
                if get_str(row, "agent_condition") == agent
                and get_str(row, "counterfactual_variant") == variant
                and get_str(row, "lane_change_q_advantage") != ""
            ]
            means.append(sum(values) / len(values) if values else 0.0)
        xs = [x + offset for x in x_base]
        ax.bar(xs, means, width=width, color=colors.get(agent), label=agent)
    ax.set_xticks(
        x_base,
        [counterfactual_variant_label(variant) for variant in variants],
        rotation=0,
    )
    ax.set_ylabel("mean lane-change Q advantage")
    ax.set_title("Counterfactual Q Preference for Lane Change")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def counterfactual_variant_label(variant: str) -> str:
    labels = {
        "original": "original",
        "no-front": "no front",
        "matched-speed-front": "matched\nspeed",
        "far-front": "far front",
    }
    return labels.get(variant, variant.replace("-", "\n"))


def plot_paired_timing_gaps(
    gap_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in gap_rows:
        if get_str(row, "gap_status") != VALID_PAIR:
            continue
        label = f"{get_str(row, 'agent_b')} - {get_str(row, 'agent_a')}"
        grouped[label].append(get_float(row, "gap_b_minus_a", 0.0))

    if not grouped:
        return None

    labels = sorted(grouped)
    data = [grouped[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for idx, values in enumerate(data, start=1):
        xs = [idx + ((i % 7) - 3) * 0.018 for i, _ in enumerate(values)]
        ax.scatter(xs, values, s=28, color="#2F6B8F", alpha=0.82, zorder=3)
    ax.set_ylabel("gap_b_minus_a (timesteps)")
    ax.set_title("Paired Matched-Exposure Timing Gaps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_outcome_stacked_bar(
    outcome_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in outcome_rows:
        grouped[get_str(row, "agent_condition")][get_str(row, "episode_outcome")] += 1

    if not grouped:
        return None

    agents = sorted(grouped)
    categories = [VALID_ONSET, NO_ONSET_CENSORED, TERMINAL_FAILURE]
    colors = {
        VALID_ONSET: "#54A24B",
        NO_ONSET_CENSORED: "#B9A44C",
        TERMINAL_FAILURE: "#D9534F",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bottoms = [0] * len(agents)
    for category in categories:
        values = [grouped[agent][category] for agent in agents]
        ax.bar(agents, values, bottom=bottoms, color=colors[category], label=category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_ylabel("rollout count")
    ax.set_title("Episode Outcomes by Agent")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


