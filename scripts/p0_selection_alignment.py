#!/usr/bin/env python3
"""Where do importance peaks land, relative to the transition onset?

This is the first measurement of the Stage-5 summary comparison. It asks one
question of the two HIGHLIGHTS importance variants: when they pick the states a
clip summary would show, how far are those states from the onset that ONSET
detects in the same episode?

The question has a structural answer to check against. An onset is a decision
boundary, the moment a policy stops preferring one action and starts preferring
another. At a decision boundary the best and second-best actions are close, so
`max_second` is near its minimum there by construction; `max_min` tracks how
much is at stake, which keeps growing after the onset as the hazard closes. So
both variants are expected to peak *after* the onset, never before. If the sign
of the offset is stably positive across arms and seeds, that is a mechanism, not
an incidental miss, and it is the strongest available form of the claim that
clip summaries do not target persistent transitions.

The measurement deliberately reports the offset distribution rather than a hit
rate alone. A hit rate on its own overstates the case: importance does carry
information about onsets, roughly five times chance in a first look, so the
defensible claim is that it is imprecise and biased late, not that it is blind.
The chance rate is computed here for exactly that comparison.

Nothing is trained and no summary is scored against any FD/BAL/SP claim, so
this does not touch the pre-registration of the true-null cells. The sealed
held-out grid is not opened.

Run from the repository root:

    PYTHONPATH=src python3 scripts/p0_selection_alignment.py --arms C
    PYTHONPATH=src python3 scripts/p0_selection_alignment.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aggregate_selected_20seed import (  # noqa: E402
    AGENTS,
    read_csv_rows,
    sign_test_p,
    write_csv,
)

from highway_transition_timing.summaries import (  # noqa: E402
    HIGHLIGHTS_BUDGET,
    HIGHLIGHTS_CONTEXT_LENGTH,
    HIGHLIGHTS_MINIMUM_GAP,
    IMPORTANCE_VARIANTS,
    context_window,
    select_highlights,
    state_importance,
)

#: Offsets are reported against several windows because no single width is
#: privileged; the upstream context length gives the middle one.
HIT_WINDOWS = (5, 10, 20)


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    run_prefix: str
    seeds: tuple[int, ...]
    analysis_target: str
    analysis_dir: str


ARMS = (
    Arm(
        "A",
        "single lane",
        "single_lane_slow_front_fixed_100k_seed",
        tuple(range(3100, 3120)),
        "slowdown_onset",
        "analysis",
    ),
    Arm(
        "B",
        "two lanes, open target lane",
        "multilane_twolane_open_fixed_100k_seed",
        tuple(range(4100, 4120)),
        "lane_change_onset",
        "analysis_onset_fixed",
    ),
    Arm(
        "C",
        "two lanes, occupied target lane",
        "multilane_twolane_occupied_fixed_100k_seed",
        tuple(range(4200, 4220)),
        "lane_change_onset",
        "analysis_onset_fixed",
    ),
)
ARMS_BY_KEY = {arm.key: arm for arm in ARMS}


def onset_by_episode(run_dir: Path, arm: Arm) -> dict[str, int]:
    """Onset step per episode, for episodes that produced a valid onset."""
    onsets: dict[str, int] = {}
    path = run_dir / arm.analysis_dir / "episode_outcomes.csv"
    for row in read_csv_rows(path):
        if row["analysis_target"] != arm.analysis_target:
            continue
        if row["episode_outcome"] != "valid_onset":
            continue
        onset = row["target_onset_t"]
        if onset not in ("", "None"):
            onsets[row["episode_id"]] = int(float(onset))
    return onsets


def steps_by_agent(run_dir: Path) -> tuple[dict[str, list[tuple[str, int, list[float]]]], dict[str, int]]:
    """Per-agent (episode_id, step, q_values) and the length of each episode."""
    per_agent: dict[str, list[tuple[str, int, list[float]]]] = defaultdict(list)
    lengths: dict[str, int] = defaultdict(int)
    with (run_dir / "evaluation" / "steps.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row["q_values_or_action_scores"]
            if not raw:
                continue
            try:
                q_values = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if len(q_values) < 2:
                continue
            episode = row["episode_id"]
            step = int(row["t"])
            per_agent[row["agent_condition"]].append((episode, step, q_values))
            lengths[episode] = max(lengths[episode], step + 1)
    return per_agent, lengths


def chance_rate(
    episodes: list[tuple[str, int, list[float]]],
    onsets: dict[str, int],
    window: int,
) -> float:
    """Share of all steps that fall inside `window` of their episode's onset.

    This is the rate a summary would achieve by sampling states uniformly, and
    it is the only fair reference for the observed hit rate.
    """
    total = 0
    near = 0
    for episode, step, _q in episodes:
        total += 1
        onset = onsets.get(episode)
        if onset is not None and abs(step - onset) <= window:
            near += 1
    return near / total if total else 0.0


def collect_arm(arm: Arm, outputs_root: Path) -> tuple[list[dict], list[dict]]:
    """Per-selection rows and per-(seed, agent, variant) summary rows."""
    selection_rows: list[dict] = []
    seed_rows: list[dict] = []

    for seed in arm.seeds:
        run_dir = outputs_root / f"{arm.run_prefix}{seed}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing run directory: {run_dir}")
        onsets = onset_by_episode(run_dir, arm)
        per_agent, lengths = steps_by_agent(run_dir)

        for agent in AGENTS:
            episodes = per_agent.get(agent, [])
            if not episodes:
                continue
            chance = {w: chance_rate(episodes, onsets, w) for w in HIT_WINDOWS}
            selections_by_variant: dict[str, list[tuple[str, int]]] = {}

            for variant in IMPORTANCE_VARIANTS:
                scored = [
                    ((episode, step), state_importance(q_values, variant))
                    for episode, step, q_values in episodes
                ]
                selected = select_highlights(
                    scored,
                    budget=HIGHLIGHTS_BUDGET,
                    context_length=HIGHLIGHTS_CONTEXT_LENGTH,
                    minimum_gap=HIGHLIGHTS_MINIMUM_GAP,
                )
                selections_by_variant[variant] = selected
                importance = dict(scored)

                offsets: list[int] = []
                hits = {w: 0 for w in HIT_WINDOWS}
                in_window = 0
                no_onset = 0
                for episode, step in selected:
                    onset = onsets.get(episode)
                    row = {
                        "arm": arm.key,
                        "seed": seed,
                        "agent_condition": agent,
                        "variant": variant,
                        "episode_id": episode,
                        "step": step,
                        "importance": round(importance[(episode, step)], 6),
                        "episode_length": lengths[episode],
                        "onset_t": "" if onset is None else onset,
                        "offset_steps": "" if onset is None else step - onset,
                    }
                    window = context_window(
                        step, lengths[episode], HIGHLIGHTS_CONTEXT_LENGTH
                    )
                    row["onset_in_context_window"] = (
                        onset is not None and onset in window
                    )
                    for w in HIT_WINDOWS:
                        hit = onset is not None and abs(step - onset) <= w
                        row[f"within_{w}"] = hit
                        hits[w] += int(hit)
                    if onset is None:
                        no_onset += 1
                    else:
                        offsets.append(step - onset)
                    in_window += int(row["onset_in_context_window"])
                    selection_rows.append(row)

                seed_row = {
                    "arm": arm.key,
                    "seed": seed,
                    "agent_condition": agent,
                    "variant": variant,
                    "n_selected": len(selected),
                    "n_without_onset": no_onset,
                    "n_onset_in_context_window": in_window,
                    "median_offset_steps": (
                        round(statistics.median(offsets), 2) if offsets else ""
                    ),
                    "min_offset_steps": min(offsets) if offsets else "",
                    "max_offset_steps": max(offsets) if offsets else "",
                    "n_offset_positive": sum(1 for d in offsets if d > 0),
                    "n_offset_negative": sum(1 for d in offsets if d < 0),
                }
                for w in HIT_WINDOWS:
                    seed_row[f"hits_{w}"] = hits[w]
                    seed_row[f"chance_{w}"] = round(chance[w], 6)
                seed_rows.append(seed_row)

            overlap = set(selections_by_variant[IMPORTANCE_VARIANTS[0]]) & set(
                selections_by_variant[IMPORTANCE_VARIANTS[1]]
            )
            for row in seed_rows[-len(IMPORTANCE_VARIANTS) :]:
                row["variant_overlap"] = len(overlap)

    return selection_rows, seed_rows


def summarise(arm: Arm, seed_rows: list[dict]) -> list[dict]:
    """Across-seed summary per agent and variant."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["agent_condition"], row["variant"])].append(row)

    summary: list[dict] = []
    for (agent, variant), rows in sorted(grouped.items()):
        medians = [r["median_offset_steps"] for r in rows if r["median_offset_steps"] != ""]
        positive = sum(1 for m in medians if m > 0)
        negative = sum(1 for m in medians if m < 0)
        selected = sum(r["n_selected"] for r in rows)
        record = {
            "arm": arm.key,
            "arm_label": arm.label,
            "agent_condition": agent,
            "variant": variant,
            "seeds": len(rows),
            "n_selected": selected,
            "n_without_onset": sum(r["n_without_onset"] for r in rows),
            "median_of_seed_medians": (
                round(statistics.median(medians), 2) if medians else ""
            ),
            "seeds_offset_positive": positive,
            "seeds_offset_negative": negative,
            "sign_test_p": round(sign_test_p(positive, negative), 6),
            "mean_variant_overlap": round(
                statistics.mean(r.get("variant_overlap", 0) for r in rows), 3
            ),
        }
        for w in HIT_WINDOWS:
            hits = sum(r[f"hits_{w}"] for r in rows)
            record[f"hit_rate_{w}"] = round(hits / selected, 4) if selected else ""
            record[f"chance_rate_{w}"] = round(
                statistics.mean(r[f"chance_{w}"] for r in rows), 6
            )
        summary.append(record)
    return summary


def print_report(summary: list[dict]) -> None:
    header = (
        f"{'arm':>3s} {'agent':5s} {'variant':11s} {'medΔ':>7s} "
        f"{'+/-seeds':>9s} {'p':>9s} {'hit±10':>7s} {'chance':>7s} {'lift':>5s} {'ovl':>4s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        hit = row["hit_rate_10"]
        chance = row["chance_rate_10"]
        lift = f"{hit / chance:.1f}x" if chance else "n/a"
        print(
            f"{row['arm']:>3s} {row['agent_condition']:5s} {row['variant']:11s} "
            f"{str(row['median_of_seed_medians']):>7s} "
            f"{row['seeds_offset_positive']:>4d}/{row['seeds_offset_negative']:<4d} "
            f"{row['sign_test_p']:>9.2e} {hit:>7.3f} {chance:>7.4f} {lift:>5s} "
            f"{row['mean_variant_overlap']:>4.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/summary_comparison/p0_selection_alignment"),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[arm.key for arm in ARMS],
        choices=[arm.key for arm in ARMS],
        help="arms to analyse; defaults to all three",
    )
    args = parser.parse_args()

    all_summary: list[dict] = []
    for key in args.arms:
        arm = ARMS_BY_KEY[key]
        selection_rows, seed_rows = collect_arm(arm, args.outputs_root)
        summary = summarise(arm, seed_rows)
        all_summary.extend(summary)

        write_csv(
            args.out / f"arm{arm.key}_selections.csv",
            list(selection_rows[0]),
            selection_rows,
        )
        write_csv(args.out / f"arm{arm.key}_by_seed.csv", list(seed_rows[0]), seed_rows)
        print(f"arm {arm.key} ({arm.label}): {len(selection_rows)} selected states")

    write_csv(args.out / "summary.csv", list(all_summary[0]), all_summary)
    print()
    print_report(all_summary)
    print()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
