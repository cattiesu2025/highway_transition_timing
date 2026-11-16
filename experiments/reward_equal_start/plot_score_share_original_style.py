#!/usr/bin/env python3
"""The original 'speed still contributes less than half' bar chart, with Arm D's two bars added.

Same layout, colours and text as the original A/B/C figure; values come from
outputs/reward_equal_start/score_share/source_data/score_first100_by_seed.csv
(score_share_first100.py). Arm D has 5 seeds, stated in the subtitle.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/reward_equal_start/score_share"
rows = list(csv.DictReader((OUT / "source_data/score_first100_by_seed.csv").open()))
arms = ["A", "B", "C", "D"]
names = {"A": "Single lane", "B": "Empty target lane", "C": "Occupied target lane", "D": "Yellow car farther ahead"}
S = {a: mean(float(r["S"]) for r in rows if r["arm"] == a) for a in arms}
D = {a: mean(float(r["D"]) for r in rows if r["arm"] == a) for a in arms}
n = {a: sum(r["arm"] == a for r in rows) for a in arms}
assert [n[a] for a in arms] == [20, 20, 20, 5]

ORANGE, GREEN, GREY, MUTED = "#EDA100", "#199E70", "#555555", "#5F5F5F"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(14.0, 7.6), dpi=100)
fig.subplots_adjust(left=0.085, right=0.985, top=0.74, bottom=0.17)
x = range(len(arms))
width, gap = 0.34, 0.02
bars_s = ax.bar([i - width / 2 - gap / 2 for i in x], [S[a] for a in arms], width, color=ORANGE,
                edgecolor="white", linewidth=1.5, label="speed_score  S", zorder=3)
bars_d = ax.bar([i + width / 2 + gap / 2 for i in x], [D[a] for a in arms], width, color=GREEN,
                edgecolor="white", linewidth=1.5, label="front_distance_score  D", zorder=3)
for bar in [*bars_s, *bars_d]:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2, f"{bar.get_height():.0f}",
            ha="center", va="bottom", fontsize=16, color="#222222")
ax.set_xticks(list(x))
ax.set_xticklabels([f"Arm {a}\n{names[a]}\nspeed share {S[a] / (S[a] + D[a]):.0%}" for a in arms], fontsize=15)
ax.set_ylim(0, 100 * 1.03)
ax.set_yticks(range(0, 101, 20))
ax.tick_params(axis="y", labelsize=17, colors=GREY, length=0)
ax.tick_params(axis="x", length=0, pad=8)
ax.set_ylabel("Unweighted score over 100 steps (max 100)", fontsize=15, color=GREY)
ax.grid(axis="y", color="#E8E8E8", linewidth=1, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#BBBBBB")
ax.set_xlim(-0.55, 3.55)
ax.text(0.0, 1.06, "Development grid, first 100 steps, mean over 20 seeds (Arm D: 5 seeds)",
        transform=ax.transAxes, fontsize=14, color=MUTED, va="bottom")
fig.text(0.015, 0.955, "Equal weights (0.70, 0.70): speed still contributes less than half",
         fontsize=19.5, fontweight="bold", va="center")
fig.legend(handles=[bars_s, bars_d], loc="upper right", bbox_to_anchor=(0.985, 0.905), ncol=2,
           frameon=False, fontsize=15, handlelength=2.6, columnspacing=3)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"score_share_first100_4arm_original_style.{ext}", dpi=300 if ext == "png" else None,
                facecolor="white")
print({a: (round(S[a], 1), round(D[a], 1), n[a]) for a in arms})
