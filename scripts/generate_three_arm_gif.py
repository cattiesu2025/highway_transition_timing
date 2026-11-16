#!/usr/bin/env python3
"""Render the highway-arm schematic, with optional legacy three-arm layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
PANEL_H = 244
HEADER_H = 66
FOOTER_H = 38
FPS = 15
SECONDS = 8.0
N_FRAMES = int(FPS * SECONDS)
BG = "#f7f8fa"
ROAD = "#343f47"
ROAD_EDGE = "#202a31"
LANE_MARK = "#aeb8bd"
TEXT = "#24313a"
MUTED = "#6b7881"
EGO = "#42a8dc"
LEAD = "#e8edf0"
TARGET = "#f2b75e"
EVENT = "#ef514b"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


F_TITLE = font(22, True)
F_SUB = font(14)
F_LABEL = font(13, True)
F_SMALL = font(12)
F_TINY = font(10)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def car(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, label: str | None = None) -> None:
    w, h = 42, 22
    draw.rounded_rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), 6, fill=color, outline="#1f2a30", width=2)
    draw.rounded_rectangle((x - 9, y - 7, x + 10, y + 7), 4, fill="#d9edf5" if color == EGO else "#c9d2d6")
    if label:
        box = draw.textbbox((0, 0), label, font=F_TINY)
        draw.text((x - (box[2] - box[0]) / 2, y + 15), label, font=F_TINY, fill="#d9e1e5")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    draw.line((*start, *end), fill=color, width=3)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    for delta in (-0.55, 0.55):
        p = (end[0] - 10 * math.cos(ang + delta), end[1] - 10 * math.sin(ang + delta))
        draw.line((*end, *p), fill=color, width=3)


def header(draw: ImageDraw.ImageDraw, y0: int, arm: str, title: str, subtitle: str, endpoint: str) -> None:
    draw.text((24, y0 + 10), f"ARM {arm}", font=F_LABEL, fill="#ffffff", stroke_width=5, stroke_fill="#68747c")
    draw.text((112, y0 + 8), title, font=F_TITLE, fill=TEXT)
    draw.text((112, y0 + 35), subtitle, font=F_SUB, fill=MUTED)
    tw = draw.textbbox((0, 0), endpoint, font=F_SMALL)[2]
    draw.rounded_rectangle((WIDTH - tw - 54, y0 + 14, WIDTH - 22, y0 + 46), 13, fill="#ffe4e2")
    draw.text((WIDTH - tw - 38, y0 + 23), endpoint, font=F_SMALL, fill="#b62d29")


def road(draw: ImageDraw.ImageDraw, top: int, lanes: int) -> list[float]:
    h = 112 if lanes == 2 else 72
    draw.rectangle((0, top, WIDTH, top + h), fill=ROAD, outline=ROAD_EDGE, width=2)
    lane_centres = []
    lane_h = h / lanes
    for lane in range(lanes):
        lane_centres.append(top + lane_h * (lane + 0.5))
        if lane:
            y = top + lane_h * lane
            for x in range(0, WIDTH, 42):
                draw.line((x, y, min(x + 22, WIDTH), y), fill=LANE_MARK, width=2)
    return lane_centres


def event_marker(draw: ImageDraw.ImageDraw, x: float, top: int, bottom: int, label: str) -> None:
    draw.line((x, top, x, bottom), fill=EVENT, width=3)
    draw.polygon(((x, top), (x - 6, top - 10), (x + 6, top - 10)), fill=EVENT)
    tw = draw.textbbox((0, 0), label, font=F_TINY)[2]
    draw.text((min(x + 8, WIDTH - tw - 8), top - 17), label, font=F_TINY, fill=EVENT)


def panel_a(draw: ImageDraw.ImageDraw, y0: int, t: float) -> None:
    header(draw, y0, "A", "Single lane — slowdown onset", "Slowing increases front spacing but sacrifices speed.", "ONSET = ego slowdown time")
    top = y0 + HEADER_H + 18
    yc = road(draw, top, 1)[0]
    lead_x = 920 + 14 * t
    before = min(t, 3.2)
    after = max(0.0, t - 3.2)
    ego_x = 250 + 28 * before + 15 * after
    car(draw, lead_x % (WIDTH + 100) - 50, yc, LEAD, "front 14 m/s")
    car(draw, ego_x, yc, EGO, "ego")
    if t >= 3.2:
        event_marker(draw, 250 + 28 * 3.2, top + 4, top + 68, "slowdown onset")
        arrow(draw, (ego_x - 58, yc), (ego_x - 92, yc), EVENT)
    # The visual canvas is schematic; use the reference's approximate 4 px/m scale.
    gap = max(0, (lead_x - ego_x) / 4)
    draw.text((WIDTH - 180, top + 77), f"gap {gap:0.0f} m", font=F_SMALL, fill=MUTED)


def lane_change_panel(draw: ImageDraw.ImageDraw, y0: int, t: float, occupied: bool, farther: bool = False) -> None:
    if farther:
        title = "Two lanes — yellow car farther ahead"
        subtitle = "Target-lane front spacing is greater than in the original lane."
        arm = "D"
    elif occupied:
        title = "Two lanes — occupied target lane"
        subtitle = "Lane change restores the speed–spacing trade-off."
        arm = "C"
    else:
        title = "Two lanes — empty target lane"
        subtitle = "Lane change improves speed and spacing together."
        arm = "B"
    header(draw, y0, arm, title, subtitle, "ONSET = ego lane-crossing time")
    top = y0 + HEADER_H + 8
    lane_y = road(draw, top, 2)
    upper, lower = lane_y
    lead_x = (750 if farther else 850) + 13 * t
    ego_x = 250 + 27 * t
    car(draw, lead_x, lower, LEAD, "slow front")
    if occupied:
        target_x = (930 if farther else 610) + 20 * t
        car(draw, target_x, upper, TARGET, "target-lane vehicle")
    change_start, change_end = (2.5, 4.0) if not occupied else (3.0, 4.8)
    progress = ease((t - change_start) / (change_end - change_start))
    ego_y = lower + (upper - lower) * progress
    car(draw, ego_x, ego_y, EGO, "ego")
    if change_start <= t < change_end:
        arrow(draw, (ego_x + 28, ego_y + 10), (ego_x + 70, ego_y - 30), "#70c8f0")
    crossing_time = (change_start + change_end) / 2
    if t >= crossing_time:
        event_marker(draw, 250 + 27 * crossing_time, top + 4, top + 108, "first lane crossing")


def make_frame(index: int, arms: int = 4) -> Image.Image:
    logical = Image.new("RGB", (WIDTH, PANEL_H * arms + FOOTER_H), BG)
    draw = ImageDraw.Draw(logical)
    t = index / FPS
    panel_a(draw, 0, t)
    lane_change_panel(draw, PANEL_H, t, occupied=False)
    lane_change_panel(draw, PANEL_H * 2, t, occupied=True)
    if arms == 4:
        lane_change_panel(draw, PANEL_H * 3, t, occupied=True, farther=True)
    footer = PANEL_H * arms
    draw.line((24, footer + 3, WIDTH - 24, footer + 3), fill="#d4d9dc", width=1)
    draw.text((24, footer + 13), f"t = {t:0.1f} s", font=F_SMALL, fill=TEXT)
    note = "Shared schematic timeline · illustrative trajectories, not measured response times"
    width = draw.textbbox((0, 0), note, font=F_SMALL)[2]
    draw.text((WIDTH - width - 24, footer + 13), note, font=F_SMALL, fill=MUTED)
    return logical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', type=int, choices=(3, 4), default=4)
    args = parser.parse_args()
    prefix = 'four' if args.arms == 4 else 'three'
    output = Path(f"outputs/{prefix}_arm_transition.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = [make_frame(i, args.arms) for i in range(N_FRAMES)]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )
    preview = Path(f"outputs/{prefix}_arm_transition_preview.png")
    frames[int(4.9 * FPS)].save(preview)
    print(output)
    print(preview)


if __name__ == "__main__":
    main()
