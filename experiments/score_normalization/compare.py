#!/usr/bin/env python3
"""Compare SN1 runs with their paired maintained-score runs (same arm, agent and seed).

Pairs: Arms A-C ``reward_equal20_arm<X>_{BAL,FL-SP,FL-FD}_100k_seed<s>``; Arm D
``arm_c_distance-affine-v1_R20-{BAL,FL-SP,FL-FD}_100k_seed<s>``. Endpoints and checks are
those of ``reward_equal_start/collect_20seed.py``: Arm A first action-defined slowdown onset,
Arms B-D first physical lane change, 36 development-grid episodes per selected model.

Writes outputs/score_normalization/comparison/{run_status, episode_onsets, run_summary,
paired_episodes}.csv. Runs without an eligible checkpoint are listed with the failed gates of
every checkpoint they evaluated and contribute no onsets.
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Both directories have a run.py, so load each module by path under a distinct name.
sn = load("score_norm_run", HERE / "run.py")
sys.path.insert(0, str(ROOT / "experiments/reward_equal_start"))
c20 = load("score_norm_collect_20seed", ROOT / "experiments/reward_equal_start/collect_20seed.py")

MAINTAINED_AGENT = {"BAL": "BAL", "SP": "FL-SP", "FD": "FL-FD"}


def maintained_job(job: dict, root: Path) -> tuple[dict, Path]:
    label = MAINTAINED_AGENT[job["variant"]]  # Directory label; the agent inside is SP/FD/BAL.
    if job["arm"] == "D":
        name = f"arm_c_distance-affine-v1_R20-{label}_100k_seed{job['seed']}"
    else:
        name = f"reward_equal20_arm{job['arm']}_{label}_100k_seed{job['seed']}"
    return dict(arm=job["arm"], variant=job["variant"], agent=job["variant"], seed=job["seed"]), root / name


def failed_gates(directory: Path, agent: str) -> str:
    counts = collections.Counter()
    checkpoints = [r for r in c20.rows(directory / "model_selection.csv") if r["agent_condition"] == agent]
    for r in checkpoints:
        counts.update(k[5:] for k, v in r.items() if k.startswith("gate_") and v.lower() == "false")
    return f"{len(checkpoints)} checkpoints; " + ", ".join(f"{k} failed {n}" for k, n in counts.most_common())


def summarise(episodes: list[dict]) -> dict:
    valid = [float(e["onset_seconds"]) for e in episodes if e["episode_outcome"] == "valid_onset"]
    return dict(n_total=len(episodes), n_onset=len(valid),
                n_censored=sum(e["episode_outcome"] == "no_onset_censored" for e in episodes),
                n_failure=sum(e["episode_outcome"] == "terminal_failure" for e in episodes),
                n_collision=sum(e["collision"] for e in episodes),
                median_onset_seconds=round(median(valid), 3) if valid else "")


def write(path: Path, table: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/score_normalization/comparison")
    args = parser.parse_args(argv)
    status, episodes, summary, paired = [], [], [], []
    for job in sn.jobs():
        sides = {"SN1": ({k: job[k] for k in ("arm", "variant", "agent", "seed")}, sn.run_directory(args.root, job)),
                 "maintained": maintained_job(job, args.root)}
        by_side = {}
        for score, (spec, directory) in sides.items():
            ok, detail = c20.selected(spec, directory)
            status.append(dict(score=score, **spec, run=directory.name, selection=("complete" if ok else "failed"),
                               detail=detail if ok else failed_gates(directory, spec["agent"])))
            if not ok:
                continue
            eps = [dict(score=score, **e) for e in c20.onsets(spec, directory)]
            episodes += eps
            by_side[score] = {e["exposure_id"]: e for e in eps}
            summary.append(dict(score=score, **spec, **summarise(eps)))
        if len(by_side) == 2:
            for exposure, new in by_side["SN1"].items():
                old = by_side["maintained"][exposure]
                both = new["onset_seconds"] != "" and old["onset_seconds"] != ""
                paired.append(dict(arm=job["arm"], variant=job["variant"], seed=job["seed"], exposure_id=exposure,
                                   maintained_outcome=old["episode_outcome"], sn1_outcome=new["episode_outcome"],
                                   maintained_onset=old["onset_seconds"], sn1_onset=new["onset_seconds"],
                                   shift_seconds=round(float(new["onset_seconds"]) - float(old["onset_seconds"]), 3)
                                   if both else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("run_status.csv", status), ("episode_onsets.csv", episodes),
                        ("run_summary.csv", summary), ("paired_episodes.csv", paired)):
        write(args.out / name, table)
    failed = [s["run"] for s in status if s["selection"] != "complete"]
    print(f"{len(status)} runs, {len(summary)} with onsets, {len(paired)} paired episodes; failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
