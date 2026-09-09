from __future__ import annotations

import importlib.util
import csv
import sys
from dataclasses import replace
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "aggregate_heldout_v2.py"
    spec = importlib.util.spec_from_file_location("aggregate_heldout_v2", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_directional_decision_requires_all_twenty_seeds_and_ci_direction():
    module = load_module()
    positive = module.summarize_across_seeds("positive", [1.0] * 20)
    negative = module.summarize_across_seeds("negative", [-1.0] * 20)
    incomplete = module.summarize_across_seeds("incomplete", [1.0] * 19)

    assert module.decision(positive, ">0") == "supported"
    assert module.decision(negative, "<0") == "supported"
    assert module.decision(positive, "<0") == "not supported"
    assert module.decision(incomplete, ">0") == "non-estimable"


def test_protocol_grid_checksums_and_exposure_counts_match_files():
    module = load_module()
    repo_root = Path(__file__).parents[1]

    for arm in module.ARMS:
        assert len(module.expected_exposures(repo_root, arm)) == 36


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_writes_complete_tables_from_valid_sealed_outputs(tmp_path):
    module = load_module()
    repo_root = Path(__file__).parents[1]
    module.ARMS = tuple(replace(arm, seeds=(arm.seeds[0],)) for arm in module.ARMS)

    for arm in module.ARMS:
        seed = arm.seeds[0]
        run_dir = tmp_path / f"{arm.run_prefix}{seed}"
        for agent in module.AGENTS:
            model = run_dir / "models" / f"{agent}_main.zip"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        (run_dir / "training_runs.csv").write_text("seed\n1\n", encoding="utf-8")
        grid_rows = module.read_csv(
            repo_root / "experiments" / "heldout_v2" / arm.grid_file
        )
        ids = [row["exposure_id"] for row in grid_rows]
        heldout = run_dir / "rollout_counterfactual_heldout_v2"
        write_rows(
            heldout / "evaluation_grid_manifest.csv",
            [
                {
                    "evaluation_grid": "heldout-v2",
                    "grid_sha256": arm.grid_sha256,
                    "n_exposures": 36,
                }
            ],
        )
        for variant in arm.variants:
            rows = []
            for agent_index, agent in enumerate(module.AGENTS):
                for exposure_id in ids:
                    common = {
                        "agent_condition": agent,
                        "exposure_id": exposure_id,
                        "collision_flag": "False",
                    }
                    if arm.kind == "single":
                        rows.append(
                            {
                                **common,
                                "analysis_target": "slowdown_onset",
                                "episode_outcome": "valid_onset",
                                "response_latency_seconds": agent_index + 1,
                            }
                        )
                    else:
                        rows.append(
                            {
                                **common,
                                "actual_lane_change_observed": "True",
                                "first_actual_lane_change_seconds": agent_index + 1,
                            }
                        )
            filename = (
                "episode_outcomes.csv"
                if arm.kind == "single"
                else "actual_lane_change_summary.csv"
            )
            write_rows(heldout / variant / "analysis" / filename, rows)

    out = tmp_path / "aggregate"
    module.aggregate(repo_root, tmp_path, out)

    assert len(module.read_csv(out / "seed_level_gaps.csv")) == 9
    assert len(module.read_csv(out / "confirmatory_decisions.csv")) == 4
    assert "Primary reversal" in (out / "synchronisation_summary.md").read_text()
