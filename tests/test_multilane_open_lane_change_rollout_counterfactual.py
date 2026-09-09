from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_module():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "multilane_open_lane_change"
        / "rollout_counterfactual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "multilane_rollout_counterfactual",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    path = (
        Path(__file__).parents[1]
        / "figures"
        / "multilane"
        / "plot_report_figures.py"
    )
    if not path.exists():
        pytest.skip("Local Git-ignored multi-lane plotting script is unavailable")
    spec = importlib.util.spec_from_file_location(
        "multilane_plot_report_figures",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_variant_rows_summarize_removed_front_effect():
    module = load_module()
    rows = [
        {
            "agent_condition": "SP",
            "exposure_id": "M0000",
            "counterfactual_variant": "original",
            "first_actual_lane_change_t": 10,
        },
        {
            "agent_condition": "SP",
            "exposure_id": "M0000",
            "counterfactual_variant": "no-front",
            "first_actual_lane_change_t": 16,
        },
        {
            "agent_condition": "SP",
            "exposure_id": "M0001",
            "counterfactual_variant": "original",
            "first_actual_lane_change_t": 12,
        },
        {
            "agent_condition": "SP",
            "exposure_id": "M0001",
            "counterfactual_variant": "no-front",
            "first_actual_lane_change_t": "",
        },
    ]

    details, summary = module.paired_variant_rows(rows)

    assert len(details) == 2
    assert summary == [
        {
            "agent_condition": "SP",
            "baseline_variant": "original",
            "counterfactual_variant": "no-front",
            "n_pairs": 2,
            "baseline_actual_lane_change_count": 2,
            "counterfactual_actual_lane_change_count": 1,
            "both_actual_lane_change_count": 1,
            "baseline_only_count": 1,
            "counterfactual_only_count": 0,
            "neither_count": 0,
            "median_latency_delta_counterfactual_minus_baseline": 6,
            "mean_latency_delta_counterfactual_minus_baseline": 6,
        }
    ]


def test_counterfactual_cli_defaults_to_full_36_exposure_grid():
    module = load_module()
    args = module.build_parser().parse_args(["--run-dir", "trained-run"])

    assert args.num_exposures == 36
    assert args.eval_grid == "development"


def test_counterfactual_cli_accepts_sealed_heldout_grid():
    module = load_module()
    args = module.build_parser().parse_args(
        ["--run-dir", "trained-run", "--eval-grid", "heldout"]
    )

    assert args.eval_grid == "heldout"


def test_counterfactual_cli_names_heldout_v2_explicitly():
    module = load_module()
    args = module.build_parser().parse_args(
        ["--run-dir", "trained-run", "--eval-grid", "heldout-v2"]
    )

    assert args.eval_grid == "heldout-v2"


def test_report_summary_keeps_observed_and_censored_lane_changes_separate():
    plot_module = load_plot_module()

    summary = plot_module.summarize_rollout_counterfactual_rows(
        [
            {
                "agent_condition": "SP",
                "counterfactual_variant": "original",
                "first_actual_lane_change_t": "10",
            },
            {
                "agent_condition": "SP",
                "counterfactual_variant": "original",
                "first_actual_lane_change_t": "",
            },
            {
                "agent_condition": "SP",
                "counterfactual_variant": "original",
                "first_actual_lane_change_t": "14",
            },
        ]
    )

    assert summary[("SP", "original")] == {
        "n": 3,
        "observed": [10.0, 14.0],
        "n_observed": 2,
        "n_censored": 1,
        "median": 12.0,
        "time_unit": "policy_steps",
        "horizon": 120.0,
    }

    x_values, y_values = plot_module.cumulative_incidence_points(
        [10.0, 14.0],
        n_episodes=4,
        horizon=120.0,
    )
    assert x_values == [0.0, 10.0, 14.0, 120.0]
    assert y_values == [0.0, 0.25, 0.5, 0.5]


def test_report_summary_prefers_physical_seconds_and_recorded_horizon():
    plot_module = load_plot_module()

    summary = plot_module.summarize_rollout_counterfactual_rows(
        [
            {
                "agent_condition": "BAL",
                "counterfactual_variant": "original",
                "first_actual_lane_change_t": "8",
                "first_actual_lane_change_seconds": "1.8",
                "recording_horizon_seconds": "120",
            },
            {
                "agent_condition": "BAL",
                "counterfactual_variant": "original",
                "first_actual_lane_change_t": "",
                "first_actual_lane_change_seconds": "",
                "recording_horizon_seconds": "120",
            },
        ]
    )

    assert summary[("BAL", "original")] == {
        "n": 2,
        "observed": [1.8],
        "n_observed": 1,
        "n_censored": 1,
        "median": 1.8,
        "time_unit": "seconds",
        "horizon": 120.0,
    }


def test_initial_action_summary_rejects_stale_exposure_count(tmp_path):
    plot_module = load_plot_module()
    path = tmp_path / "summary.csv"
    path.write_text(
        "agent_condition,counterfactual_variant,n\n"
        "FD,original,24\n"
        "BAL,original,24\n"
        "SP,original,24\n"
    )

    assert not plot_module.initial_action_summary_matches_exposure_count(path, 36)


def test_open_target_lane_variant_removes_only_the_merged_vehicle():
    """Arms B and C differ by one training variable, so the between-arm
    occupancy contrast also carries the difference between two training runs.
    Removing the merged vehicle from an arm C scene makes the same comparison
    inside one policy, which is what the report needs to attribute the effect
    to occupancy rather than to the retraining."""

    module = load_module()

    class Spec:
        front_distance = 165.0
        front_speed = 14.0
        ego_speed = 28.0
        target_lane_gap = 40.0

    original = module.counterfactual_settings(Spec(), "original")
    opened = module.counterfactual_settings(Spec(), "open-target-lane")

    assert original["target_lane_gap"] == 40.0
    assert opened["target_lane_gap"] is None
    for field in ("include_front_vehicle", "front_distance", "front_speed"):
        assert opened[field] == original[field]


def test_every_variant_declares_the_target_lane_axis():
    module = load_module()

    class Spec:
        front_distance = 165.0
        front_speed = 14.0
        ego_speed = 28.0
        target_lane_gap = 15.0

    for variant in module.VARIANTS:
        settings = module.counterfactual_settings(Spec(), variant)
        assert "target_lane_gap" in settings, variant


def test_config_restores_the_scenario_geometry_from_the_training_run(tmp_path):
    """make_eval_specs picks the evaluation grid from the scenario geometry, so
    losing target_lane_vehicle would evaluate an occupied-lane policy on the
    open-lane grid."""

    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_runs.csv").write_text(
        "seed,policy_frequency_hz,lanes_count,ego_lane,target_lane_vehicle,"
        "target_lane_speed\n4200,5,2,1,True,29.0\n",
        encoding="utf-8",
    )

    experiment = module.load_experiment_module()
    config = module.config_from_training_run(experiment, run_dir, 120)

    assert config.target_lane_vehicle is True
    assert config.lanes_count == 2
    assert config.ego_lane == 1
    assert config.target_lane_speed == 29.0

    specs = experiment.make_eval_specs(36, config)
    assert sorted({spec.target_lane_gap for spec in specs}) == list(
        experiment.TARGET_LANE_MERGE_GAPS
    )


def test_opening_action_summary_reports_the_modal_action_per_variant():
    module = load_module()
    rows = [
        {
            "agent_condition": "FD",
            "counterfactual_variant": "original",
            "exposure_id": f"M{index:04d}",
            "opening_action": "IDLE" if index else "LANE_LEFT",
        }
        for index in range(4)
    ]

    summary = module.opening_action_summary(rows)

    assert len(summary) == 1
    assert summary[0]["modal_opening_action"] == "IDLE"
    assert summary[0]["modal_share"] == 0.75
    assert summary[0]["distinct_opening_actions"] == 2
