from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
        / "experiments"
        / "multilane_open_lane_change"
        / "plot_report_figures.py"
    )
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
