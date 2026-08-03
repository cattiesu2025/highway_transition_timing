"""Tests for the isolated no-blocker multi-lane experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_experiment_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "multilane_open_lane_change"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("multilane_open_lane_change_run", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_specs_are_open_lane_slow_front_only():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=11, lanes_count=4, ego_lane=1)

    specs = module.make_eval_specs(6, config)

    assert len(specs) == 6
    assert {spec.scenario_type for spec in specs} == {"open_lane_slow_front"}
    assert {spec.ego_lane for spec in specs} == {1}
    assert all(spec.front_distance >= 140.0 for spec in specs)


def test_eval_specs_cover_full_factorial_grid_before_repeating():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=11, lanes_count=4, ego_lane=1)

    specs = module.make_eval_specs(37, config)
    first_cycle = specs[:36]
    combinations = {
        (spec.ego_speed, spec.front_distance, spec.front_speed)
        for spec in first_cycle
    }

    assert len(combinations) == 36
    assert {spec.ego_speed for spec in first_cycle} == {26.0, 28.0, 30.0}
    assert {spec.front_distance for spec in first_cycle} == {
        140.0,
        180.0,
        220.0,
        260.0,
    }
    assert {spec.front_speed for spec in first_cycle} == {10.0, 14.0, 18.0}
    assert (
        specs[36].ego_speed,
        specs[36].front_distance,
        specs[36].front_speed,
    ) == (
        specs[0].ego_speed,
        specs[0].front_distance,
        specs[0].front_speed,
    )


def test_training_spec_mixes_slow_front_and_no_front_scenes():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=3, lanes_count=4, ego_lane=1)

    specs = [module.make_training_spec(index, config) for index in range(100)]
    scenario_types = {spec.scenario_type for spec in specs}
    no_front_count = sum(not spec.include_front_vehicle for spec in specs)

    assert scenario_types == {"train_slow_front", "train_no_front"}
    assert 10 <= no_front_count <= 30
    assert all(spec.front_distance >= 120.0 for spec in specs)
    assert {spec.ego_lane for spec in specs} == {1}


def test_training_spec_fraction_controls_front_vehicle_presence():
    module = load_experiment_module()
    always_front = module.ExperimentConfig(no_front_train_fraction=0.0)
    never_front = module.ExperimentConfig(no_front_train_fraction=1.0)

    assert all(
        module.make_training_spec(index, always_front).include_front_vehicle
        for index in range(12)
    )
    assert all(
        not module.make_training_spec(index, never_front).include_front_vehicle
        for index in range(12)
    )


def test_no_front_training_fraction_is_validated():
    module = load_experiment_module()

    with pytest.raises(ValueError, match="no_front_train_fraction"):
        module.ExperimentConfig(no_front_train_fraction=-0.01)
    with pytest.raises(ValueError, match="no_front_train_fraction"):
        module.ExperimentConfig(no_front_train_fraction=1.01)


def test_evaluation_duration_is_converted_from_seconds_to_policy_steps():
    module = load_experiment_module()
    config = module.ExperimentConfig(
        evaluation_duration=120,
        policy_frequency=5,
    )

    assert config.policy_step_seconds == pytest.approx(0.2)
    assert config.evaluation_max_policy_steps == 600
    assert module.policy_time_fields(4, 0, config) == {
        "t_seconds": 0.8,
        "post_step_t_seconds": 1.0,
        "exposure_t_seconds": 0.0,
        "policy_frequency_hz": 5,
        "policy_step_seconds": 0.2,
    }


def test_actual_lane_change_summary_separates_action_and_actual_lane_change():
    module = load_experiment_module()
    steps = [
        {
            "episode_id": "FD_M0000_r0",
            "agent_condition": "FD",
            "exposure_id": "M0000",
            "t": 0,
            "ego_lane": 1,
            "post_ego_lane": 1,
            "ego_position": "[100.0,4.0]",
            "action": "LANE_RIGHT",
            "collision_flag": "False",
        },
        {
            "episode_id": "FD_M0000_r0",
            "agent_condition": "FD",
            "exposure_id": "M0000",
            "t": 1,
            "ego_lane": 1,
            "post_ego_lane": 2,
            "ego_position": "[105.0,4.8]",
            "action": "IDLE",
            "collision_flag": "False",
        },
    ]

    [summary] = module.summarize_actual_lane_changes(steps)

    assert summary["first_lane_change_action_t"] == 0
    assert summary["first_lane_change_action_seconds"] == 0.0
    assert summary["first_lateral_motion_t"] == 1
    assert summary["first_lateral_motion_seconds"] == 0.2
    assert summary["first_actual_lane_change_t"] == 1
    assert summary["first_actual_lane_change_seconds"] == 0.4
    assert summary["action_to_actual_delay"] == 1
    assert summary["first_action_direction_matches_actual"] is True
    assert summary["collision_flag"] is False
