"""Tests for the isolated no-blocker multi-lane experiment."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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


def test_sealed_heldout_grid_is_fixed_and_marginally_disjoint_from_development():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=0, lanes_count=4, ego_lane=1)

    heldout = module.make_sealed_heldout_specs(config)
    heldout_other_training_seed = module.make_sealed_heldout_specs(
        module.ExperimentConfig(seed=4, lanes_count=4, ego_lane=1)
    )
    development = module.make_eval_specs(36, config)

    assert len(heldout) == 36
    assert module.file_sha256(module.SEALED_HELDOUT_GRID_PATH) == (
        module.SEALED_HELDOUT_GRID_SHA256
    )
    assert [spec.exposure_seed for spec in heldout] == [
        spec.exposure_seed for spec in heldout_other_training_seed
    ]
    assert {spec.exposure_id for spec in heldout} == {
        f"H{index:04d}" for index in range(36)
    }
    assert {spec.ego_speed for spec in heldout} == {25.0, 27.0, 29.0}
    assert {spec.front_distance for spec in heldout} == {
        130.0,
        170.0,
        210.0,
        250.0,
    }
    assert {spec.front_speed for spec in heldout} == {11.0, 15.0, 19.0}
    for field in ("ego_speed", "front_distance", "front_speed"):
        assert {
            getattr(spec, field) for spec in heldout
        }.isdisjoint({getattr(spec, field) for spec in development})


def test_sealed_heldout_grid_checksum_rejects_mutation(tmp_path):
    module = load_experiment_module()
    changed = tmp_path / "heldout.csv"
    changed.write_bytes(module.SEALED_HELDOUT_GRID_PATH.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        module.load_eval_specs(
            changed,
            module.ExperimentConfig(),
            expected_sha256=module.SEALED_HELDOUT_GRID_SHA256,
        )


def test_stratified_training_block_has_exact_family_quotas():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=3, lanes_count=4, ego_lane=1)

    specs = [module.make_training_spec(index, config) for index in range(20)]
    counts = Counter(spec.scenario_type for spec in specs)

    assert counts == {
        "train_no_front": 4,
        "train_exact_matched_speed_front": 2,
        "train_non_closing_front": 2,
        "train_near_closing_front": 2,
        "train_visible_slow_front": 8,
        "train_boundary_visible_front": 2,
    }
    assert {spec.ego_lane for spec in specs} == {1}


def test_visible_positive_resets_balance_the_four_frozen_difficulty_cells():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=7)

    specs = [module.make_training_spec(index, config) for index in range(20)]
    visible_positive = [
        spec
        for spec in specs
        if spec.scenario_type == "train_visible_slow_front"
    ]
    difficulty_counts = Counter(
        (spec.ttc_bin, spec.required_deceleration_bin)
        for spec in visible_positive
    )

    assert difficulty_counts == {
        ("gradual", "gentle"): 2,
        ("easy", "gentle"): 2,
        ("medium", "moderate"): 2,
        ("hard", "strong"): 2,
    }
    assert all(
        module.VISIBLE_TRAIN_DISTANCE_RANGE[0]
        <= spec.front_distance
        <= module.VISIBLE_TRAIN_DISTANCE_RANGE[1]
        for spec in visible_positive
    )
    assert all(
        module.SLOW_FRONT_SPEED_RANGE[0]
        <= spec.front_speed
        <= module.SLOW_FRONT_SPEED_RANGE[1]
        for spec in visible_positive
    )
    assert all(spec.ttc_bin != "critical" for spec in specs)


def test_training_controls_and_visibility_are_physically_labelled():
    module = load_experiment_module()
    specs = [
        module.make_training_spec(index, module.ExperimentConfig(seed=13))
        for index in range(20)
    ]

    no_front = [spec for spec in specs if spec.scenario_type == "train_no_front"]
    non_closing = [
        spec for spec in specs if spec.scenario_type == "train_non_closing_front"
    ]
    exact_matched = [
        spec
        for spec in specs
        if spec.scenario_type == "train_exact_matched_speed_front"
    ]
    near_closing = [
        spec for spec in specs if spec.scenario_type == "train_near_closing_front"
    ]
    boundary_visible = [
        spec for spec in specs if spec.scenario_type == "train_boundary_visible_front"
    ]

    assert all(not spec.include_front_vehicle for spec in no_front)
    assert all(spec.ttc_bin == "not_applicable" for spec in no_front)
    assert all(spec.closing_speed == 0.0 for spec in exact_matched)
    assert all(spec.ttc_bin == "non_closing" for spec in exact_matched)
    assert all(
        spec.required_deceleration_bin == "none" for spec in exact_matched
    )
    assert all(spec.closing_speed < 0.0 for spec in non_closing)
    assert all(spec.ttc_bin == "non_closing" for spec in non_closing)
    assert all(spec.required_deceleration_bin == "none" for spec in non_closing)
    assert all(spec.closing_speed > 0.0 for spec in near_closing)
    assert all(spec.ttc_bin == "gradual" for spec in near_closing)
    assert all(
        spec.visible_at_t0
        for spec in near_closing + non_closing + exact_matched + boundary_visible
    )
    assert all(spec.ttc_bin == "gradual" for spec in boundary_visible)
    assert all(
        spec.required_deceleration_bin == "gentle"
        for spec in boundary_visible
    )


def test_stratified_training_specs_are_deterministic_by_seed_and_reset():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=19)

    first = [module.make_training_spec(index, config) for index in range(40)]
    repeated = [module.make_training_spec(index, config) for index in range(40)]
    other_seed = [
        module.make_training_spec(index, module.ExperimentConfig(seed=20))
        for index in range(40)
    ]

    assert first == repeated
    assert first != other_seed
    assert Counter(spec.scenario_type for spec in first[:20]) == Counter(
        spec.scenario_type for spec in first[20:]
    )


def test_legacy_random_fraction_controls_front_vehicle_presence():
    module = load_experiment_module()
    always_front = module.ExperimentConfig(
        training_scenario_profile="legacy-random",
        no_front_train_fraction=0.0,
    )
    never_front = module.ExperimentConfig(
        training_scenario_profile="legacy-random",
        no_front_train_fraction=1.0,
    )

    assert all(
        module.make_training_spec(index, always_front).include_front_vehicle
        for index in range(12)
    )
    assert all(
        not module.make_training_spec(index, never_front).include_front_vehicle
        for index in range(12)
    )


def test_training_manifest_records_physics_and_block_position():
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=5)
    spec = module.make_training_spec(23, config)

    row = module.training_spec_row(23, spec, config)

    assert row["scene_seed"] == config.seed + module.TRAINING_SPEC_SEED_OFFSET + 23
    assert row["block_index"] == 1
    assert row["block_position"] == 3
    assert row["block_size"] == 20
    assert row["scenario_type"] == spec.scenario_type
    assert row["ttc_bin"] == spec.ttc_bin
    assert row["required_deceleration_bin"] == spec.required_deceleration_bin
    assert row["difficulty_bin"] == (
        f"{spec.ttc_bin}__{spec.required_deceleration_bin}"
    )
    assert row["adjacent_lanes_open"] is True
    assert row["background_vehicles"] == 0


def test_training_audit_writes_rows_and_exact_full_block_counts(tmp_path):
    module = load_experiment_module()
    config = module.ExperimentConfig(seed=5)

    rows, summary_rows = module.audit_training_scenarios(tmp_path, 40, config)

    family_rows = {
        row["scenario_type"]: row["n"]
        for row in summary_rows
        if row["record_type"] == "family"
    }
    assert len(rows) == 40
    assert family_rows == {
        "train_boundary_visible_front": 4,
        "train_exact_matched_speed_front": 4,
        "train_near_closing_front": 4,
        "train_no_front": 8,
        "train_non_closing_front": 4,
        "train_visible_slow_front": 16,
    }
    assert (tmp_path / "training_scenario_audit.csv").exists()
    assert (tmp_path / "training_scenario_audit_summary.csv").exists()


def test_stratified_training_audit_requires_complete_blocks(tmp_path):
    module = load_experiment_module()

    with pytest.raises(ValueError, match="multiple of the stratified block size"):
        module.audit_training_scenarios(
            tmp_path,
            21,
            module.ExperimentConfig(seed=5),
        )


def test_multilane_uses_the_same_frozen_positive_cells_as_single_lane():
    module = load_experiment_module()
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "single_lane_slow_front"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("single_lane_slow_front_run", path)
    assert spec is not None
    single_lane = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = single_lane
    spec.loader.exec_module(single_lane)

    assert module.VISIBLE_DIFFICULTY_CELLS == single_lane.VISIBLE_DIFFICULTY_CELLS
    multilane_positive_cells = Counter(
        (ttc_bin, deceleration_bin)
        for scenario_type, ttc_bin, deceleration_bin in module.STRATIFIED_TRAINING_BLOCK
        if scenario_type == "train_visible_slow_front"
    )
    single_lane_positive_cells = Counter(
        (ttc_bin, deceleration_bin)
        for scenario_type, ttc_bin, deceleration_bin in single_lane.STRATIFIED_TRAINING_BLOCK
        if scenario_type == "train_visible_slow_front"
    )
    assert multilane_positive_cells == single_lane_positive_cells


def test_no_front_training_fraction_is_validated():
    module = load_experiment_module()

    with pytest.raises(ValueError, match="no_front_train_fraction"):
        module.ExperimentConfig(no_front_train_fraction=-0.01)
    with pytest.raises(ValueError, match="no_front_train_fraction"):
        module.ExperimentConfig(no_front_train_fraction=1.01)
    with pytest.raises(ValueError, match="training_scenario_profile"):
        module.ExperimentConfig(training_scenario_profile="unknown")
    with pytest.raises(ValueError, match="near_matched_speed_delta"):
        module.ExperimentConfig(near_matched_speed_delta=0.2)
    with pytest.raises(ValueError, match="lane_change_penalty"):
        module.ExperimentConfig(lane_change_penalty=-0.01)

    assert module.ExperimentConfig().lane_change_penalty == 0.2


def test_evaluation_duration_is_converted_from_seconds_to_policy_steps():
    module = load_experiment_module()
    config = module.ExperimentConfig(
        duration=20,
        evaluation_duration=120,
        policy_frequency=5,
    )

    assert config.policy_step_seconds == pytest.approx(0.2)
    assert config.training_max_policy_steps == 100
    assert config.evaluation_max_policy_steps == 600
    assert module.policy_time_fields(4, 0, config) == {
        "t_seconds": 0.8,
        "post_step_t_seconds": 1.0,
        "exposure_t_seconds": 0.0,
        "policy_frequency_hz": 5,
        "policy_step_seconds": 0.2,
    }


def test_training_wrapper_enforces_exact_policy_step_horizon():
    module = load_experiment_module()

    class EndlessEnv:
        def step(self, action):
            return "obs", 0.0, False, False, {"action": action}

    wrapper = object.__new__(module.OpenLaneTrainingResetWrapper)
    wrapper.env = EndlessEnv()
    wrapper.config = module.ExperimentConfig(duration=20, policy_frequency=5)
    wrapper.episode_step_count = 0

    for _ in range(99):
        _obs, _reward, terminated, truncated, info = wrapper.step(1)
        assert terminated is False
        assert truncated is False
        assert "training_horizon_reached" not in info

    _obs, _reward, terminated, truncated, info = wrapper.step(1)

    assert wrapper.episode_step_count == 100
    assert terminated is False
    assert truncated is True
    assert info["training_horizon_reached"] is True


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
