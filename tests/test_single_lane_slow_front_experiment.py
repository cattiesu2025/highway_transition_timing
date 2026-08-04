import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest


def load_experiment_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "single_lane_slow_front"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("single_lane_slow_front_run", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "single_lane_slow_front"
        / "plot_report_figures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "single_lane_slow_front_plot_report_figures",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = load_experiment_module()
ExperimentConfig = experiment.ExperimentConfig
make_training_spec = experiment.make_training_spec


def test_eval_specs_cover_full_factorial_grid_before_repeating():
    config = ExperimentConfig(seed=11)

    specs = experiment.make_eval_specs(37, config)
    first_cycle = specs[:36]
    combinations = {
        (spec.ego_speed, spec.front_distance, spec.front_speed)
        for spec in first_cycle
    }

    assert len(combinations) == 36
    assert {spec.ego_speed for spec in first_cycle} == {26.0, 28.0, 30.0}
    assert {spec.front_distance for spec in first_cycle} == {
        90.0,
        120.0,
        150.0,
        180.0,
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


def test_no_front_train_fraction_controls_training_spec_variant():
    always_front = make_training_spec(
        0,
        ExperimentConfig(
            training_scenario_profile="legacy-random",
            no_front_train_fraction=0.0,
            near_matched_speed_train_fraction=0.0,
        ),
    )
    assert always_front.include_front_vehicle is True
    assert always_front.scenario_type == "train_slow_front"

    never_front = make_training_spec(
        0,
        ExperimentConfig(
            training_scenario_profile="legacy-random",
            no_front_train_fraction=1.0,
            near_matched_speed_train_fraction=0.0,
        ),
    )
    assert never_front.include_front_vehicle is False
    assert never_front.scenario_type == "train_no_front"


def test_no_front_train_fraction_must_be_probability():
    with pytest.raises(ValueError, match="no_front_train_fraction"):
        ExperimentConfig(no_front_train_fraction=-0.1)

    with pytest.raises(ValueError, match="no_front_train_fraction"):
        ExperimentConfig(no_front_train_fraction=1.1)


def test_near_matched_training_specs_use_held_out_continuous_states():
    config = ExperimentConfig(
        seed=11,
        training_scenario_profile="legacy-random",
        no_front_train_fraction=0.0,
        near_matched_speed_train_fraction=1.0,
        near_matched_speed_delta=2.0,
    )

    specs = [make_training_spec(index, config) for index in range(128)]
    speed_deltas = [spec.front_speed - spec.ego_speed for spec in specs]

    assert all(spec.include_front_vehicle for spec in specs)
    assert all(spec.scenario_type == "train_near_matched_front" for spec in specs)
    assert all(
        experiment.NEAR_MATCHED_SPEED_MIN_ABS_DELTA - 0.001
        <= abs(delta)
        <= config.near_matched_speed_delta + 0.001
        for delta in speed_deltas
    )
    assert any(delta < 0.0 for delta in speed_deltas)
    assert any(delta > 0.0 for delta in speed_deltas)
    assert all(
        spec.exposure_seed >= experiment.TRAINING_SPEC_SEED_OFFSET
        for spec in specs
    )


def test_training_control_fractions_must_form_valid_mixture():
    with pytest.raises(ValueError, match="near_matched_speed_train_fraction"):
        ExperimentConfig(near_matched_speed_train_fraction=-0.1)

    with pytest.raises(ValueError, match="must be <= 1"):
        ExperimentConfig(
            no_front_train_fraction=0.6,
            near_matched_speed_train_fraction=0.5,
        )

    with pytest.raises(ValueError, match="near_matched_speed_delta"):
        ExperimentConfig(near_matched_speed_delta=0.1)


def test_default_time_and_target_speed_configuration_is_physically_explicit():
    config = ExperimentConfig(
        evaluation_duration=120,
        policy_frequency=5,
    )

    assert config.policy_step_seconds == pytest.approx(0.2)
    assert config.evaluation_max_policy_steps == 600
    assert config.target_speeds == (10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
    assert min(config.target_speeds) <= experiment.SLOW_FRONT_SPEED_RANGE[0]
    assert max(config.target_speeds) >= experiment.EGO_SPEED_RANGE[1]


def test_target_speed_grid_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="at least two"):
        ExperimentConfig(target_speeds=(10.0,))

    with pytest.raises(ValueError, match="strictly increasing"):
        ExperimentConfig(target_speeds=(10.0, 20.0, 20.0))


def test_reward_strength_multiplier_is_restricted_to_approved_levels():
    assert ExperimentConfig(reward_strength_multiplier=1.0).reward_strength_multiplier == 1.0
    assert ExperimentConfig(reward_strength_multiplier=2.0).reward_strength_multiplier == 2.0
    assert ExperimentConfig(reward_strength_multiplier=4.0).reward_strength_multiplier == 4.0

    with pytest.raises(ValueError, match="reward_strength_multiplier"):
        ExperimentConfig(reward_strength_multiplier=3.0)


def test_reward_strength_cli_and_effective_weights_follow_frozen_rules():
    args = experiment.build_parser().parse_args(
        ["--out", "validation", "--reward-strength-multiplier", "4"]
    )
    config = experiment.config_from_args(args)

    fd = experiment.effective_reward_weights("FD", config)
    bal = experiment.effective_reward_weights("BAL", config)
    sp = experiment.effective_reward_weights("SP", config)

    assert fd.speed_score == pytest.approx(0.45)
    assert fd.front_distance_score == pytest.approx(4.0)
    assert fd.collision_risk_penalty == pytest.approx(3.0)
    assert sp.speed_score == pytest.approx(4.0)
    assert sp.front_distance_score == pytest.approx(0.25)
    assert sp.collision_risk_penalty == pytest.approx(3.0)
    assert bal.speed_score == pytest.approx(2.8)
    assert bal.front_distance_score == pytest.approx(2.8)
    assert bal.collision_penalty == pytest.approx(8.0)
    assert bal.collision_risk_penalty == pytest.approx(12.0)


def test_stratified_training_block_has_exact_family_and_difficulty_counts():
    config = ExperimentConfig(seed=17)
    specs = [
        make_training_spec(index, config)
        for index in range(len(experiment.STRATIFIED_TRAINING_BLOCK))
    ]

    assert Counter(spec.scenario_type for spec in specs) == {
        "train_no_front": 4,
        "train_non_closing_front": 4,
        "train_near_closing_front": 2,
        "train_visible_slow_front": 8,
        "train_delayed_visible_front": 2,
    }
    assert Counter(
        (spec.ttc_bin, spec.required_deceleration_bin)
        for spec in specs
        if spec.scenario_type == "train_visible_slow_front"
    ) == {
        ("gradual", "gentle"): 2,
        ("easy", "gentle"): 2,
        ("medium", "moderate"): 2,
        ("hard", "strong"): 2,
    }


def test_stratified_training_controls_have_separate_closing_semantics():
    config = ExperimentConfig(seed=23)
    specs = [make_training_spec(index, config) for index in range(40)]
    no_front = [spec for spec in specs if spec.scenario_type == "train_no_front"]
    non_closing = [
        spec
        for spec in specs
        if spec.scenario_type == "train_non_closing_front"
    ]
    near_closing = [
        spec
        for spec in specs
        if spec.scenario_type == "train_near_closing_front"
    ]
    delayed = [
        spec
        for spec in specs
        if spec.scenario_type == "train_delayed_visible_front"
    ]

    assert all(not spec.include_front_vehicle for spec in no_front)
    assert all(spec.closing_speed < 0.0 for spec in non_closing)
    assert all(spec.ttc_bin == "non_closing" for spec in non_closing)
    assert all(
        experiment.NEAR_MATCHED_SPEED_MIN_ABS_DELTA - 0.001
        <= spec.closing_speed
        <= config.near_matched_speed_delta + 0.001
        for spec in near_closing
    )
    assert all(
        spec.front_distance > experiment.OBSERVATION_DISTANCE_METRES
        for spec in delayed
    )
    assert all(not spec.visible_at_t0 for spec in delayed)


def test_stratified_training_specs_are_reproducible_by_seed_and_reset_index():
    config = ExperimentConfig(seed=31)
    first = [make_training_spec(index, config) for index in range(60)]
    repeated = [make_training_spec(index, config) for index in range(60)]
    different_seed = [
        make_training_spec(index, ExperimentConfig(seed=32))
        for index in range(60)
    ]

    assert first == repeated
    assert first != different_seed


def test_visible_slow_front_specs_respect_declared_state_constraints():
    config = ExperimentConfig(seed=41)
    specs = [make_training_spec(index, config) for index in range(200)]
    visible = [
        spec
        for spec in specs
        if spec.scenario_type == "train_visible_slow_front"
    ]

    assert visible
    assert all(
        experiment.VISIBLE_TRAIN_DISTANCE_RANGE[0]
        <= spec.front_distance
        <= experiment.VISIBLE_TRAIN_DISTANCE_RANGE[1]
        for spec in visible
    )
    assert all(
        experiment.EGO_SPEED_RANGE[0]
        <= spec.ego_speed
        <= experiment.EGO_SPEED_RANGE[1]
        for spec in visible
    )
    assert all(
        experiment.SLOW_FRONT_SPEED_RANGE[0]
        <= spec.front_speed
        <= experiment.SLOW_FRONT_SPEED_RANGE[1]
        for spec in visible
    )
    assert all(spec.closing_speed > 0.0 for spec in visible)
    assert all(spec.visible_at_t0 for spec in visible)


def test_rollout_plot_summary_keeps_observed_and_missing_onsets_separate():
    plot_module = load_plot_module()

    summary = plot_module.summarize_rollout_outcome_rows(
        [
            {
                "agent_condition": "SP",
                "episode_outcome": "valid_onset",
                "response_latency": "8",
            },
            {
                "agent_condition": "SP",
                "episode_outcome": "no_onset_censored",
                "response_latency": "",
            },
            {
                "agent_condition": "SP",
                "episode_outcome": "valid_onset",
                "response_latency": "12",
            },
        ],
        "original",
    )

    assert summary[("SP", "original")] == {
        "n": 3,
        "observed": [8.0, 12.0],
        "n_observed": 2,
        "n_without_onset": 1,
        "median": 10.0,
    }


def test_report_plot_cli_accepts_run_directory():
    plot_module = load_plot_module()

    args = plot_module.build_parser().parse_args(["--run-dir", "trained-100k"])

    assert args.run_dir == "trained-100k"
