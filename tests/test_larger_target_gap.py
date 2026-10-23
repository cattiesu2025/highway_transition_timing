"""Checks of linked geometry, pairing, interventions and fixed-lane diagnostics."""
import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('larger_gap_pilot', ROOT / 'experiments/larger_target_gap/run.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_training_support_and_paired_schedule():
    _, m = pilot.load_modules()
    config = m.ExperimentConfig(seed=5200, target_lane_vehicle=True)
    scenes = [pilot.training_spec(m, i, config) for i in range(100)]
    assert scenes == [pilot.training_spec(m, i, config) for i in range(100)]
    for start in range(0, 100, 10):
        assert Counter(s.scenario_type for s in scenes[start:start + 10]) == {
            'larger_gap_slow': 6, 'larger_gap_no_front': 2, 'larger_gap_matched': 2}
    for s in scenes:
        assert 40 <= s.front_distance <= 80
        assert 30 - 1e-6 <= s.target_lane_gap - s.front_distance <= 80 + 1e-6
        assert 70 <= s.target_lane_gap <= 160
        if s.scenario_type == 'larger_gap_no_front':
            assert not s.include_front_vehicle and s.target_lane_gap is not None
    assert scenes != [pilot.training_spec(m, i, m.ExperimentConfig(seed=5201)) for i in range(100)]


def test_grid_and_original_defaults_are_independent():
    _, m = pilot.load_modules()
    config = m.ExperimentConfig(target_lane_vehicle=True)
    original = m.make_eval_specs(36, config)
    pilot.install(m)
    grid = pilot.eval_specs(m, config)
    assert len({(s.front_distance, s.front_speed, s.target_lane_gap) for s in grid}) == 27
    assert min(s.target_lane_gap for s in grid) == 80
    assert max(s.target_lane_gap for s in grid) == 150
    assert m.make_eval_specs(36, config) == original
    _, fresh = pilot.load_modules()
    assert fresh.make_training_spec(0, config).scenario_type != 'larger_gap_slow'
    assert fresh.TARGET_LANE_TRAIN_GAP_RANGE == (10., 90.)


def test_real_scene_observation_and_fixed_identity_after_change():
    _, m = pilot.load_modules()
    m.configure_headless_runtime()
    pilot.install(m)
    config = m.ExperimentConfig(target_lane_vehicle=True, duration=20, evaluation_duration=20)
    s = pilot.eval_specs(m, config)[0]
    env = m.make_env('BAL', config, training=False)
    try:
        env.reset(seed=s.exposure_seed)
        obs = m.apply_open_lane_scene(env, s)
        before = m.extract_diagnostics(env)
        assert obs[:, 0].sum() == 3
        assert before['lg_original_net_gap'] == 40
        assert before['lg_target_net_gap'] == 75
        assert before['lg_extra_gap'] == 35
        original = env.unwrapped.larger_gap_vehicles['original']
        target = env.unwrapped.larger_gap_vehicles['target']
        for i in range(15):
            env.step(0 if i == 0 else 1)
        after = m.extract_diagnostics(env)
        assert after['ego_lane'] == 0
        assert after['lg_original_x'] == original.position[0]
        assert after['lg_target_x'] == target.position[0]
        assert after['lg_extra_gap'] > 35
        assert not after['collision_flag']
        obs = m.apply_open_lane_scene(env, s, include_front_vehicle=False)
        removed = m.extract_diagnostics(env)
        assert obs[:, 0].sum() == 2
        assert not removed['lg_original_present']
        assert removed['lg_target_center_distance'] == s.target_lane_gap
        assert removed['lg_target_more_space'] == ''
    finally:
        env.close()
