import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import gymnasium as gym
import pytest


spec = importlib.util.spec_from_file_location(
    'overtake_pilot_test', Path(__file__).parents[1] / 'experiments/overtake_success/pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def tracker_fixture():
    ego = SimpleNamespace(position=np.array([0., 4.]), LENGTH=5., WIDTH=2.,
                          heading=0., lane_index=('0', '1', 1), crashed=False, on_road=True)
    target = SimpleNamespace(position=np.array([20., 4.]), LENGTH=5., WIDTH=2., heading=0.)
    road = SimpleNamespace(vehicles=[ego, target], step=lambda dt: None,
                           network=SimpleNamespace(get_lane=lambda idx: SimpleNamespace(
                               local_coordinates=lambda position: (position[0], position[1] - 4*idx[2]))))
    env = SimpleNamespace(vehicle=ego, road=road)
    env.unwrapped = env
    return env, ego, target


def qualify(ego, env):
    ego.position = np.array([30., 0.])
    ego.lane_index = ('0', '1', 0)
    env.road.step(1/15)
    ego.position[1] = 4.
    ego.lane_index = ('0', '1', 1)


def test_success_needs_body_clearance_center_and_elapsed_window():
    env, ego, target = tracker_fixture()
    tracker = pilot.SuccessTracker(env)
    qualify(ego, env)
    ego.position[0] = 29.9
    for _ in range(20):
        env.road.step(1/15)
    assert not tracker.finish_policy_step()
    ego.position[0] = 30.
    ego.position[1] = 4.51
    for _ in range(20):
        env.road.step(1/15)
    assert not tracker.finish_policy_step()
    ego.position[1] = 4.5
    for _ in range(15):
        env.road.step(1/15)
    assert not tracker.finish_policy_step()  # 15 observations span only 14 intervals
    env.road.step(1/15)
    assert tracker.finish_policy_step()
    assert not tracker.finish_policy_step()
    assert tracker.target is target


def test_collision_same_policy_step_overrides_success():
    env, ego, _ = tracker_fixture()
    tracker = pilot.SuccessTracker(env)
    qualify(ego, env)
    for _ in range(16):
        env.road.step(1/15)
    ego.crashed = True
    env.road.step(1/15)
    assert not tracker.finish_policy_step()
    assert tracker.success_time is None


def test_interruption_resets_window_and_no_target_is_not_success():
    env, ego, _ = tracker_fixture()
    tracker = pilot.SuccessTracker(env)
    qualify(ego, env)
    for _ in range(10):
        env.road.step(1/15)
    ego.position[1] = 1.
    env.road.step(1/15)
    ego.position[1] = 4.
    for _ in range(10):
        env.road.step(1/15)
    assert not tracker.finish_policy_step()
    env, ego, _ = tracker_fixture()
    tracker = pilot.SuccessTracker(env, False)
    qualify(ego, env)
    for _ in range(30):
        env.road.step(1/15)
    assert not tracker.finish_policy_step()
    assert not tracker.summary()['success_applicable']


def test_replay_audit_counts_real_draws_without_changing_samples():
    buffer = pilot.AuditedReplayBuffer(10, gym.spaces.Box(-1, 1, (2,)),
                                      gym.spaces.Discrete(5), device='cpu')
    for event in [False, True]:
        buffer.add(np.zeros((1, 2)), np.ones((1, 2)), np.array([1]),
                   np.array([30 if event else 0]), np.array([event]),
                   [{'overtake_success_event': event}])
    data = buffer._get_samples(np.array([1, 1, 0]))
    assert buffer.event_inserts == 1
    assert buffer.event_draws == 2
    assert buffer.unique_events_sampled == 1
    assert data.rewards.flatten().tolist() == [30, 30, 0]
    assert data.dones.flatten().tolist() == [1, 1, 0]


def test_passive_tracking_does_not_change_simulation():
    _, module = pilot.load_modules()
    config = module.ExperimentConfig(seed=4190, target_lane_vehicle=True)
    scene = module.make_eval_specs(1, config)[0]
    envs = [module.make_env('BAL', config) for _ in range(2)]
    try:
        for env in envs:
            env.reset(seed=scene.exposure_seed)
            module.apply_open_lane_scene(env, scene)
        tracker = pilot.SuccessTracker(envs[1])
        for action in [0] + [1] * 40:
            left = envs[0].step(action)
            right = envs[1].step(action)
            tracker.finish_policy_step(other_terminal=right[2])
            np.testing.assert_array_equal(left[0], right[0])
            assert left[1:4] == right[1:4]
            if left[2] or left[3]:
                break
    finally:
        for env in envs:
            env.close()


def test_staying_left_after_passing_is_not_complete_overtake():
    env, ego, _ = tracker_fixture()
    tracker = pilot.SuccessTracker(env)
    ego.position = np.array([30., 0.])
    ego.lane_index = ('0', '1', 0)
    for _ in range(40):
        env.road.step(1/15)
    assert tracker.passed_in_left_time is not None
    assert not tracker.finish_policy_step()


def test_right_lane_pass_without_left_maneuver_cannot_succeed():
    env, ego, _ = tracker_fixture()
    tracker = pilot.SuccessTracker(env)
    ego.position[0] = 30.
    for _ in range(40):
        env.road.step(1/15)
    assert tracker.passed
    assert tracker.passed_in_left_time is None
    assert not tracker.finish_policy_step()


def test_return_before_passing_is_not_success():
    env, ego, _ = tracker_fixture()
    tracker = pilot.SuccessTracker(env)
    ego.lane_index = ('0', '1', 0)
    ego.position = np.array([25., 0.])
    env.road.step(1/15)
    ego.position = np.array([30., 4.])
    ego.lane_index = ('0', '1', 1)
    for _ in range(40):
        env.road.step(1/15)
    assert tracker.left_completed_time is not None
    assert tracker.passed_in_left_time is None
    assert not tracker.finish_policy_step()


def test_local_retry_keeps_training_parameters_and_verifies_environment():
    root = Path(__file__).parents[1] / 'experiments/overtake_success/scripts'
    regular = (root / 'katana_overtake_B_BAL_S10.pbs').read_text()
    local = (root / 'katana_overtake_local_retry.pbs').read_text()
    command = regular.split('python3 -u experiments/', 1)[1].strip()
    assert local.split('python3 -u experiments/', 1)[1].startswith(command)
    assert 'export EXPECTED_LOCAL_VENV="$VIRTUAL_ENV"' in local
    assert 'sha256sum -c venv.tar.sha256' in local
    assert 'export PATH="${VIRTUAL_ENV}/bin:${PATH}"' in local
    assert 'bin/activate' not in local
    assert 'full_B_BAL_S10_seed4190_local_retry1' in local
