"""Arm C places a vehicle in the target lane so the lane change becomes a trade.

With an empty target lane a lane change maximises the speed and the
front-distance terms at once, so the FD and SP preferences point at the same
action and cannot disagree. Occupying the target lane at the ego target speed
puts the cost on the distance axis, where the reward weights differ most.

These tests check the scenario is built as designed and that arm C differs from
arm B in exactly one variable, so the two arms stay comparable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location(
        "occupied_multi", REPO_ROOT / "experiments/multilane_open_lane_change/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["occupied_multi"] = module
    spec.loader.exec_module(module)
    return module


run = load()


def test_arms_differ_only_in_the_merge_gap_axis():
    """B and C share ego-lane distances and front speeds; only the third axis moves."""
    open_lane = run.make_eval_specs(36, run.ExperimentConfig())
    occupied = run.make_eval_specs(
        36, run.ExperimentConfig(target_lane_vehicle=True)
    )
    for field in ("front_distance", "front_speed"):
        assert {getattr(spec, field) for spec in open_lane} == {
            getattr(spec, field) for spec in occupied
        }
    assert {spec.target_lane_gap for spec in open_lane} == {None}
    assert {spec.target_lane_gap for spec in occupied} == set(
        run.TARGET_LANE_MERGE_GAPS
    )


def test_both_grids_are_full_factorial():
    for config in (
        run.ExperimentConfig(),
        run.ExperimentConfig(target_lane_vehicle=True),
    ):
        specs = run.make_eval_specs(36, config)
        cells = {
            (
                spec.front_distance,
                spec.front_speed,
                spec.ego_speed,
                spec.target_lane_gap,
            )
            for spec in specs
        }
        assert len(cells) == 36


def test_merge_gaps_span_the_predicted_switching_thresholds():
    """Solving the frozen weights puts BAL near 19 m and FD near 64 m.

    The sweep has to straddle both, otherwise every agent gives the same answer
    at every level and the dose response carries no information.
    """
    gaps = run.TARGET_LANE_MERGE_GAPS
    assert min(gaps) < 19.0 < max(gaps)
    assert min(gaps) < 64.0 < max(gaps)


def test_target_lane_vehicle_is_placed_beside_the_ego():
    config = run.ExperimentConfig(target_lane_vehicle=True, evaluation_duration=20)
    spec = run.make_eval_specs(36, config)[0]
    env = run.make_env("FD", config, training=False)
    env.reset(seed=spec.exposure_seed)
    run.apply_open_lane_scene(env, spec)

    vehicles = env.unwrapped.road.vehicles
    ego = env.unwrapped.vehicle
    target_lane = [
        vehicle
        for vehicle in vehicles
        if vehicle is not ego and vehicle.lane_index[2] == spec.ego_lane - 1
    ]
    assert len(target_lane) == 1
    merged = target_lane[0]
    assert merged.speed == run.TARGET_LANE_SPEED
    assert merged.position[0] - ego.position[0] == spec.target_lane_gap
    env.close()


def test_target_lane_speed_removes_the_speed_trade():
    """Matched to the ego target, so changing lanes costs spacing, not speed."""
    assert 28.0 <= run.TARGET_LANE_SPEED <= 30.0


def test_training_occupies_the_target_lane_whenever_a_front_vehicle_exists():
    config = run.ExperimentConfig(target_lane_vehicle=True)
    specs = [run.make_training_spec(index, config) for index in range(40)]
    with_front = [spec for spec in specs if spec.include_front_vehicle]
    assert all(spec.target_lane_gap is not None for spec in with_front)
    assert all(
        spec.target_lane_gap is None
        for spec in specs
        if not spec.include_front_vehicle
    )
    gaps = [spec.target_lane_gap for spec in with_front]
    low, high = run.TARGET_LANE_TRAIN_GAP_RANGE
    assert all(low <= gap <= high for gap in gaps)
    # Drawn across the sweep rather than from its three levels, so the policy is
    # not fitted to the evaluation grid.
    assert len(set(gaps)) > len(run.TARGET_LANE_MERGE_GAPS)


def test_open_lane_training_leaves_the_target_lane_empty():
    config = run.ExperimentConfig()
    specs = [run.make_training_spec(index, config) for index in range(40)]
    assert all(spec.target_lane_gap is None for spec in specs)
