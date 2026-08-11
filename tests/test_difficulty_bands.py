"""The two experiments must share one difficulty definition.

`VISIBLE_DIFFICULTY_CELLS` and `classify_ttc` are duplicated in the two
experiment modules. If they drift apart the experiments train on different
hazard distributions while appearing to share one, which would confound the
cross-experiment comparison silently. These tests pin them together, and pin
the cell boundaries to the classifier that labels the sampled result.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


single = load("bands_single", "experiments/single_lane_slow_front/run.py")
multi = load("bands_multi", "experiments/multilane_open_lane_change/run.py")


def test_difficulty_cells_match_across_experiments():
    assert single.VISIBLE_DIFFICULTY_CELLS == multi.VISIBLE_DIFFICULTY_CELLS


def test_classifiers_agree_across_experiments():
    for ttc in [x / 4 for x in range(4, 160)]:
        assert single.classify_ttc(ttc, 10.0) == multi.classify_ttc(ttc, 10.0)


def test_every_cell_lies_inside_the_training_episode():
    """A cell whose contact cannot occur inside the episode is unusable."""
    for (_ttc_bin, _decel_bin), (ttc_range, _decel) in (
        multi.VISIBLE_DIFFICULTY_CELLS.items()
    ):
        assert ttc_range[1] <= 20.0


def test_classifier_mirrors_the_cell_boundaries():
    """The sampler draws from a cell and the classifier labels the result."""
    for (ttc_bin, _decel_bin), (ttc_range, _decel) in (
        multi.VISIBLE_DIFFICULTY_CELLS.items()
    ):
        low, high = ttc_range
        for ttc in (low, (low + high) / 2, high):
            assert multi.classify_ttc(ttc, 10.0) == ttc_bin


def test_beyond_horizon_starts_above_the_episode_length():
    assert multi.classify_ttc(20.0, 10.0) == "gradual"
    assert multi.classify_ttc(20.001, 10.0) == "beyond_horizon"


def test_every_cell_is_reachable_by_the_rejection_sampler():
    """A declared cell that cannot be sampled would stall training."""
    rng = random.Random(0)
    for (ttc_bin, _decel_bin), (ttc_range, decel_range) in (
        multi.VISIBLE_DIFFICULTY_CELLS.items()
    ):
        accepted = 0
        for _ in range(2000):
            ttc = rng.uniform(*ttc_range)
            decel = rng.uniform(*decel_range)
            closing = 2.0 * decel * ttc
            front_distance = closing * ttc + multi.VEHICLE_LENGTH_METRES
            ego_speed = rng.uniform(*multi.EGO_SPEED_RANGE)
            front_speed = ego_speed - closing
            low, high = multi.VISIBLE_TRAIN_DISTANCE_RANGE
            if not low <= front_distance <= high:
                continue
            low, high = multi.SLOW_FRONT_SPEED_RANGE
            if not low <= front_speed <= high:
                continue
            accepted += 1
        assert accepted > 100, f"{ttc_bin} is hard to sample: {accepted}/2000"


def test_evaluation_grids_match_and_stay_observable():
    """Both grids must sit inside the observation clip and training range."""
    for module in (single, multi):
        distances = {
            spec.front_distance
            for spec in module.make_eval_specs(36, module.ExperimentConfig())
        }
        assert distances == {150.0, 165.0, 180.0, 195.0}
        assert max(distances) <= module.OBSERVATION_DISTANCE_METRES
        assert max(distances) <= module.VISIBLE_TRAIN_DISTANCE_RANGE[1]
