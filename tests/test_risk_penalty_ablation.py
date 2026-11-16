import importlib.util
import math
from pathlib import Path
import pytest
from highway_transition_timing.highway_adapter import closing_collision_risk_score

spec = importlib.util.spec_from_file_location('risk_pilot', Path(__file__).parents[1]/'experiments/risk_penalty_ablation/run.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_baseline_is_unchanged():
    for d in [0.01, 5., 12., 24., 90., 150., float('inf')]:
        for ego in [10., 28., 30.]:
            for front in [None, 10., 28., 30., 35.]:
                assert pilot.risk_score(d, ego, front, 'baseline') == pytest.approx(closing_collision_risk_score(d, ego, front))


def test_modes_separate_long_range_closing_without_penalizing_safe_nonclosing():
    assert pilot.risk_score(150,30,28,'baseline') == 0
    assert pilot.risk_score(150,30,28,'early30') == 0
    assert pilot.risk_score(150,30,28,'smooth6') == pytest.approx(12/162)
    assert pilot.risk_score(50,30,28,'early30') > 0
    for mode in pilot.MODES:
        assert pilot.risk_score(150,28,28,mode) == 0
        assert pilot.risk_score(math.inf,30,None,mode) == 0
        assert pilot.risk_score(12,28,28,mode) == .5


def test_signal_monotonic_and_bounded():
    for mode in pilot.MODES:
        scores = [pilot.risk_score(d,30,28,mode) for d in [200,100,50,24,12,5,.01]]
        assert scores == sorted(scores)
        assert all(0 <= s <= 1 for s in scores)
