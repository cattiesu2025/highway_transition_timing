import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/score_normalization"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sn():
    return load("score_normalization_run", HERE / "run.py")


def test_scores_span_environment_ranges(sn):
    assert [sn.speed_score(v) for v in (5, 10, 22.5, 35, 40)] == [0, 0, 0.5, 1, 1]
    assert [sn.front_distance_score(d) for d in (0, 50, 200, 250, math.inf)] == [0, 0.25, 1, 1, 1]


def test_speed_distance_exchange_rate_matches_maintained_scores(sn):
    # Both slopes shrink by the same factor, which the preference scale undoes.
    speed_slope = 1 / (sn.SPEED_RANGE[1] - sn.SPEED_RANGE[0])
    distance_slope = 1 / sn.DISTANCE_HORIZON
    assert speed_slope / (1 / 15) == pytest.approx(0.6)
    assert distance_slope / (1 / 120) == pytest.approx(0.6)
    assert sn.PREFERENCE_SCALE * 0.6 == pytest.approx(1)
    assert sn.BAL == pytest.approx((0.7 * sn.PREFERENCE_SCALE,) * 2, abs=1e-6)


def test_weights_keep_ratio_and_equalise_first_log(sn):
    for arm, (S, D) in sn.FIRST_LOG_SCORES.items():
        target = sn.BAL[0] * (S + D)
        sp, fd = sn.WEIGHTS[arm]["SP"], sn.WEIGHTS[arm]["FD"]
        assert sp[0] == pytest.approx(2 * sp[1], abs=2e-6)
        assert fd[1] == pytest.approx(2 * fd[0], abs=2e-6)
        for speed, spacing in (sp, fd):
            assert speed * S + spacing * D == pytest.approx(target, abs=1e-3)


def test_layout_matches_frozen_manifest(sn):
    jobs = sn.jobs()
    assert jobs == json.loads((HERE / "jobs.json").read_text())
    assert len(jobs) == 24 and [j["index"] for j in jobs] == list(range(24))
    cells = {}
    for job in jobs:
        cells.setdefault((job["arm"], job["variant"]), []).append(job["seed"])
    assert len(cells) == 12
    for (arm, _variant), seeds in cells.items():
        assert tuple(seeds) == sn.SEEDS[arm]
    names = {sn.run_directory(Path("/x"), job).name for job in jobs}
    assert len(names) == 24  # C and D share seeds but not directories.


def test_training_command_is_maintained_command_behind_entry(sn):
    job = sn.jobs()[14]
    assert (job["arm"], job["variant"]) == ("C", "SP")
    (call,) = sn.sweep_commands(job, Path("/x"), smoke=False)
    assert call[1:4] == [str(HERE / "run.py"), "entry", str(ROOT / "experiments/multilane_open_lane_change/run.py")]
    assert call[call.index("--speed-weight") + 1] == "1.572866"
    assert call[call.index("--timesteps") + 1] == "100000"
    assert call[call.index("--collision-risk-penalty") + 1] == "3.0"
    (smoke,) = sn.sweep_commands(job, Path("/x"), smoke=True)
    assert smoke[smoke.index("--timesteps") + 1] == "1200"
    assert "_smoke" in smoke[smoke.index("--out") + 1]


def test_install_replaces_only_the_two_scores_and_is_reversible(sn, monkeypatch):
    from highway_transition_timing import highway_adapter as adapter

    maintained = dict(ego_speed=22.5, nearest_front_distance=150.0, speed_score=1 / 6,
                      front_distance_score=1.0, collision_risk_score=0.25)
    monkeypatch.setattr(adapter, "extract_diagnostics", lambda env: dict(maintained))
    original = adapter.extract_diagnostics
    sn.install()
    sn.install()  # Idempotent.
    try:
        result = adapter.extract_diagnostics(None)
        assert result == dict(maintained, speed_score=0.5, front_distance_score=0.75)
        assert adapter.reward_components(dict(result, collision_flag=False))["front_distance_score"] == 0.75
    finally:
        sn.uninstall()
    assert adapter.extract_diagnostics is original


def test_training_environment_rewards_use_candidate_scores(sn):
    pytest.importorskip("highway_env")
    from highway_transition_timing import highway_adapter as adapter

    sn.install()
    try:
        module = sn.load("score_norm_arm_a", ROOT / "experiments/single_lane_slow_front/run.py")
        module.configure_headless_runtime()
        env = module.make_env("BAL", module.ExperimentConfig(seed=3100, duration=20), training=True)
        try:
            env.reset(seed=3100)
            for _ in range(15):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                state = adapter.extract_diagnostics.maintained(env)
                parts = env.last_reward_components
                assert parts["speed_score"] == pytest.approx(sn.speed_score(state["ego_speed"]))
                assert parts["front_distance_score"] == pytest.approx(
                    sn.front_distance_score(state["nearest_front_distance"]))
                if terminated or truncated:
                    break
        finally:
            env.close()
    finally:
        sn.uninstall()
