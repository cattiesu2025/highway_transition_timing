"""Equal-start weights: exact algebra, the 2:1 ratios, and block-balanced sets."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def calibrate(monkeypatch):
    path = ROOT / "experiments/reward_equal_start/calibrate.py"
    spec = importlib.util.spec_from_file_location("equal_start_calibrate", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("S,D", [(39.2, 76.4), (32.1, 72.8), (50.0, 50.0), (10.0, 90.0)])
def test_all_three_weight_pairs_score_the_reference_set_identically(calibrate, S, D):
    weights = calibrate.solve_weights(S, D)
    row = {"c": -17.3, "S": S, "D": D}
    bal = calibrate.episode_return(row, weights["BAL"]["speed"], weights["BAL"]["spacing"])
    for name in ("SP", "FD"):
        value = calibrate.episode_return(row, weights[name]["speed"], weights[name]["spacing"])
        assert value == pytest.approx(bal, abs=1e-12), name


def test_preference_ratios_are_two_to_one(calibrate):
    weights = calibrate.solve_weights(39.2, 76.4)
    assert weights["BAL"] == {"speed": 0.7, "spacing": 0.7}
    assert weights["SP"]["speed"] == pytest.approx(2 * weights["SP"]["spacing"])
    assert weights["FD"]["spacing"] == pytest.approx(2 * weights["FD"]["speed"])


def test_episode_return_is_linear_in_the_preference_weights(calibrate):
    row = {"c": -5.0, "S": 10.0, "D": 20.0}
    assert calibrate.episode_return(row, 0.5, 0.25) == pytest.approx(5.0)


@pytest.mark.parametrize("S,D", [(0.0, 10.0), (10.0, 0.0), (-1.0, 10.0)])
def test_rejects_non_positive_components(calibrate, S, D):
    with pytest.raises(ValueError):
        calibrate.solve_weights(S, D)


def test_reference_set_must_cover_whole_scenario_blocks(calibrate, tmp_path):
    with pytest.raises(SystemExit):
        calibrate.reference(tmp_path, 30)


@pytest.mark.parametrize("arm", ["A", "C"])
def test_block_constant_matches_the_arm_training_block(calibrate, arm):
    module = calibrate.load_arm(arm)
    assert len(module.STRATIFIED_TRAINING_BLOCK) == calibrate.BLOCK_EPISODES


def test_training_settings_come_from_the_real_sweep_command(calibrate):
    _module, config = calibrate.training_config("C", 9200)
    assert config.lanes_count == 2 and config.ego_lane == 1
    assert config.target_lane_vehicle is True
    assert config.lane_change_penalty == 0.2 and config.slow_down_penalty == 0.2
    assert config.collision_risk_penalty == 3.0 and config.duration == 20
    assert config.seed == 9200


@pytest.fixture
def sweep(monkeypatch, calibrate):
    path = ROOT / "experiments/reward_equal_start/run.py"
    spec = importlib.util.spec_from_file_location("equal_start_run", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_sweep_is_twenty_paired_new_jobs_without_bal(sweep):
    jobs = sweep.jobs()
    assert [j["index"] for j in jobs] == list(range(20))
    assert {j["variant"] for j in jobs} == {"SP", "FD"}
    for arm, base in sweep.BASE_SEEDS.items():
        for variant in ("SP", "FD"):
            seeds = [j["seed"] for j in jobs if j["arm"] == arm and j["variant"] == variant]
            assert seeds == list(range(base, base + 5))


def test_frozen_manifest_matches_configured_jobs(sweep):
    assert json.loads(sweep.MANIFEST.read_text()) == sweep.jobs()


@pytest.mark.parametrize("arm,S,D", [
    ("A", 33.09936796213824, 73.78941428863084),
    ("C", 31.297065332071266, 73.69823304642752),
])
def test_frozen_weights_are_the_solved_weights_at_six_decimals(sweep, calibrate, arm, S, D):
    solved = calibrate.solve_weights(S, D)
    for variant in ("SP", "FD"):
        assert sweep.WEIGHTS[arm][variant] == (
            round(solved[variant]["speed"], 6), round(solved[variant]["spacing"], 6))


@pytest.mark.parametrize("index", [0, 19])
def test_command_carries_exact_weights_and_full_budget(sweep, index):
    job = sweep.jobs()[index]
    command = sweep.commands(job, ROOT / "tmp" / "unused")[0]
    joined = " ".join(command)
    assert f"--speed-weight {job['speed_weight']}" in joined
    assert f"--front-distance-weight {job['front_distance_weight']}" in joined
    assert "--timesteps 100000" in joined and f"--agents {job['agent']}" in joined
    assert sweep.run_directory(ROOT, job).name.startswith("reward_equal_start_arm")
    assert ("--target-lane-vehicle" in command) == (job["arm"] == "C")


def test_collection_covers_twenty_new_and_ten_reused_bal_runs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments/reward_sensitivity"))
    for name in ("collect", "run"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    path = ROOT / "experiments/reward_equal_start/collect_results.py"
    spec = importlib.util.spec_from_file_location("equal_start_collect", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    tasks = module.tasks(Path("/data"))
    assert len(tasks) == 30
    bal = [(job, directory) for job, directory in tasks if job["variant"] == "BAL"]
    assert len(bal) == 10 and all(job["speed_weight"] == 0.7 for job, _ in bal)
    assert {directory.name for job, directory in bal if job["arm"] == "A"} == {
        f"single_lane_slow_front_fixed_100k_seed{seed}" for seed in range(3100, 3105)}


@pytest.fixture
def first_log(monkeypatch, calibrate):
    path = ROOT / "experiments/reward_equal_start/run_first_log.py"
    spec = importlib.util.spec_from_file_location("first_log_run", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_first_log_sweep_is_thirty_paired_jobs_without_bal_or_arm_c(first_log):
    jobs = first_log.jobs()
    assert [j["index"] for j in jobs] == list(range(30))
    assert {j["arm"] for j in jobs} == {"A", "B", "D"}
    assert {j["variant"] for j in jobs} == {"SP", "FD"}
    for arm, base in first_log.BASE_SEEDS.items():
        for variant in ("SP", "FD"):
            seeds = [j["seed"] for j in jobs if j["arm"] == arm and j["variant"] == variant]
            assert seeds == list(range(base, base + 5))


def test_first_log_manifest_and_weights_are_frozen(first_log):
    assert json.loads(first_log.MANIFEST.read_text()) == first_log.jobs()
    solved = json.loads((ROOT / "outputs/reward_equal_start/first_log_weights.json").read_text())
    for arm, table in first_log.WEIGHTS.items():
        for variant, (speed, spacing) in table.items():
            w = solved["arms"][arm]["weights"][variant]
            assert (speed, spacing) == (round(w["speed"], 6), round(w["spacing"], 6))
            assert speed == pytest.approx(2 * spacing if variant == "SP" else spacing / 2, abs=2e-6)


def test_first_log_weights_equalise_the_replayed_first_log_mean(first_log):
    import csv
    with (ROOT / "outputs/reward_equal_start/first_log_episodes.csv").open() as handle:
        episodes = list(csv.DictReader(handle))
    for arm, table in first_log.WEIGHTS.items():
        rows = [e for e in episodes if e["arm"] == arm]
        assert len(rows) == 20

        def mean(speed, spacing):
            return sum(float(e["c"]) + speed * float(e["S"]) + spacing * float(e["D"]) for e in rows) / 20
        for speed, spacing in table.values():
            assert mean(speed, spacing) == pytest.approx(mean(0.7, 0.7), abs=1e-3)


def test_first_log_replay_matched_every_existing_run():
    import csv
    with (ROOT / "outputs/reward_equal_start/first_log_replay_check.csv").open() as handle:
        checks = list(csv.DictReader(handle))
    assert len(checks) == 45  # A/B/D x 5 seeds x SP/BAL/FD
    for row in checks:
        assert float(row["step"]) == float(row["logged_step"])
        assert float(row["predicted"]) == pytest.approx(float(row["logged"]), abs=1e-3)


@pytest.mark.parametrize("index,arm", [(0, "A"), (10, "B")])
def test_first_log_sweep_command_carries_new_weights_and_names(first_log, index, arm):
    job = first_log.jobs()[index]
    assert job["arm"] == arm
    joined = " ".join(first_log.calibrate.load_sweep().commands(
        job, ROOT / "tmp" / "unused", "train", run_dir=first_log.run_directory(ROOT, job))[0])
    assert f"--speed-weight {job['speed_weight']}" in joined
    assert "--timesteps 100000" in joined and "--target-lane-vehicle" not in joined
    assert first_log.run_directory(ROOT, job).name.startswith("reward_first_log_arm")


def test_first_log_arm_d_names_do_not_collide_with_earlier_runs(first_log):
    job = first_log.jobs()[20]
    assert job["arm"] == "D"
    assert first_log.run_directory(ROOT, job).name == "arm_c_distance-affine-v1_FL-SP_100k_seed4200"
