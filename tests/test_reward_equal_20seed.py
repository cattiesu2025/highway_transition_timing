import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/reward_equal_start"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r20():
    return load("reward_equal_20seed", HERE / "run_20seed.py")


def test_layout_is_400_jobs_with_20_paired_seeds(r20):
    jobs = r20.jobs()
    assert len(jobs) == 400
    assert [j["index"] for j in jobs] == list(range(400))
    blocks = {}
    for j in jobs:
        blocks.setdefault((j["design"], j["arm"], j["variant"]), []).append(j["seed"])
    for (design, arm, variant), seeds in blocks.items():
        base = r20.BASE_SEEDS[arm]
        assert seeds == list(range(base, base + 20))
    assert {k[1] for k in blocks if k[0] == "BAL"} == set("ABCD")
    assert {k[1] for k in blocks if k[0] == "ES"} == set("ABCD")
    assert {k[1] for k in blocks if k[0] == "FL"} == set("ABCD")
    # Agreed submission 0-359 excludes only the optional Arm C first-log block.
    assert {(j["design"], j["arm"]) for j in jobs[360:]} == {("FL", "C")}


def test_manifest_is_frozen(r20):
    assert json.loads(r20.MANIFEST.read_text()) == r20.jobs()


def test_first_log_weights_match_20seed_calibration(r20):
    solved = json.loads((ROOT / "outputs/reward_equal_start/first_log_weights_20seed.json").read_text())
    for arm, table in r20.FL_WEIGHTS.items():
        assert solved["arms"][arm]["seeds"] == list(range(r20.BASE_SEEDS[arm], r20.BASE_SEEDS[arm] + 20))
        for variant, (speed, spacing) in table.items():
            w = solved["arms"][arm]["weights"][variant]
            assert (speed, spacing) == (round(w["speed"], 6), round(w["spacing"], 6))


def test_es_weights_match_earlier_runners(r20):
    run = load("es_run", HERE / "run.py")
    bd = load("es_run_bd", HERE / "run_bd.py")
    for arm in "AC":
        assert r20.ES_WEIGHTS[arm] == run.WEIGHTS[arm]
    for arm in "BD":
        assert r20.ES_WEIGHTS[arm] == {k: bd.WEIGHTS[arm][k] for k in ("SP", "FD")}


def test_run_directories_are_unique_and_new(r20, tmp_path):
    names = [r20.run_directory(tmp_path, j).name for j in r20.jobs()]
    assert len(set(names)) == len(names)
    assert all(("reward_equal20_" in n) or ("_R20-" in n) for n in names)


@pytest.mark.parametrize("index", [0, 20, 40, 80, 150, 240, 300, 380])
def test_sweep_command_carries_weights_and_full_budget(r20, index, tmp_path):
    job = r20.jobs()[index]
    if job["arm"] == "D":
        pytest.skip("Arm D uses the distance-variant runner")
    (cmd,) = r20.calibrate.load_sweep().commands(job, tmp_path, "train", run_dir=r20.run_directory(tmp_path, job))
    assert cmd[cmd.index("--speed-weight") + 1] == str(job["speed_weight"])
    assert cmd[cmd.index("--front-distance-weight") + 1] == str(job["front_distance_weight"])
    assert cmd[cmd.index("--timesteps") + 1] == "100000"
    assert cmd[cmd.index("--seed") + 1] == str(job["seed"])
    assert cmd[cmd.index("--agents") + 1] == job["variant"]
