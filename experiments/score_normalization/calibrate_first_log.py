"""First-log equal start under candidate scores S=clip((v-10)/25), D=clip(d/200).
Replays each training seed's first 4 (uniform-random) episodes through first_log_calibrate.replay_arm,
logging post-step speed/distance.
Usage: PYTHONPATH=src python calibrate_first_log.py <arm A-D> <seeds, 20>. It checks S, D against the project's first_log_episodes_20seed.csv."""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; OUTDIR = ROOT / "outputs/score_normalization/calibration"; OUTDIR.mkdir(parents=True, exist_ok=True)
arm, nseeds = sys.argv[1], int(sys.argv[2])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
flc = load("flc", ROOT / "experiments/reward_equal_start/first_log_calibrate.py")
scores = flc.load("equal_start_random_scores", ROOT / "experiments/reward_equal_start/random_policy_scores.py")
cal = scores.calibrate
steps, marks = [], []
def patch(module):
    name = "local_reward_components" if hasattr(module, "local_reward_components") else "reward_components"
    orig = getattr(module, name)
    if getattr(orig, "_logged", False): return
    def patched(diag):
        comp = orig(diag)
        steps.append((float(diag["ego_speed"]), float(diag["nearest_front_distance"]), comp["speed_score"], comp["front_distance_score"]))
        return comp
    patched._logged = True; setattr(module, name, patched)
def wrap(fn):
    def cfg(a, seed):
        module, config = fn(a, seed); patch(module); return module, config
    return cfg
cal.training_config = wrap(cal.training_config); scores.arm_d_config = wrap(scores.arm_d_config)
orig_collect = cal.collect
def collect(a, seed, episodes):
    start = len(steps); rows = orig_collect(a, seed, episodes); marks.append((start, rows)); return rows
cal.collect = collect
flc.SEEDS.clear(); flc.SEEDS[arm] = range(flc.BASE_SEEDS[arm], flc.BASE_SEEDS[arm] + nseeds)
rows = flc.replay_arm(arm)
out = []
for start, rs in marks:
    i = start
    for r in rs:
        seg = np.array(steps[i:i + r["length"]]); i += r["length"]
        assert abs(seg[:, 2].sum() - r["S"]) < 1e-6 and abs(seg[:, 3].sum() - r["D"]) < 1e-6, r
        out.append(dict(arm=arm, seed=r["seed"], episode=r["episode"], length=r["length"], S_old=r["S"], D_old=r["D"], c=r["c"],
                        S_new=float(np.clip((seg[:, 0] - 10) / 25, 0, 1).sum()), D_new=float(np.clip(seg[:, 1] / 200, 0, 1).sum())))
ref = {(r["arm"], int(r["seed"]), int(r["episode"])): r for r in csv.DictReader(open(ROOT / "outputs/reward_equal_start/first_log_episodes_20seed.csv"))}
matched = 0
for r in out:
    k = (arm, r["seed"], r["episode"])
    if k in ref:
        assert abs(float(ref[k]["S"]) - r["S_old"]) < 1e-6 and abs(float(ref[k]["D"]) - r["D_old"]) < 1e-6 and int(ref[k]["length"]) == r["length"], (k, ref[k], r)
        matched += 1
with open(OUTDIR / f"first_log_arm{arm}.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(json.dumps(dict(arm=arm, seeds=nseeds, episodes=len(out), matched_project_rows=matched)))
