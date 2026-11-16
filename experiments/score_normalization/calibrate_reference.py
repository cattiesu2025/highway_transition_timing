"""Random-policy (t=0) reference set for arms A-D, logging post-step ego speed and front distance,
so alternative normalisations can be scored offline. Usage: PYTHONPATH=src python calibrate_reference.py <arm A-D> <episodes, 2000>.
Current:  S = clip((v-20)/15), D = clip(d/120).  Candidate: S = clip((v-10)/25), D = clip(d/200)."""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs/score_normalization/calibration"; OUTDIR.mkdir(parents=True, exist_ok=True)
arm, episodes = sys.argv[1], int(sys.argv[2])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
rps = load("rps", ROOT / "experiments/reward_equal_start/random_policy_scores.py")
cal = rps.calibrate
steps = []
def patch(module):
    name = "local_reward_components" if hasattr(module, "local_reward_components") else "reward_components"
    orig = getattr(module, name)
    if getattr(orig, "_logged", False): return
    def patched(diagnostics):
        comp = orig(diagnostics)
        steps.append((float(diagnostics["ego_speed"]), float(diagnostics["nearest_front_distance"]),
                      comp["speed_score"], comp["front_distance_score"], -comp["collision_risk_penalty"]))
        return comp
    patched._logged = True
    setattr(module, name, patched)
if arm == "B":
    cal.ARMS["B"] = ("multilane_open_lane_change", rps.SEED); cal.RESETS_BEFORE_MODEL["B"] = False
if arm == "D":
    cal.RESETS_BEFORE_MODEL["D"] = False; base_cfg = rps.arm_d_config
else:
    base_cfg = cal.training_config
def cfg(a, seed):
    module, config = base_cfg(a, seed); patch(module); return module, config
cal.training_config = cfg
seed = cal.ARMS[arm][1] if arm in cal.ARMS else rps.SEED
rows = cal.collect(arm, seed, episodes)
n_steps = sum(r["length"] for r in rows); assert n_steps <= len(steps)
a = np.array(steps[:n_steps]); v, d, S, D, R = a.T
i = 0
for r in rows:  # alignment check against the wrapper's own per-episode sums
    j = i + r["length"]; assert abs(S[i:j].sum() - r["S"]) < 1e-6 and abs(D[i:j].sum() - r["D"]) < 1e-6, r; i = j
np.savez_compressed(OUTDIR / f"reference_arm{arm}.npz", v=v, d=d, S=S, D=D, R=R, lengths=np.array([r["length"] for r in rows]))
n = len(rows)
def desc(x, name):
    return {"term": name, "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "per_episode": float(x.sum() / n),
            "at0": float((x <= 0).mean()), "at1": float((x >= 1).mean())}
terms = [desc(S, "S_logged"), desc(D, "D_logged"),
         desc(np.clip((v - 20) / 15, 0, 1), "S_20_35"), desc(np.clip((v - 10) / 25, 0, 1), "S_10_35"),
         desc(np.clip((v - 20) / 10, 0, 1), "S_20_30"),
         desc(np.clip(d / 120, 0, 1), "D_120"), desc(np.clip(d / 200, 0, 1), "D_200")]
res = dict(arm=arm, seed=seed, episodes=n, steps=int(n_steps), mean_length=n_steps / n, terms=terms,
           speed=dict(mean=float(v.mean()), p01=float(np.percentile(v, 1)), p05=float(np.percentile(v, 5)),
                      p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                      min=float(v.min()), max=float(v.max())),
           distance=dict(no_front=float(np.isinf(d).mean()), ge200=float(((d >= 200) & ~np.isinf(d)).mean()),
                         band120_200=float(((d >= 120) & (d < 200)).mean()),
                         p05=float(np.percentile(d[~np.isinf(d)], 5)), p50=float(np.percentile(d[~np.isinf(d)], 50)),
                         p95=float(np.percentile(d[~np.isinf(d)], 95))),
           risk_on=float((R > 0).mean()))
json.dump(res, open(OUTDIR / f"reference_arm{arm}.json", "w"), indent=1)
print(json.dumps(res))
