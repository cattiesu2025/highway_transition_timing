"""Deterministic action-timing return audit, without training or learned policies."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path

import run as pilot

DELAYS = (0, 1, 2, 4)


def contributions(module, components, weights):
    parts = {key: weight * (-abs(components.get(key, 0.)) if key in
             ('lane_change_penalty', 'slow_down_penalty') else components.get(key, 0.))
             for key, weight in asdict(weights).items()}
    assert math.isclose(sum(parts.values()), module.weighted_reward(components, weights), abs_tol=1e-10)
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, m = pilot.load_modules()
    m.configure_headless_runtime()
    pilot.install(m)
    config = m.ExperimentConfig(duration=20, evaluation_duration=20, target_lane_vehicle=True)
    weights = {a: m.with_common_slow_down_penalty(m.MAIN_REWARD_WEIGHTS[a],
               config.slow_down_penalty, config.collision_risk_penalty,
               config.collision_penalty, config.lane_change_penalty) for a in pilot.AGENTS}
    protocol = dict(front_distances=[60, 90, 120, 150], extra_distances=[30, 45],
        front_speeds=[10, 14, 18], ego_speed=28, target_speed=29, horizon_seconds=20,
        gamma_per_policy_step=config.gamma, policy_frequency=config.policy_frequency,
        schemes=['left_at_0', 'left_at_1', 'left_at_2', 'left_at_4', 'stay_idle'],
        longitudinal_control='IDLE, unchanged native target-speed controller; not constant speed or safe following',
        collision_handling='retain terminal collision reward; stop at environment termination; zero future return, no bootstrap',
        weights={a: asdict(w) for a, w in weights.items()}, config=asdict(config))
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    sources = [Path(__file__), Path(pilot.__file__), Path(m.__file__),
               *sorted((pilot.ROOT/'src/highway_transition_timing').glob('*.py'))]
    (args.out/'source_sha256.json').write_text(json.dumps({str(p.relative_to(pilot.ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)+'\n')
    summaries, initials = [], []
    with gzip.open(args.out/'steps.csv.gz', 'wt', newline='') as handle:
        writer = None
        for i, (distance, extra, speed) in enumerate(itertools.product((60,90,120,150), (30,45), (10,14,18))):
            spec = m.build_open_lane_spec(exposure_id=f'TA{i:02d}', exposure_seed=9_400_000+i,
                ego_speed=28, front_distance=distance, front_speed=speed, scenario_type='timing_audit',
                include_front_vehicle=True, ego_lane=1, target_lane_gap=distance+extra)
            initial_reference = None
            for delay in (*DELAYS, None):
                scheme = f'left_at_{delay}' if delay is not None else 'stay_idle'
                env = m.make_env('BAL', config, training=False)
                try:
                    env.reset(seed=spec.exposure_seed)
                    obs = m.apply_open_lane_scene(env, spec)
                    pre = m.extract_diagnostics(env)
                    initial = (obs.tolist(), pre['ego_speed'], pre['lg_original_center_distance'], pre['lg_target_center_distance'])
                    if initial_reference is None:
                        initial_reference = initial
                    assert initial == initial_reference
                    assert obs[:,0].sum() == 3 and not (abs(obs[1:3,1]) >= 1).any()
                    if delay == 0:
                        initials.append(dict(scene=spec.exposure_id, front_distance=distance,
                            target_distance=distance+extra, front_speed=speed,
                            original_gap=pre['lg_original_net_gap'], target_gap=pre['lg_target_net_gap'],
                            front_distance_score=pre['front_distance_score'], risk=pre['collision_risk_score'],
                            ego_initial_target_speed=env.unwrapped.vehicle.target_speed, visible_vehicles=3))
                    totals = {a: {k: 0. for k in asdict(w)} for a,w in weights.items()}
                    undiscounted = {a:0. for a in weights}
                    first_change = None
                    for t in range(config.evaluation_max_policy_steps):
                        action = 0 if delay is not None and t == delay*config.policy_frequency else 1
                        _, reward, term, trunc, _ = env.step(action)
                        post = m.extract_diagnostics(env)
                        if first_change is None and post['ego_lane'] != 1:
                            first_change = (t+1)/config.policy_frequency
                        components = env.last_reward_components
                        row = dict(scene=spec.exposure_id, scheme=scheme, t=t, time_before=t/5,
                            action=m.action_name_from_env(env, action), ego_speed=pre['ego_speed'],
                            ego_target_speed=env.unwrapped.vehicle.target_speed, post_lane=post['ego_lane'],
                            original_gap_before=pre['lg_original_net_gap'], target_gap_before=pre['lg_target_net_gap'],
                            collision=post['collision_flag'], terminated=term, truncated=trunc,
                            **{f'component_{k}':v for k,v in components.items()})
                        for a,w in weights.items():
                            parts = contributions(m, components, w)
                            value = sum(parts.values())
                            if a == 'BAL':
                                assert math.isclose(value, reward, abs_tol=1e-10)
                            row[f'reward_{a}'] = value
                            undiscounted[a] += value
                            for k,v in parts.items():
                                totals[a][k] += config.gamma**t*v
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row)
                        pre = post
                        if term or trunc:
                            break
                    for a in weights:
                        summaries.append(dict(scene=spec.exposure_id, front_distance=distance, extra_gap=extra,
                            target_distance=distance+extra, front_speed=speed, agent=a, scheme=scheme,
                            discounted_return=sum(totals[a].values()), undiscounted_return=undiscounted[a],
                            collision=post['collision_flag'], actual_change_seconds=first_change if first_change is not None else '',
                            elapsed_seconds=(t+1)/5, terminated=term, truncated=trunc,
                            **{f'discounted_{k}':v for k,v in totals[a].items()}))
                finally:
                    env.close()
            print(f'Completed scene {i+1}/24', flush=True)
    m.write_csv_rows(args.out/'initial_states.csv', initials)
    m.write_csv_rows(args.out/'returns.csv', summaries)
    pairs, best = [], []
    for scene in sorted({r['scene'] for r in summaries}):
        for a in weights:
            g=[r for r in summaries if r['scene']==scene and r['agent']==a]
            now=next(r for r in g if r['scheme']=='left_at_0')
            timed=[r for r in g if r['scheme']!='stay_idle']
            maximum=max(r['discounted_return'] for r in timed)
            best.append(dict(scene=scene, agent=a, front_distance=now['front_distance'],
                extra_gap=now['extra_gap'], front_speed=now['front_speed'],
                best_timed_schemes=';'.join(r['scheme'] for r in timed if abs(r['discounted_return']-maximum)<1e-9)))
            for r in g:
                if r is now: continue
                pairs.append(dict(scene=scene, agent=a, front_distance=now['front_distance'],
                    extra_gap=now['extra_gap'], front_speed=now['front_speed'], alternative=r['scheme'],
                    immediate_minus_alternative=now['discounted_return']-r['discounted_return'],
                    immediate_collision=now['collision'], alternative_collision=r['collision'],
                    **{f'delta_{k}':now[f'discounted_{k}']-r[f'discounted_{k}'] for k in asdict(weights[a])}))
    m.write_csv_rows(args.out/'paired_differences.csv', pairs)
    m.write_csv_rows(args.out/'best_timing.csv', best)
    (args.out/'complete.json').write_text(json.dumps(dict(complete=True, scenes=24, trajectories=120,
        reward_evaluations=len(summaries), learned_policy=False))+'\n')


if __name__ == '__main__':
    main()
