"""Collect checked development logs; missing runs are explicit, never imputed."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
import run as sweep

PREFIX = {'A': 'single_lane_slow_front_fixed', 'B': 'multilane_twolane_open_fixed', 'C': 'multilane_twolane_occupied_fixed'}
BASE = {'FD': (.45, 1.), 'BAL': (.7, .7), 'SP': (1., .25)}


def rows(path):
    with path.open(newline='') as f:
        yield from csv.DictReader(f)


def write(path, data):
    if not data:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in data for k in row)))
        writer.writeheader()
        writer.writerows(data)


def collect(root, out, allow_incomplete=False):
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(job, sweep.run_directory(root, job)) for job in sweep.jobs()]
    for arm, seed0 in sweep.BASE_SEEDS.items():
        for agent, weights in BASE.items():
            for seed in range(seed0, seed0 + 5):
                job = dict(arm=arm, variant=agent, agent=agent, seed=seed,
                           speed_weight=weights[0], front_distance_weight=weights[1])
                tasks.append((job, root / f'{PREFIX[arm]}_100k_seed{seed}'))
    progress, outcomes, summaries, status, counterfactuals = [], [], [], [], []
    for job, directory in tasks:
        key = {k: job[k] for k in ('arm', 'variant', 'agent', 'seed', 'speed_weight', 'front_distance_weight')}
        files = [directory/'training_runs.csv', directory/'training_logs'/job['agent']/'progress.csv',
                 directory/'analysis'/'episode_outcomes.csv', directory/'evaluation'/'steps.csv']
        if job['arm'] != 'A':
            files.append(directory/'analysis'/'actual_lane_change_summary.csv')
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            status.append(dict(**key, status='missing', detail='; '.join(missing)))
            continue
        metadata = [r for r in rows(files[0]) if r['agent_condition'] == job['agent']]
        if len(metadata) != 1:
            raise ValueError(f'Expected one metadata row: {directory}')
        meta = metadata[0]
        expected = {'reward_speed_score': job['speed_weight'], 'reward_front_distance_score': job['front_distance_weight'],
                    'reward_collision_penalty': 2., 'reward_collision_risk_penalty': 3.,
                    'reward_slow_down_penalty': 0. if job['arm'] == 'A' else .2,
                    'reward_lane_change_penalty': .1 if job['arm'] == 'A' else .2,
                    'seed': job['seed'], 'total_timesteps': 100000, 'training_duration_seconds': 20,
                    'evaluation_duration_seconds': 120, 'policy_frequency_hz': 5}
        for field, value in expected.items():
            if not math.isclose(float(meta[field]), value):
                raise ValueError(f'{directory}: {field} differs from planned value {value}')
        if meta['dqn_variant'] != 'double-dqn' or meta['training_scenario_profile'] != 'stratified':
            raise ValueError(f'Incompatible training protocol: {directory}')
        if job['arm'] != 'A':
            occupied = meta['target_lane_vehicle'].lower() == 'true'
            if occupied != (job['arm'] == 'C') or int(meta['lanes_count']) != 2:
                raise ValueError(f'Incorrect lane geometry: {directory}')
        baseline_scenarios = root/f"{PREFIX[job['arm']]}_100k_seed{job['seed']}"/'training_scenarios.csv'
        if job['variant'].startswith('R'):
            def scenario_rows(path):
                return [{k:v for k,v in r.items() if k!='agent_condition'} for r in rows(path) if r['agent_condition']==job['agent']]
            baseline = scenario_rows(baseline_scenarios)
            current = scenario_rows(directory/'training_scenarios.csv')
            common = min(len(baseline),len(current))
            if not common or baseline[:common] != current[:common]:
                raise ValueError(f'Training scenario prefix differs from paired baseline: {directory}')
        trace = []
        for row in rows(files[1]):
            if row.get('rollout/ep_rew_mean'):
                step, reward = float(row['time/total_timesteps']), float(row['rollout/ep_rew_mean'])
                if not math.isfinite(reward):
                    raise ValueError(f'Nonfinite training reward: {directory}')
                trace.append(dict(**key, timesteps=step, episode_reward=reward))
        if not trace or trace[-1]['timesteps'] < 99000:
            status.append(dict(**key, status='incomplete_training', detail=str(directory)))
            continue
        episode_rows = [r for r in rows(files[2]) if r['agent_condition'] == job['agent']]
        if len(episode_rows) != 36:
            raise ValueError(f'Expected 36 development outcomes: {directory}')
        if job['arm'] != 'A':
            physical = {r['episode_id']: r for r in rows(files[4]) if r['agent_condition'] == job['agent']}
            if len(physical) != 36:
                raise ValueError(f'Expected 36 physical onset records: {directory}')
            for row in episode_rows:
                actual = physical[row['episode_id']]
                observed = actual['actual_lane_change_observed'].lower() == 'true'
                row['response_latency_seconds'] = actual['first_actual_lane_change_seconds'] if observed else ''
                row['episode_outcome'] = ('valid_onset' if observed else 'terminal_failure'
                    if row['collision_flag'].lower() == 'true' or row['episode_outcome'] == 'terminal_failure' else 'no_onset_censored')
        # First average within episodes, then across episodes; short crashes do not
        # receive less weight merely because their traces contain fewer steps.
        motion = {}
        for row in rows(files[3]):
            if row['agent_condition'] != job['agent']:
                continue
            values = motion.setdefault(row['episode_id'], [0., 0, 0., 0])
            values[0] += float(row['ego_speed']); values[1] += 1
            distance = row['nearest_front_distance']
            if distance and math.isfinite(float(distance)):
                values[2] += float(distance); values[3] += 1
        if len(motion) != 36:
            raise ValueError(f'Expected 36 motion traces: {directory}')
        speed = sum(v[0]/v[1] for v in motion.values())/36
        finite_spacing = [v[2]/v[3] for v in motion.values() if v[3]]
        spacing = sum(finite_spacing)/len(finite_spacing) if finite_spacing else ''
        progress.extend(trace)
        valid, collisions, censored, failures = [], 0, 0, 0
        for row in episode_rows:
            collision = row['collision_flag'].lower() == 'true'
            collisions += collision
            censored += row['episode_outcome'] == 'no_onset_censored'
            failures += row['episode_outcome'] == 'terminal_failure'
            latency = row.get('response_latency_seconds', '')
            if row['episode_outcome'] == 'valid_onset':
                valid.append(float(latency))
            outcomes.append(dict(**key, episode_id=row['episode_id'], episode_outcome=row['episode_outcome'],
                                 collision=collision, response_latency_seconds=latency))
        summaries.append(dict(**key, n_total=36, n_onset=len(valid), n_censored=censored, n_failure=failures,
                              mean_episode_speed_mps=speed, mean_episode_finite_front_distance_m=spacing,
                              n_episodes_with_finite_front_distance=len(finite_spacing),
                              n_collision=collisions, collision_rate=collisions/36, onset_rate=len(valid)/36,
                              conditional_median_onset_seconds=median(valid) if valid else ''))
        cf_dir = 'rollout_counterfactual_front_vehicle' if job['arm']=='A' else 'rollout_counterfactual_open_lane'
        cf_file = directory/cf_dir/'counterfactual_rollout_summary.csv'
        if cf_file.exists():
            for row in rows(cf_file):
                if row.get('agent_condition') == job['agent'] and row.get('counterfactual_variant') in ('original','no-front','matched-speed-front','far-front'):
                    counterfactuals.append(dict(**key, **row))
        status.append(dict(**key, status='complete', detail=str(directory)))
    write(out/'run_status.csv', status)
    incomplete = [r for r in status if r['status'] != 'complete']
    if incomplete and not allow_incomplete:
        raise RuntimeError(f'{len(incomplete)} of 135 run/condition records incomplete; see run_status.csv. Use --allow-incomplete for explicitly labelled previews only.')
    write(out/'training_progress.csv', progress)
    write(out/'episode_outcomes.csv', outcomes)
    write(out/'seed_summary.csv', summaries)
    write(out/'counterfactual_summary.csv', counterfactuals)
    (out/'collection.json').write_text(json.dumps(dict(expected=135, complete=len(status)-len(incomplete),
                                                     incomplete=len(incomplete), exploratory=True), indent=2)+'\n')
    print(f'Collected {len(status)-len(incomplete)}/135 run/condition records into {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=sweep.ROOT/'outputs')
    parser.add_argument('--out', type=Path, default=sweep.ROOT/'outputs/reward_sensitivity/analysis')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    collect(args.root, args.out, args.allow_incomplete)
