"""Collect all nine pilot jobs without silently dropping missing or failed runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from run import JOBS, PROFILES, profile_settings


def truth(value):
    return str(value).lower() in ('true', '1')


def summarize_group(group, seed, agent, variant):
    changed = [r for r in group if truth(r['actual_lane_change_observed'])]
    collisions = sum(truth(r['collision_flag']) for r in group)
    valid_space = [r for r in changed if r['physical_change_target_more_space'] != '']
    return dict(seed=seed, agent=agent, variant=variant, total=len(group),
        changed=len(changed), collisions=collisions,
        no_change_no_collision=sum(not truth(r['actual_lane_change_observed']) and not truth(r['collision_flag']) for r in group),
        change_rate=len(changed) / len(group), collision_rate=collisions / len(group),
        mean_seconds_observed_only=mean(float(r['first_actual_lane_change_seconds']) for r in changed) if changed else '',
        first_step_commands=sum(float(r['first_lane_change_action_seconds']) == 0 for r in group if r['first_lane_change_action_seconds'] != ''),
        space_comparison_observed=len(valid_space),
        target_more_space=sum(truth(r['physical_change_target_more_space']) for r in valid_space))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='pilot-v1')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status, results, distance_results = [], [], []
    settings = profile_settings(args.profile)
    expected_scenes = len(settings['eval_distances']) * len(settings['eval_extras']) * 3
    for seed, agent in JOBS:
        path = args.root / f"{settings['prefix']}_{agent}_100k_seed{seed}"
        complete = path / 'complete.json'
        ok = complete.exists() and json.loads(complete.read_text()).get('complete') is True
        if ok and json.loads(complete.read_text()).get('smoke'):
            raise ValueError(f'Smoke result cannot enter pilot summary: {path}')
        status.append(dict(seed=seed, agent=agent, status='complete' if ok else 'incomplete' if path.exists() else 'missing'))
        if not ok:
            continue
        protocol = json.loads((path / 'protocol.json').read_text())
        if protocol.get('profile', 'pilot-v1') != args.profile or protocol['timesteps'] != 100000:
            raise ValueError(f'Unexpected protocol: {path}')
        with (path / 'episode_summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        with (path / 'evaluation_grid.csv').open() as handle:
            distances = {r['exposure_id']: float(r['front_distance']) for r in csv.DictReader(handle)}
        for variant in ('original', 'no-front', 'open-target-lane', 'matched-speed-front'):
            group = [r for r in rows if r['variant'] == variant]
            if len(group) != expected_scenes or len({r['exposure_id'] for r in group}) != expected_scenes:
                raise ValueError(f'Expected {expected_scenes} unique scenes: {path}/{variant}')
            results.append(summarize_group(group, seed, agent, variant))
            for distance in settings['eval_distances']:
                subset = [r for r in group if distances[r['exposure_id']] == distance]
                if len(subset) != len(settings['eval_extras']) * 3:
                    raise ValueError(f'Incomplete distance stratum: {path}/{variant}/{distance}')
                distance_results.append(dict(front_distance=distance,
                    **summarize_group(subset, seed, agent, variant)))
    for name, data in (('job_status.csv', status), ('summary.csv', results), ('by_distance.csv', distance_results)):
        if data:
            with (args.out / name).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    report = {'complete_jobs': sum(r['status'] == 'complete' for r in status), 'expected_jobs': 9,
              'scope': 'development pilot; observed-only times are descriptive, no inferential ranking'}
    (args.out / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
