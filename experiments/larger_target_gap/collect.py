"""Collect all nine pilot jobs without silently dropping missing or failed runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from run import JOBS


def truth(value):
    return str(value).lower() in ('true', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status, results = [], []
    for seed, agent in JOBS:
        path = args.root / f'larger_target_gap_pilot_{agent}_100k_seed{seed}'
        complete = path / 'complete.json'
        ok = complete.exists() and json.loads(complete.read_text()).get('complete') is True
        if ok and json.loads(complete.read_text()).get('smoke'):
            raise ValueError(f'Smoke result cannot enter pilot summary: {path}')
        status.append(dict(seed=seed, agent=agent, status='complete' if ok else 'incomplete' if path.exists() else 'missing'))
        if not ok:
            continue
        with (path / 'episode_summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        for variant in ('original', 'no-front', 'open-target-lane', 'matched-speed-front'):
            group = [r for r in rows if r['variant'] == variant]
            if len(group) != 27:
                raise ValueError(f'Expected 27 scenes: {path}/{variant}')
            changed = [r for r in group if truth(r['actual_lane_change_observed'])]
            collisions = sum(truth(r['collision_flag']) for r in group)
            valid_space = [r for r in changed if r['physical_change_target_more_space'] != '']
            results.append(dict(seed=seed, agent=agent, variant=variant, total=len(group),
                changed=len(changed), collisions=collisions,
                no_change_no_collision=sum(not truth(r['actual_lane_change_observed']) and not truth(r['collision_flag']) for r in group),
                change_rate=len(changed) / len(group), collision_rate=collisions / len(group),
                mean_seconds_observed_only=mean(float(r['first_actual_lane_change_seconds']) for r in changed) if changed else '',
                first_step_commands=sum(float(r['first_lane_change_action_seconds']) == 0 for r in group if r['first_lane_change_action_seconds'] != ''),
                space_comparison_observed=len(valid_space),
                target_more_space=sum(truth(r['physical_change_target_more_space']) for r in valid_space)))
    for name, data in (('job_status.csv', status), ('summary.csv', results)):
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
