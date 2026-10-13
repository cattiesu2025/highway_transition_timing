"""Collect completed trigger diagnostics, retaining censored and inapplicable rows."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    old = root / 'old_policy_2seed'
    if not (old / 'complete.json').exists():
        raise RuntimeError('Old-policy evaluation is incomplete')
    groups = defaultdict(list)
    for r in read(old / 'old_policy_episodes.csv'):
        if r.get('success_definition') != 'complete_overtake_return_v1':
            raise ValueError('Cannot mix historical left-lane success with complete overtaking')
        groups[r['arm'], r['agent'], r['target_lane_gap']].append(r)
    summary = []
    for (arm, agent, gap), rows in sorted(groups.items()):
        times = [float(r['success_time']) for r in rows if r['success_time']]
        summary.append(dict(arm=arm, agent=agent, target_lane_gap=gap,
                            total=len(rows), success20=sum(r['success_by_20']=='True' for r in rows),
                            success120=len(times), median_success_time=median(times) if times else '',
                            collision_before_stop=sum(r['collision']=='True' for r in rows),
                            ever_left=sum(r['left_entered']=='True' for r in rows),
                            left_completed=sum(bool(r['left_completed_time']) for r in rows),
                            passed_in_left=sum(bool(r['passed_in_left_time']) for r in rows),
                            returned_ahead=sum(bool(r['returned_ahead_time']) for r in rows),
                            ever_clearance5=sum(r['ever_clearance_5m']=='True' for r in rows)))
    write(root / 'old_policy_summary.csv', summary)
    training, families = [], []
    # short_* runs compressed exploration and are retained but explicitly excluded.
    for folder in sorted([*root.glob('prefix_*'), *root.glob('full_*')]):
        if not (folder / 'complete.json').exists():
            raise RuntimeError(f'Incomplete run: {folder}')
        report = json.loads((folder / 'training_summary.json').read_text())
        if report.get('success_definition') != 'complete_overtake_return_v1':
            raise ValueError(f'Wrong success definition: {folder}')
        if report.get('exploration_decay_steps') != 25000:
            raise ValueError(f'Wrong exploration schedule: {folder}')
        rows = read(folder / 'training_episodes.csv')
        eligible = [r for r in rows if r['success_applicable']=='True' and r['reason']!='budget_stop']
        summary = {k:report[k] for k in ['arm','agent','seed','group','steps','first_success_step',
                    'success_transitions_inserted','success_transition_sample_draws',
                    'unique_success_transitions_sampled','final_epsilon']}
        training.append(dict(run=folder.name, **summary,
                             completed_applicable=len(eligible),
                             completed_applicable_successes=sum(bool(r['success_time']) for r in eligible),
                             budget_stop=sum(r['reason']=='budget_stop' for r in rows)))
        by_family = defaultdict(list)
        for r in rows:
            by_family[r['scenario_type']].append(r)
        for family, rs in sorted(by_family.items()):
            completed = [r for r in rs if r['reason']!='budget_stop']
            families.append(dict(run=folder.name, scenario_type=family,
                                 observed_episodes=len(rs), completed=len(completed),
                                 applicable_completed=sum(r['success_applicable']=='True' for r in completed),
                                 successes=sum(bool(r['success_time']) for r in completed),
                                 collisions=sum(r['collision']=='True' for r in completed),
                                 budget_stop=len(rs)-len(completed)))
    write(root / 'training_summary.csv', training)
    write(root / 'training_by_family.csv', families)
    print(json.dumps(dict(old_policy_groups=len(groups), training_runs=len(training)), indent=2))


if __name__ == '__main__':
    main()
