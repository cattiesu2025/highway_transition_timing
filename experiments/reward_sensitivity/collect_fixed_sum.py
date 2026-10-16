"""Validate and collect 90 fixed-sum conditions, including 20 reused records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import collect as collector
import collect_symmetric as paired
import run_fixed_sum as fixed

PAIRS = [('F1_F9', 'F1', 'F9'), ('F2_F8', 'F2', 'F8'),
         ('F3_F7', 'F3', 'F7'), ('F4_F6', 'F4', 'F6')]
CF_VARIANTS = {'original', 'no-front', 'matched-speed-front', 'far-front'}


def tasks(root, include_arm_b=False):
    return [(job, fixed.source_directory(root, job)) for job in fixed.conditions(dict(fixed.BASE_SEEDS, B=4100) if include_arm_b else None)]


def verify_training(root, task_list):
    """F labels must not bypass the original collector's R-only prefix check."""
    for job, directory in task_list:
        path = directory / 'training_scenarios.csv'
        if not path.exists():
            # A missing run is diagnosed by the collector; an otherwise complete
            # run without a schedule is rejected by verify_evaluations below.
            continue
        baseline = Path(root) / f"{fixed.PREFIX[job['arm']]}_100k_seed{job['seed']}" / 'training_scenarios.csv'
        def schedule(file):
            return [{k: v for k, v in row.items() if k != 'agent_condition'}
                    for row in collector.rows(file) if row['agent_condition'] == job['agent']]
        original, current = schedule(baseline), schedule(path)
        n = min(len(original), len(current))
        if not n or original[:n] != current[:n]:
            raise ValueError(f'Training scenario prefix differs: {directory}')


def verify_evaluations(task_list, outcomes):
    available = {(r['arm'], r['variant'], int(r['seed'])) for r in outcomes}
    signatures = {}
    for job, directory in task_list:
        if (job['arm'], job['variant'], job['seed']) not in available:
            continue
        for file in (directory / 'training_scenarios.csv',
                     directory / 'models' / f"{job['agent']}_main.zip"):
            if not file.is_file():
                raise FileNotFoundError(file)
        if job['variant'] not in fixed.REUSED:
            planned = next(j for j in fixed.jobs({job['arm']: 4100} if job['arm'] == 'B' else None) if all(j[k] == job[k] for k in ('arm', 'variant', 'seed')))
            if json.loads((directory / 'sweep_job.json').read_text()) != planned:
                raise ValueError(f'Run manifest differs: {directory}')
        signature = {}
        for row in collector.rows(directory / 'analysis' / 'episode_outcomes.csv'):
            if row['agent_condition'] != job['agent']:
                continue
            key = paired.episode_key(dict(row, agent=job['agent']))
            if key in signature:
                raise ValueError(f'Duplicate exposure: {directory}, {key}')
            signature[key] = tuple(row[k] for k in ('exposure_id', 'exposure_seed', 'rollout_id', 'rollout_seed'))
        if len(signature) != 36:
            raise ValueError(f'Expected 36 unique exposures: {directory}')
        key = job['arm'], job['seed']
        if key in signatures and signatures[key] != signature:
            raise ValueError(f'Exposure metadata differs across configurations: {directory}')
        signatures[key] = signature


def counterfactual_status(task_list):
    result = []
    for job, directory in task_list:
        subdir = 'rollout_counterfactual_front_vehicle' if job['arm'] == 'A' else 'rollout_counterfactual_open_lane'
        path = directory / subdir / 'counterfactual_rollout_summary.csv'
        records = [r for r in collector.rows(path) if r.get('agent_condition') == job['agent']
                   and r.get('counterfactual_variant') in CF_VARIANTS] if path.exists() else []
        names = [r['counterfactual_variant'] for r in records]
        count_field = 'n' if job['arm'] == 'A' else 'n_episodes'
        complete = (len(names) == 4 and set(names) == CF_VARIANTS
                    and all(r.get(count_field) == '36' for r in records))
        result.append(dict(arm=job['arm'], variant=job['variant'], seed=job['seed'],
                           source_variant=job['source_variant'], status='complete' if complete else 'missing_or_incomplete',
                           n_variants=len(set(names)), detail=str(path)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=fixed.sweep.ROOT / 'outputs')
    parser.add_argument('--out', type=Path, default=fixed.sweep.ROOT / 'outputs/reward_fixed_sum/analysis')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--include-arm-b', action='store_true', help='Collect all 135 A/B/C conditions')
    args = parser.parse_args(argv)
    task_list = tasks(args.root, args.include_arm_b)
    base_seeds = dict(fixed.BASE_SEEDS, B=4100) if args.include_arm_b else fixed.BASE_SEEDS
    n_arms = len(base_seeds)
    verify_training(args.root, task_list)
    outcomes = collector.collect(args.root, args.out, args.allow_incomplete, tasks=task_list)
    verify_evaluations(task_list, outcomes)
    episodes, seeds, statuses = paired.paired_tables(outcomes, pairs=PAIRS, base_seeds=base_seeds)
    for name, data, fields in (
        ('paired_episode_gaps.csv', episodes, ['arm', 'pair', 'seed', 'gap_seconds']),
        ('paired_seed_gaps.csv', seeds, ['arm', 'pair', 'seed', 'conditional_median_gap_seconds']),
        ('pair_status.csv', statuses, ['arm', 'pair', 'seed', 'status']),
    ):
        paired.write_table(args.out / name, data, fields)
    collector.write(args.out / 'condition_sources.csv', [dict(**job, source_directory=str(directory)) for job, directory in task_list])
    cf = counterfactual_status(task_list)
    collector.write(args.out / 'counterfactual_status.csv', cf)
    complete_cf = sum(row['status'] == 'complete' for row in cf)
    collection_file = args.out / 'collection.json'
    metadata = json.loads(collection_file.read_text())
    metadata.update(expected_new=35*n_arms, expected_reused=10*n_arms, counterfactual_expected=45*n_arms,
                    counterfactual_complete=complete_cf, expected_seed_pairs=20*n_arms, complete_seed_pairs=len(seeds))
    collection_file.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Paired {len(seeds)}/{20*n_arms} mirror seed comparisons; counterfactual coverage {complete_cf}/{45*n_arms}.')
    print('Gap = lower-spacing preference onset minus higher-spacing preference onset; missing onsets remain excluded.')


if __name__ == '__main__':
    main()
