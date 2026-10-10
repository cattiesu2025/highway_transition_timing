"""Collect 20 new + 20 existing conditions and export paired SP-minus-FD gaps."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import median
import collect as collector
import run as sweep
import run_symmetric as supplement

PAIRS = [('sum1.25', 'R7', 'SP'), ('sum1.45', 'FD', 'R8')]


def tasks(root):
    result = [(j, sweep.run_directory(root, j)) for j in supplement.jobs()]
    for arm, base in supplement.BASE_SEEDS.items():
        for agent in ('FD', 'SP'):
            speed, spacing = collector.BASE[agent]
            for seed in range(base, base + 5):
                job = dict(arm=arm, variant=agent, agent=agent, seed=seed,
                           speed_weight=speed, front_distance_weight=spacing)
                result.append((job, root / f'{collector.PREFIX[arm]}_100k_seed{seed}'))
    return result


def episode_key(row):
    prefix = row['agent'] + '_'
    if not row['episode_id'].startswith(prefix):
        raise ValueError(f"Unexpected episode ID: {row['episode_id']}")
    return row['episode_id'][len(prefix):]


def paired_tables(outcomes):
    groups = defaultdict(dict)
    for row in outcomes:
        group = groups[row['arm'], row['variant'], int(row['seed'])]
        key = episode_key(row)
        if key in group:
            raise ValueError(f'Duplicate paired episode: {key}')
        group[key] = row
    episodes, seeds, statuses = [], [], []
    for arm, base in supplement.BASE_SEEDS.items():
        for pair, fd, sp in PAIRS:
            for seed in range(base, base + 5):
                key = dict(arm=arm, pair=pair, fd_variant=fd, sp_variant=sp, seed=seed)
                left, right = groups.get((arm, fd, seed)), groups.get((arm, sp, seed))
                if left is None or right is None:
                    statuses.append(dict(**key, status='missing_condition'))
                    continue
                if left.keys() != right.keys() or len(left) != 36:
                    raise ValueError(f'Paired scene IDs do not match: {key}')
                gaps = []
                for eid in sorted(left):
                    f, s = left[eid], right[eid]
                    valid = f['episode_outcome'] == s['episode_outcome'] == 'valid_onset'
                    gap = float(s['response_latency_seconds']) - float(f['response_latency_seconds']) if valid else ''
                    if valid:
                        if not math.isfinite(gap):
                            raise ValueError(f'Nonfinite paired gap: {key}, {eid}')
                        gaps.append(gap)
                    episodes.append(dict(**key, scenario_rollout=eid,
                        fd_outcome=f['episode_outcome'], sp_outcome=s['episode_outcome'],
                        fd_collision=f['collision'], sp_collision=s['collision'],
                        fd_onset_seconds=f['response_latency_seconds'],
                        sp_onset_seconds=s['response_latency_seconds'], gap_seconds=gap))
                seeds.append(dict(**key, n_pairs=36, n_valid_pairs=len(gaps),
                    n_excluded_pairs=36-len(gaps),
                    n_positive=sum(g > 1e-8 for g in gaps),
                    n_negative=sum(g < -1e-8 for g in gaps),
                    n_tied=sum(abs(g) <= 1e-8 for g in gaps),
                    conditional_median_gap_seconds=median(gaps) if gaps else ''))
                statuses.append(dict(**key, status='complete'))
    return episodes, seeds, statuses


def verify_exposure_pairing(task_list, outcomes):
    """Check logged scene/rollout identifiers and seeds, not row order or policy IDs."""
    available = {(r['arm'], r['variant'], int(r['seed'])) for r in outcomes}
    signatures = {}
    fields = ('exposure_id', 'exposure_seed', 'rollout_id', 'rollout_seed')
    for job, directory in task_list:
        key = (job['arm'], job['variant'], job['seed'])
        if key not in available:
            continue
        records = [r for r in collector.rows(directory/'analysis'/'episode_outcomes.csv')
                   if r['agent_condition'] == job['agent']]
        signature = {}
        for row in records:
            scene = episode_key(dict(row, agent=job['agent']))
            signature[scene] = tuple(row[field] for field in fields)
        if len(signature) != 36:
            raise ValueError(f'Nonunique exposure records: {key}')
        signatures[key] = signature
    for arm, base in supplement.BASE_SEEDS.items():
        for _, fd, sp in PAIRS:
            for seed in range(base, base + 5):
                a, b = signatures.get((arm, fd, seed)), signatures.get((arm, sp, seed))
                if a is not None and b is not None and a != b:
                    raise ValueError(f'Paired exposure metadata differs: {arm}, {fd}, {sp}, {seed}')


def write_table(path, rows, fields):
    # Write an empty header too: a rerun must not leave an earlier paired result.
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=sweep.ROOT/'outputs')
    parser.add_argument('--out', type=Path, default=sweep.ROOT/'outputs/reward_symmetric/analysis')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    task_list = tasks(args.root)
    outcomes = collector.collect(args.root, args.out, args.allow_incomplete, tasks=task_list)
    verify_exposure_pairing(task_list, outcomes)
    episodes, seeds, statuses = paired_tables(outcomes)
    for name, data, fields in (
        ('paired_episode_gaps.csv', episodes, ['arm', 'pair', 'seed', 'gap_seconds']),
        ('paired_seed_gaps.csv', seeds, ['arm', 'pair', 'seed', 'conditional_median_gap_seconds']),
        ('pair_status.csv', statuses, ['arm', 'pair', 'seed', 'status']),
    ):
        write_table(args.out/name, data, fields)
    print(f'Paired {len(seeds)}/20 seed comparisons; gap = SP onset - FD onset.')


if __name__ == '__main__':
    main()
