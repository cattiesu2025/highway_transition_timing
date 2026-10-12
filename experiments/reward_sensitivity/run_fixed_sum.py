"""Fixed-sum development sweep: 70 new jobs; F3/R2 and F5/BAL are reused."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import run as sweep

BASE_SEEDS = {'A': 3100, 'C': 4200}
# Integer hundredths freeze the exact symmetric design, with sum 140.
COEFFICIENTS = {
    'F1': (28, 112), 'F2': (35, 105), 'F3': (42, 98),
    'F4': (56, 84), 'F5': (70, 70), 'F6': (84, 56),
    'F7': (98, 42), 'F8': (105, 35), 'F9': (112, 28),
}
REUSED = {'F3': 'R2', 'F5': 'BAL'}
PREFIX = {'A': 'single_lane_slow_front_fixed', 'C': 'multilane_twolane_occupied_fixed'}
MANIFEST = Path(__file__).with_name('jobs_fixed_sum.json')


def condition(arm, variant, seed):
    speed, spacing = COEFFICIENTS[variant]
    group = 'FD-like' if spacing > 2 * speed else 'SP-like' if 2 * spacing < speed else 'Intermediate'
    agent = {'FD-like': 'FD', 'Intermediate': 'BAL', 'SP-like': 'SP'}[group]
    return dict(arm=arm, variant=variant, agent=agent, seed=seed,
                speed_weight=speed / 100, front_distance_weight=spacing / 100,
                total_weight=1.4, group=group, source_variant=REUSED.get(variant, variant))


def conditions():
    return [condition(arm, variant, seed)
            for arm, base in BASE_SEEDS.items() for variant in COEFFICIENTS
            for seed in range(base, base + 5)]


def jobs():
    return [dict(index=i, **job) for i, job in enumerate(
        job for job in conditions() if job['variant'] not in REUSED)]


def run_directory(root, job):
    return Path(root) / f"reward_fixed_sum_arm{job['arm']}_{job['variant']}_100k_seed{job['seed']}"


def source_directory(root, job):
    if job['variant'] == 'F3':
        return sweep.run_directory(root, dict(job, variant='R2'))
    if job['variant'] == 'F5':
        return Path(root) / f"{PREFIX[job['arm']]}_100k_seed{job['seed']}"
    return run_directory(root, job)


def commands(job, root, phase):
    if job['variant'] in REUSED:
        raise ValueError('Reused conditions must not be submitted for training or evaluation here')
    return sweep.commands(job, root, phase, run_dir=run_directory(root, job))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(70))
    parser.add_argument('--out-root', type=Path, default=sweep.ROOT / 'outputs')
    parser.add_argument('--phase', choices=('train', 'evaluate'), default='train')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path, help='Export the 70 new jobs; no execution required')
    args = parser.parse_args(argv)
    tasks = jobs()
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(tasks, indent=2) + '\n')
    if args.index is None:
        if args.execute:
            parser.error('--execute requires --index')
        return 0
    job = tasks[args.index]
    root = args.out_root.resolve()
    directory = run_directory(root, job)
    calls = commands(job, root, args.phase)
    if args.execute:
        if json.loads(MANIFEST.read_text()) != tasks:
            raise ValueError('Frozen fixed-sum manifest differs from the configured jobs')
        if args.phase == 'train':
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'sweep_job.json').write_text(json.dumps(job, indent=2) + '\n')
        else:
            if json.loads((directory / 'sweep_job.json').read_text()) != job:
                raise ValueError('Run manifest does not match requested fixed-sum task')
            if not (directory / 'models' / f"{job['agent']}_main.zip").is_file():
                raise FileNotFoundError('Selected model is missing')
    for call in calls:
        print(shlex.join(call), flush=True)
        if args.execute:
            subprocess.run(call, cwd=sweep.ROOT,
                           env=dict(os.environ, PYTHONPATH=str(sweep.ROOT / 'src')), check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
