"""Frozen F1/F4/F7 validation: three arms x three configurations x 20 seeds.

Reuse the first five seeds per arm/configuration; train the other 15. Each
completed policy receives standalone rollout controls, without reselection.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

import run_fixed_sum as fixed

BASE_SEEDS = {'A': 3100, 'B': 4100, 'C': 4200}
VARIANTS = ('F1', 'F4', 'F7')
MANIFEST = Path(__file__).with_name('jobs_selected_validation.json')


def jobs():
    return [dict(index=i, **fixed.condition(arm, variant, seed), reuse=seed < BASE_SEEDS[arm]+5)
            for i, (arm, variant, seed) in enumerate(
                (arm, variant, seed) for arm, base in BASE_SEEDS.items()
                for variant in VARIANTS for seed in range(base, base+20))]


def directory(root, job):
    if job['reuse']:
        return fixed.run_directory(root, job)
    return Path(root)/f"reward_selected_validation_arm{job['arm']}_{job['variant']}_100k_seed{job['seed']}"


def verify_model(root, job):
    path = directory(root, job)
    if job['reuse']:
        original_jobs = fixed.jobs({'B': 4100}) if job['arm']=='B' else fixed.jobs()
        expected = next(j for j in original_jobs if all(j[k]==job[k] for k in ('arm','variant','seed')))
        record = path/'sweep_job.json'
    else:
        expected, record = job, path/'validation_job.json'
    if json.loads(record.read_text()) != expected:
        raise ValueError(f'Manifest mismatch: {record}')
    with (path/'training_runs.csv').open() as handle:
        metadata = [r for r in csv.DictReader(handle) if r['agent_condition']==job['agent']]
    if len(metadata)!=1:
        raise ValueError(f'Expected one training record: {path}')
    expected_fields = {'seed':job['seed'], 'total_timesteps':100000,
        'reward_speed_score':job['speed_weight'], 'reward_front_distance_score':job['front_distance_weight'],
        'training_duration_seconds':20, 'evaluation_duration_seconds':120, 'policy_frequency_hz':5,
        'reward_collision_penalty':2, 'reward_collision_risk_penalty':3,
        'reward_slow_down_penalty':0 if job['arm']=='A' else .2,
        'reward_lane_change_penalty':.1 if job['arm']=='A' else .2}
    meta=metadata[0]
    for key, value in expected_fields.items():
        if not math.isclose(float(meta[key]),value):
            raise ValueError(f'{path}: incorrect {key}')
    if meta['dqn_variant']!='double-dqn' or meta['training_scenario_profile']!='stratified':
        raise ValueError(f'{path}: incompatible protocol')
    if job['arm']!='A' and (int(meta['lanes_count'])!=2 or (meta['target_lane_vehicle'].lower()=='true')!=(job['arm']=='C')):
        raise ValueError(f'{path}: incorrect lane geometry')
    model=path/'models'/f"{job['agent']}_main.zip"
    if not model.is_file():
        raise FileNotFoundError(model)
    with (path/'model_selection.csv').open() as handle:
        selected=[r for r in csv.DictReader(handle) if r['agent_condition']==job['agent'] and r['selected'].lower()=='true']
    if len(selected)!=1 or selected[0]['eligible'].lower()!='true':
        raise ValueError(f'{path}: missing eligible selected checkpoint')


def commands(root, job, phase):
    train=[] if job['reuse'] or phase=='evaluate' else fixed.sweep.commands(job,root,'train',run_dir=directory(root,job))
    # A's second original command is a one-step probe, not needed for these figures.
    controls=fixed.sweep.commands(job,root,'evaluate',run_dir=directory(root,job))[:1]
    return train,controls


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,choices=range(180))
    parser.add_argument('--out-root',type=Path,default=fixed.sweep.ROOT/'outputs')
    parser.add_argument('--phase',choices=('complete','evaluate'),default='complete')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv)
    tasks=jobs()
    if args.manifest:
        args.manifest.write_text(json.dumps(tasks,indent=2)+'\n')
    if args.index is None:
        if args.execute: parser.error('--execute requires --index')
        return 0
    if json.loads(MANIFEST.read_text())!=tasks:
        raise ValueError('Frozen validation manifest differs')
    job=tasks[args.index]; root=args.out_root.resolve()
    train,controls=commands(root,job,args.phase)
    print(json.dumps(job),flush=True)
    for call in train+controls: print(shlex.join(call),flush=True)
    if not args.execute: return 0
    env=dict(os.environ,PYTHONPATH=str(fixed.sweep.ROOT/'src'))
    if train:
        path=directory(root,job)
        path.mkdir(parents=True,exist_ok=False)
        (path/'validation_job.json').write_text(json.dumps(job,indent=2)+'\n')
        for call in train: subprocess.run(call,cwd=fixed.sweep.ROOT,env=env,check=True)
    verify_model(root,job)
    for call in controls: subprocess.run(call,cwd=fixed.sweep.ROOT,env=env,check=True)
    print(f"Completed validation task {args.index}: original + four standalone rollout controls",flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
