"""Collect all pilot runs, including gate failures; never cherry-pick checkpoints."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import median


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'outputs')
    args=parser.parse_args();out=args.root/'risk_penalty_ablation/analysis';out.mkdir(parents=True,exist_ok=True)
    statuses=[];results=[];risks=[]
    for mode in ['baseline','early30','smooth6']:
        for seed in [4201,4202]:
            d=args.root/f'risk_penalty_armC_R6_{mode}_100k_seed{seed}'
            complete=d/'complete.json'
            if not complete.exists():
                statuses.append(dict(mode=mode,seed=seed,status='incomplete_or_missing'));continue
            protocol=json.loads((d/'protocol.json').read_text())
            assert protocol['mode']==mode and protocol['seed']==seed and not protocol['smoke']
            assert protocol['timesteps']==100000 and protocol['primary_checkpoint']=='final_fixed_step'
            gate=json.loads((d/'final_checkpoint_gate.json').read_text())
            for variant in ['original','no-front','matched-speed-front','far-front']:
                with (d/'evaluation_controls'/variant/'episode_summary.csv').open() as h: eps=list(csv.DictReader(h))
                assert len(eps)==36
                # Physical crossing is observed after the step; use exported seconds.
                times=[float(r['first_actual_lane_change_seconds']) for r in eps if r['first_actual_lane_change_seconds']!='']
                collisions=sum(r['collision_flag'].lower()=='true' for r in eps)
                results.append(dict(mode=mode,seed=seed,control=variant,n=36,collisions=collisions,collision_rate=collisions/36,
                    n_actual_onset=len(times),conditional_median_actual_onset_s=median(times) if times else '',final_gate_eligible=gate['eligible']))
            grouped={};nsteps=0
            with gzip.open(d/'training_risk_steps.csv.gz','rt') as h:
                for row in csv.DictReader(h):
                    nsteps+=1
                    r=grouped.setdefault(row['scenario_type'],dict(steps=0,active_steps=0,penalty_sum=0.,collision_steps=0))
                    r['steps']+=1;r['active_steps']+=float(row['risk_score'])>1e-12
                    r['penalty_sum']+=float(row['risk_reward']);r['collision_steps']+=row['collision'].lower()=='true'
            assert nsteps==100000
            for family,r in grouped.items():risks.append(dict(mode=mode,seed=seed,family=family,**r,active_fraction=r['active_steps']/r['steps'],active_seconds=r['active_steps']*.2))
            statuses.append(dict(mode=mode,seed=seed,status='complete'))
    for name,rows in [('run_status',statuses),('control_comparison',results),('training_risk_summary',risks)]:
        if rows:
            with (out/f'{name}.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f"{sum(r['status']=='complete' for r in statuses)}/6 runs complete: {out}")

if __name__=='__main__':main()
