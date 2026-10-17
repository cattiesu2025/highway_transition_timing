"""Validate selected-weight runs and export onset states for two validation figures."""
import argparse
import json
import math
from pathlib import Path

import collect as base
import collect_fixed_sum as fixed_collect
import run_selected_validation as selected


def onset_states(job, path):
    outcomes={r['episode_id']:r for r in base.rows(path/'analysis/episode_outcomes.csv')
              if r['agent_condition']==job['agent']}
    if len(outcomes)!=36: raise ValueError(f'Expected 36 exposures: {path}')
    physical={} if job['arm']=='A' else {r['episode_id']:r for r in base.rows(path/'analysis/actual_lane_change_summary.csv')
                                        if r['agent_condition']==job['agent']}
    onsets={}
    for episode,r in outcomes.items():
        if job['arm']=='A':
            t=r['target_onset_t'] if r['episode_outcome']=='valid_onset' else ''
        else:
            row=physical[episode]
            t=row['first_actual_lane_change_t'] if row['actual_lane_change_observed'].lower()=='true' else ''
        onsets[episode]=None if t in ('','None') else int(float(t))
    gaps={}
    for r in base.rows(path/'evaluation/steps.csv'):
        if r['episode_id'] in onsets and onsets[r['episode_id']]==int(r['t']):
            gap=float(r['nearest_front_distance'])
            gaps[r['episode_id']]=gap if math.isfinite(gap) else ''
    result=[]
    for episode,r in outcomes.items():
        t=onsets[episode]
        if t is not None and episode not in gaps: raise ValueError(f'Missing onset state: {path}, {episode}')
        result.append(dict(arm=job['arm'],variant=job['variant'],agent=job['agent'],seed=job['seed'],
            exposure_id=r['exposure_id'],exposure_seed=r['exposure_seed'],rollout_id=r['rollout_id'],rollout_seed=r['rollout_seed'],
            onset_t='' if t is None else t,onset_gap_m=gaps.get(episode,''),
            onset_observed=t is not None,left_censored=t==0,collision=r['collision_flag']))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=selected.fixed.sweep.ROOT/'outputs')
    parser.add_argument('--out',type=Path,default=selected.fixed.sweep.ROOT/'outputs/reward_selected_validation/analysis')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args(argv)
    tasks=[(j,selected.directory(args.root,j)) for j in selected.jobs()]
    fixed_collect.verify_training(args.root,tasks)
    outcomes=base.collect(args.root,args.out,args.allow_incomplete,tasks=tasks)
    available={(r['arm'],r['variant'],int(r['seed'])) for r in outcomes}
    states=[]; signatures={}
    for job,path in tasks:
        if (job['arm'],job['variant'],job['seed']) not in available: continue
        selected.verify_model(args.root,job)
        records=onset_states(job,path)
        signature=sorted(tuple(r[k] for k in ('exposure_id','exposure_seed','rollout_id','rollout_seed')) for r in records)
        key=job['arm'],job['seed']
        if key in signatures and signatures[key]!=signature: raise ValueError(f'Unpaired exposures: {path}')
        signatures[key]=signature
        states.extend(records)
    base.write(args.out/'onset_states.csv',states)
    base.write(args.out/'condition_sources.csv',[dict(**j,source_directory=str(p)) for j,p in tasks])
    coverage=fixed_collect.counterfactual_status(tasks)
    base.write(args.out/'counterfactual_status.csv',coverage)
    complete=sum(r['status']=='complete' for r in coverage)
    path=args.out/'collection.json'; meta=json.loads(path.read_text())
    meta.update(expected_reused=45,expected_new=135,counterfactual_expected=180,counterfactual_complete=complete,
                figures_ready=meta['complete']==180 and complete==180)
    path.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Standalone controls: {complete}/180; figures_ready={meta["figures_ready"]}')
    if complete!=180 and not args.allow_incomplete:
        raise RuntimeError('Standalone controls incomplete; do not present final validation figures')


if __name__=='__main__':
    main()
