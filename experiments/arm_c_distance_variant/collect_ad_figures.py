"""Non-visual source-data preparation for A + D figures; no failed runs omitted from status."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/arm_ad_figures/source_data'
A = ROOT / 'outputs/reward_selected_validation/analysis'
D = ROOT / 'outputs/arm_c_distance_figure_data'
VARIANTS = ('original', 'far-front', 'matched-speed-front', 'no-front')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def flag(s):
    return str(s).lower() == 'true'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((D / 'figure_data_manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == digest
    runs, episodes, controls = [], [], []
    for r in read(A / 'run_status.csv'):
        if r['arm'] == 'A':
            assert r['status'] == 'complete'
            runs.append(dict(arm='A', agent=r['agent'], seed=int(r['seed']), status='complete'))
    for r in read(A / 'episode_outcomes.csv'):
        if r['arm'] != 'A': continue
        observed = r['episode_outcome'] == 'valid_onset'
        episodes.append(dict(arm='A', agent=r['agent'], seed=int(r['seed']),
            exposure_id=r['episode_id'].split('_')[1], observed=observed,
            seconds=float(r['response_latency_seconds']) if observed else '', collision=flag(r['collision'])))
    for r in read(A / 'counterfactual_summary.csv'):
        if r['arm'] != 'A' or r['record_type'] != 'agent_outcome': continue
        n=int(float(r['valid_onset_count']))
        controls.append(dict(arm='A', agent=r['agent'], seed=int(r['seed']),
            control=r['counterfactual_variant'], n_observed=n, n_total=int(float(r['n'])),
            seconds=float(r['median_response_latency_seconds']) if n else ''))
    for agent,v in [('FD','F1'),('BAL','F4'),('SP','F7')]:
        for seed in range(4200,4220):
            p=D/f'arm_c_distance-affine-v1_{v}_100k_seed{seed}'
            status=json.loads((p/'status.json').read_text())
            runs.append(dict(arm='D', agent=agent, seed=seed, status=status['status']))
            if status['status']!='complete': continue
            selected=[r for r in read(p/'model_selection.csv') if flag(r['selected']) and flag(r['eligible'])]
            assert len(selected)==1 and status['scientific_result']
            raw=read(p/'analysis/actual_lane_change_summary.csv')
            assert len(raw)==36
            for r in raw:
                observed=flag(r['actual_lane_change_observed'])
                episodes.append(dict(arm='D', agent=agent, seed=seed, exposure_id=r['exposure_id'],
                    observed=observed, seconds=float(r['first_actual_lane_change_seconds']) if observed else '',
                    collision=flag(r['collision_flag'])))
            for variant in VARIANTS:
                rows=read(p/f'controls/{variant}/analysis/actual_lane_change_summary.csv')
                assert len(rows)==36
                seconds=[float(r['first_actual_lane_change_seconds']) for r in rows if flag(r['actual_lane_change_observed'])]
                controls.append(dict(arm='D', agent=agent, seed=seed, control=variant,
                    n_observed=len(seconds), n_total=len(rows), seconds=statistics.median(seconds) if seconds else ''))
                if variant=='original':
                    assert {r['exposure_id']:r['first_actual_lane_change_seconds'] for r in raw}=={r['exposure_id']:r['first_actual_lane_change_seconds'] for r in rows}
    assert len(runs)==120 and len(episodes)==118*36 and len(controls)==118*4
    assert len({(r['arm'],r['agent'],r['seed'],r['exposure_id']) for r in episodes})==len(episodes)
    assert all(r['control'] in VARIANTS and r['n_total']==36 for r in controls)
    write('run_status.csv',runs);write('main_episodes.csv',episodes);write('controls_by_seed.csv',controls)
    provenance={'D_archive_sha256':hashlib.sha256((D/'figure_data.tar.gz').read_bytes()).hexdigest(),
        'A_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in A.glob('*.csv')},
        'D_endpoint':'first_actual_lane_change_seconds = (first_actual_lane_change_t + 1) / 5',
        'A_endpoint':'sustained deceleration response_latency_seconds; valid_onset',
        'D_failed':[r for r in runs if r['status']=='failed'],
        'scope':'Selected policies on development scenes; controls also used by checkpoint gates.'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Wrote',len(episodes),'main episodes and',len(controls),'control seed summaries')

if __name__=='__main__':main()
