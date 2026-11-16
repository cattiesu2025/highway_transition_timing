#!/usr/bin/env python3
"""Aggregate only audited, complete per-leader coverage runs; show all missing cells."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
from clip_onset_coverage import AGENTS, write_csv as write_nonempty_csv, read_csv


def write_csv(path, rows):
    if rows:
        write_nonempty_csv(path, rows)
    else:
        # A later audit may exclude every row; never leave an old result table.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')


def aggregate(root):
    rows, inventory, counts = [], [], []
    for arm, base in zip('ABCD', (3100, 4100, 4200, 4200)):
        for seed in range(base, base+20):
            for leader in AGENTS:
                directory = root / f'arm{arm}_seed{seed}_{leader}'
                status_path = directory / 'status.json'
                status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'pending'}
                state = status['status']
                if state == 'complete':
                    if status['partial'] or status['exposures'] != 36:
                        state = 'partial'
                    else:
                        audit = read_csv(directory / 'replay_audit.csv')
                        if len(audit) != 36 or any(r['verified'] != 'True' for r in audit):
                            raise ValueError(f'Invalid audit: {directory}')
                        rows += read_csv(directory / 'selections.csv')
                        counts += read_csv(directory / 'candidate_counts.csv')
                if state == 'pending' and (directory / 'highlights.csv').exists():
                    # HIGHLIGHTS is an independent stored-table calculation.
                    rows += read_csv(directory / 'highlights.csv')
                inventory.append(dict(arm=arm, seed=seed, leader=leader, status=state))
    write_csv(root / 'inventory.csv', inventory)
    write_csv(root / 'all_selections.csv', rows)
    write_csv(root / 'all_candidate_counts.csv', counts)
    groups = defaultdict(list)
    for r in rows:
        groups[r['arm'], r['method'], r['leader'], r['follower']].append(r)
        groups[r['arm'], r['method'], 'ALL', 'ALL'].append(r)
    for r in counts:
        groups[r['arm'], 'disagreements_last_state', r['leader'], r['follower']]
        groups[r['arm'], 'disagreements_last_state', 'ALL', 'ALL']
    summary = []
    for (arm, method, leader, follower), group in sorted(groups.items()):
        hits = sum(r['onset_in_context_window'] == 'True' for r in group)
        native = sum(r['onset_in_native_window'] == 'True' for r in group)
        support = [r for r in counts if r['arm']==arm and
                   (leader=='ALL' or (r['leader']==leader and r['follower']==follower))] if method.startswith('disagreements') else group
        summary.append(dict(arm=arm, method=method, leader=leader, follower=follower,
            seeds=len({r['seed'] for r in support}),
            selection_groups=len({(r['seed'], r['leader'], r['follower']) for r in support}),
            selected=len(group), covered=hits, coverage=hits/len(group) if group else '',
            no_onset=sum(r['onset_t'] == '' for r in group),
            native_covered=native if method.startswith('disagreements') else '',
            native_coverage=native/len(group) if group and method.startswith('disagreements') else ''))
    write_csv(root / 'summary.csv', summary)
    # Paired HIGHLIGHTS rows repeat once per actually available directed pair,
    # giving exactly the same seed/leader eligibility support as DISAGREEMENTS.
    hs = defaultdict(list)
    ds = defaultdict(list)
    for r in rows:
        if r['method'].startswith('highlights'):
            hs[r['arm'], r['seed'], r['leader'], r['method']].append(r)
        else:
            ds[r['arm'], r['seed'], r['leader'], r['follower']].append(r)
    paired = []
    for count in counts:
        arm, seed, leader, follower = (count[k] for k in ('arm', 'seed', 'leader', 'follower'))
        dr = ds[arm, seed, leader, follower]
        for variant in ('max_min', 'max_second'):
            hr = hs[arm, seed, leader, 'highlights_'+variant]
            if len(hr) != 5:
                raise ValueError('Missing HIGHLIGHTS paired selection')
            paired.append(dict(arm=arm, seed=seed, leader=leader, follower=follower,
                variant=variant, highlights_selected=len(hr),
                highlights_covered=sum(r['onset_in_context_window']=='True' for r in hr),
                disagreements_selected=len(dr),
                disagreements_covered=sum(r['onset_in_context_window']=='True' for r in dr)))
    write_csv(root / 'paired_by_seed.csv', paired)
    paired_summary=[]
    for arm in 'ABCD':
        for variant in ('max_min','max_second'):
            group=[r for r in paired if r['arm']==arm and r['variant']==variant]
            if not group:
                continue
            nh=sum(r['highlights_selected'] for r in group)
            nd=sum(r['disagreements_selected'] for r in group)
            hh=sum(r['highlights_covered'] for r in group)
            hd=sum(r['disagreements_covered'] for r in group)
            paired_summary.append(dict(arm=arm,variant=variant,directed_pairs=len(group),
                highlights_selected=nh,highlights_covered=hh,highlights_coverage=hh/nh,
                disagreements_selected=nd,disagreements_covered=hd,
                disagreements_coverage=hd/nd if nd else '',
                zero_candidate_pairs=sum(r['disagreements_selected']==0 for r in group)))
    write_csv(root / 'paired_summary.csv', paired_summary)
    complete=sum(r['status']=='complete' for r in inventory)
    ineligible=sum(r['status']=='ineligible' for r in inventory)
    state='complete' if complete==238 and ineligible==2 else 'PROVISIONAL'
    lines = ['# Clip/onset coverage', '',
        f'Status: {state}; {complete}/238 eligible leaders completed, {ineligible}/2 ineligible models accounted for.', '',
        'Descriptive development diagnostic; not a ranking of explanation quality.', '',
        '| Arm | HIGHLIGHTS max-min | HIGHLIGHTS max-second | DISAGREEMENTS |',
        '| --- | --- | --- | --- |']
    lookup = {(r['arm'], r['method']):r for r in summary if r['leader']=='ALL'}
    for arm in 'ABCD':
        values=[]
        for method in ('highlights_max_min','highlights_max_second','disagreements_last_state'):
            r=lookup.get((arm,method))
            values.append((f"{r['covered']}/{r['selected']} ({100*r['coverage']:.1f}%)" if r['selected'] else '0/0 (N/A)') if r else 'pending')
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines += ['', 'Inventory:']
    for arm in 'ABCD':
        states=defaultdict(int)
        for r in inventory:
            if r['arm']==arm: states[r['status']]+=1
        lines.append(f'- {arm}: {dict(states)}')
    lines += ['', '## Matched seed/leader/pair support', '',
              '| Arm | HIGHLIGHTS variant | Directed pairs | HIGHLIGHTS | DISAGREEMENTS |',
              '| --- | --- | --- | --- | --- |']
    for r in paired_summary:
        d=f"{100*r['disagreements_coverage']:.1f}%" if r['disagreements_selected'] else 'N/A'
        lines.append(f"| {r['arm']} | {r['variant']} | {r['directed_pairs']} | {100*r['highlights_coverage']:.1f}% | {d} |")
    lines += ['', '## DISAGREEMENTS native-window sensitivity', '',
              '| Arm | Common window | Native window |', '| --- | --- | --- |']
    for r in summary:
        if r['method']=='disagreements_last_state' and r['leader']=='ALL':
            common=f"{100*r['coverage']:.1f}%" if r['selected'] else 'N/A'
            native=f"{100*r['native_coverage']:.1f}%" if r['selected'] else 'N/A'
            lines.append(f"| {r['arm']} | {common} | {native} |")
    lines += ['', 'Each HIGHLIGHTS selection group is one seed/leader (budget 5). Each DISAGREEMENTS group is one seed/leader/follower (budget 5); both directions are retained. Denominators are selected clips, including episodes without an onset. No-candidate and ineligible groups are not assigned fabricated clips; consult inventory and candidate counts.', '',
        'Endpoint: stored target_onset_t. A uses persistent slowdown onset. B/C/D use the direction-matching lane-change command onset confirmed by a realised lane-index change; this is not the physical crossing/completion timestamp. Step-zero onsets are retained.', '',
        'The common window is the historical P0 range [max(t-10,0), min(t+10,L-1)), including its right-open terminal clamp. DISAGREEMENTS uses its native window for selection/ranking and reports native-window coverage separately. Only the real leader trajectory onset is tested; no leader onset is assigned to a synthetic follower branch.', '',
        'Arm D = C distance-affine-v1 with F1/F4/F7. Original A/B/C checkpoints and settings are unchanged. Arm D is not a geometry-only causal comparison against these historical A/B/C reward settings.', '',
        'Any pending/partial inventory means this table is provisional. paired_by_seed.csv restricts HIGHLIGHTS to the same available directed policy pairs as DISAGREEMENTS.']
    (root / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    aggregate(parser.parse_args().root)
