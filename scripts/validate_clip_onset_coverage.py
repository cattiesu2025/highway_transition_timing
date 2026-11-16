#!/usr/bin/env python3
"""Check complete coverage accounting, windows, source manifests and P0 fidelity."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

from clip_onset_coverage import read_csv
from highway_transition_timing.summaries import context_window


def validate(root, legacy_root, allow_pending=False, outputs_root=Path('outputs')):
    inventory=read_csv(root/'inventory.csv')
    assert len(inventory)==240
    counts=Counter(r['status'] for r in inventory)
    if not allow_pending:
        assert counts=={'complete':238,'ineligible':2}, counts
    missing={(r['arm'],r['seed'],r['leader']) for r in inventory if r['status']=='ineligible'}
    assert missing <= {('D','4209','FD'),('D','4205','BAL')}, missing
    rows=read_csv(root/'all_selections.csv')
    if not allow_pending:
        candidate_groups=read_csv(root/'all_candidate_counts.csv')
        assert len(candidate_groups)==472
        assert len({tuple(r[k] for k in ('arm','seed','leader','follower')) for r in candidate_groups})==472
        assert sum(r['method'].startswith('highlights') for r in rows)==2380
        assert sum(r['method'].startswith('disagreements') for r in rows)==sum(int(r['selected']) for r in candidate_groups)
    keys=[tuple(r[k] for k in ('arm','seed','leader','follower','method','episode_id','step')) for r in rows]
    assert len(keys)==len(set(keys)), 'Duplicate selections'
    for r in rows:
        onset=int(r['onset_t']) if r['onset_t'] else None
        w=context_window(int(r['step']),int(r['episode_length']))
        assert (w.start,w.stop)==(int(r['window_start']),int(r['window_end_exclusive']))
        assert (r['onset_in_context_window']=='True')==(onset is not None and onset in w)
        if r['method'].startswith('disagreements'):
            assert (r['onset_in_native_window']=='True')==(onset is not None and int(r['native_start'])<=onset<=int(r['native_end_inclusive']))
    # Exact selection identities, rather than just coincident aggregate rates.
    legacy_sets={}
    for arm in 'ABC':
        legacy=read_csv(legacy_root/f'arm{arm}_selections.csv')
        expected={(r['seed'],r['agent_condition'],r['variant'],r['episode_id'],r['step']) for r in legacy}
        actual={(r['seed'],r['leader'],r['method'].removeprefix('highlights_'),r['episode_id'],r['step'])
                for r in rows if r['arm']==arm and r['method'].startswith('highlights')}
        assert actual <= expected, f'{arm}: HIGHLIGHTS selections differ from P0'
        if not allow_pending: assert actual==expected
        legacy_sets[arm]=len(actual)
    known_sources=[]
    for archive in sorted((root/'provenance').glob('source_v*.tar.gz')):
        with tarfile.open(archive) as f:
            known_sources.append({m.name:hashlib.sha256(f.extractfile(m).read()).hexdigest()
                                  for m in f.getmembers() if m.isfile()})
    assert known_sources, 'Missing source archives'
    ties=0
    completed=0
    for item in inventory:
        if item['status']!='complete': continue
        d=root/f"arm{item['arm']}_seed{item['seed']}_{item['leader']}"
        manifest=json.loads((d/'manifest.json').read_text())
        assert any(all(s.get(p)==sha for p,sha in manifest['sources'].items()) for s in known_sources), d
        outcome_key=next(p for p in manifest['inputs'] if p.endswith('/episode_outcomes.csv'))
        outcome_path=outputs_root/outcome_key
        if not outcome_path.exists() and item['arm']=='D':
            outcome_path=outputs_root/'arm_c_distance_figure_data'/outcome_key
        assert hashlib.sha256(outcome_path.read_bytes()).hexdigest()==manifest['inputs'][outcome_key], outcome_path
        target='slowdown_onset' if item['arm']=='A' else 'lane_change_onset'
        onsets={r['episode_id']:str(int(float(r['target_onset_t']))) for r in read_csv(outcome_path)
                if r['analysis_target']==target and r['episode_outcome']=='valid_onset'}
        audit=read_csv(d/'replay_audit.csv')
        assert len(audit)==36 and len({r['episode_id'] for r in audit})==36
        assert all(r['verified']=='True' and r['stored_steps']==r['replayed_steps'] for r in audit)
        ties+=sum(int(r['numerical_action_ties']) for r in audit)
        selected=read_csv(d/'selections.csv')
        assert all(r['onset_t']==onsets.get(r['episode_id'],'') for r in selected), d
        by_follower=Counter(r['follower'] for r in selected if r['method'].startswith('disagreements'))
        for c in read_csv(d/'candidate_counts.csv'):
            assert int(c['selected'])==min(5,int(c['candidates']))
            assert by_follower[c['follower']]==int(c['selected'])
        completed+=1
    result=dict(complete_leaders=completed,inventory=dict(counts),selections=len(rows),
                numerical_action_ties=ties,legacy_selection_identities_matched=legacy_sets,
                final=not allow_pending,checks='passed')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--legacy-root',type=Path,default=Path('outputs/summary_comparison/p0_selection_alignment'))
    p.add_argument('--allow-pending',action='store_true')
    p.add_argument('--outputs-root',type=Path,default=Path('outputs'))
    a=p.parse_args()
    validate(a.root,a.legacy_root,a.allow_pending,a.outputs_root)
