"""Coverage denominator, branch ranking and upstream selection edge cases."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('clip_coverage', Path(__file__).resolve().parents[1] / 'scripts/clip_onset_coverage.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(t, score, episode='e'):
    return m.coverage_row('A', 3100, 'FD', 'SP', 'disagreements_last_state',
        episode, t, 600, 100, score, max(0, t-9), t+10)


def test_common_window_and_no_onset_denominator():
    r = m.coverage_row('D', 4200, 'FD', '', 'highlights_max_second', 'e', 100, 600, None, 1)
    assert r['onset_t'] == '' and not r['onset_in_context_window']
    assert row(110, 1)['onset_in_context_window']
    assert not row(110, 1)['onset_in_native_window']
    assert not row(90, 1)['onset_in_context_window']
    assert row(90, 1)['onset_in_native_window']


def test_selection_union_overlap_and_fallback():
    candidates = [row(100, 9), row(101, 8), row(115, 7), row(200, 6)]
    assert [r['step'] for r in m.select_disagreements(candidates, budget=3)] == [100, 115, 200]
    assert [r['step'] for r in m.select_disagreements(candidates, budget=5)] == [100, 115, 200, 101]
    assert len(m.select_disagreements([row(100, 1), row(100, 1, 'other')])) == 2


def test_last_state_uses_both_policies_and_separate_scales():
    assert m.last_state_importance([[3, 5], [10, 30]], [[1, 3], [5, 10]], [0, 0], [10, 100]) == pytest.approx(.4)
    assert m.last_state_importance([[3, 5], [10, 30]], [[1, 3], [5, 10]], [0, 0], [10, 100], True) == 0
    with pytest.raises(ValueError):
        m.last_state_importance([[1, 1]], [[1, 1]], [1], [1])


def test_failed_distance_gate_never_uses_provisional_model(tmp_path):
    import json
    (tmp_path/'status.json').write_text(json.dumps({'status':'failed','error':"No checkpoint passed the eligibility gate for: ['FD']. See model_selection.csv."}))
    m.write_csv(tmp_path/'model_selection.csv',[dict(agent_condition='FD',eligible=False,selected=False)])
    (tmp_path/'models').mkdir()
    (tmp_path/'models/FD_main.zip').write_bytes(b'provisional')
    assert not m.eligible(tmp_path, 'D', 'FD')


def test_aggregate_retains_empty_disagreement_group_and_rejects_partial(tmp_path):
    import json
    from aggregate_clip_onset_coverage import aggregate
    d=tmp_path/'armA_seed3100_FD'
    d.mkdir()
    (d/'status.json').write_text(json.dumps(dict(status='complete',partial=False,exposures=36)))
    m.write_csv(d/'replay_audit.csv',[dict(episode_id=str(i),verified=True) for i in range(36)])
    selected=[]
    for variant in ('max_min','max_second'):
        for t in range(5):
            selected.append(m.coverage_row('A',3100,'FD','', 'highlights_'+variant,'e',t,600,None,1))
    m.write_csv(d/'selections.csv',selected)
    m.write_csv(d/'candidate_counts.csv',[dict(arm='A',seed=3100,leader='FD',follower='SP',candidates=0,selected=0)])
    aggregate(tmp_path)
    summary=m.read_csv(tmp_path/'summary.csv')
    r=next(r for r in summary if r['method']=='disagreements_last_state' and r['leader']=='ALL')
    assert r['selected']=='0' and r['coverage']=='' and r['selection_groups']=='1'
    paired=m.read_csv(tmp_path/'paired_by_seed.csv')
    assert len(paired)==2 and all(r['disagreements_selected']=='0' for r in paired)
    assert all(r['no_onset']=='5' for r in summary if r['method'].startswith('highlights') and r['leader']=='ALL')
    (d/'status.json').write_text(json.dumps(dict(status='complete',partial=True,exposures=1)))
    aggregate(tmp_path)
    assert m.read_csv(tmp_path/'inventory.csv')[0]['status']=='partial'
    assert not m.read_csv(tmp_path/'summary.csv')
