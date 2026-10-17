import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/reward_sensitivity'


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(FOLDER))
    for name in ('run','run_fixed_sum','run_selected_validation'):
        spec=importlib.util.spec_from_file_location(name,FOLDER/f'{name}.py')
        module=importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules,name,module)
        spec.loader.exec_module(module)
    return module


def test_frozen_design_and_original_protocol(validation,tmp_path):
    v=validation; jobs=v.jobs()
    assert jobs==json.loads(v.MANIFEST.read_text())
    assert len(jobs)==180 and sum(j['reuse'] for j in jobs)==45
    assert {j['variant'] for j in jobs}=={'F1','F4','F7'}
    for j in jobs:
        train,controls=v.commands(tmp_path,j,'complete')
        assert len(train)==int(not j['reuse']) and len(controls)==1
        assert 'heldout' not in ' '.join(controls[0])
        assert all(name in controls[0] for name in ('original','no-front','matched-speed-front','far-front'))
        if train:
            cmd=train[0]
            assert cmd[cmd.index('--timesteps')+1]=='100000'
            assert ('--target-lane-vehicle' in cmd)==(j['arm']=='C')
            assert float(cmd[cmd.index('--speed-weight')+1])==j['speed_weight']
        assert not v.commands(tmp_path,j,'evaluate')[0]
    for arm,base in v.BASE_SEEDS.items():
        for variant in v.VARIANTS:
            assert {j['seed'] for j in jobs if j['arm']==arm and j['variant']==variant}==set(range(base,base+20))


def test_dry_run_and_existing_folder_protection(validation,tmp_path,monkeypatch):
    v=validation; calls=[]
    monkeypatch.setattr(v.subprocess,'run',lambda *a,**kw:calls.append(a))
    v.main(['--index','5','--out-root',str(tmp_path)])
    assert not calls and not list(tmp_path.iterdir())
    v.directory(tmp_path,v.jobs()[5]).mkdir()
    with pytest.raises(FileExistsError):
        v.main(['--index','5','--out-root',str(tmp_path),'--execute'])
    assert not calls


def test_reused_models_never_trigger_training(validation,tmp_path,monkeypatch):
    v=validation; calls=[]
    monkeypatch.setattr(v.subprocess,'run',lambda *a,**kw:calls.append(a))
    with pytest.raises(FileNotFoundError):
        v.main(['--index','0','--out-root',str(tmp_path),'--execute'])
    assert not calls and not list(tmp_path.iterdir())


def test_recovery_mapping_and_shared_pbs(validation):
    plan=json.loads((FOLDER/'recovery_9543211.json').read_text())
    entries=plan['recovery_tasks']
    indices=[r['index'] for r in entries]
    assert len(indices)==len(set(indices))==53
    assert sum(r['reuse'] for r in entries)==14
    assert sum(r['reason']=='lock_timeout' for r in entries)==52
    assert sum(r['reason']=='space_wait_timeout' for r in entries)==1
    for r in entries:assert validation.jobs()[r['index']]['reuse']==r['reuse']
    script=(ROOT/'scripts/katana_selected_validation_recover.pbs').read_text()
    assert [int(x) for x in script.split('missing=(',1)[1].split(')',1)[0].split()]==indices
    assert '#PBS -J 0-52' in script
    for name in ('katana_selected_validation.pbs','katana_selected_validation_retry.pbs'):
        assert '--shared-env' in (ROOT/'scripts'/name).read_text()
