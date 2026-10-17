"""Exercise local staging lifecycle, capacity guard and node serialization."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stage():
    spec = importlib.util.spec_from_file_location('local_stage', ROOT/'scripts/katana_local_env.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    yield module
    for s, handler in original.items():
        signal.signal(s, handler)


def test_lock_waits_until_owner_releases(stage, tmp_path):
    owner = stage.acquire_lock(tmp_path, 0)
    with pytest.raises(TimeoutError):
        stage.acquire_lock(tmp_path, 0)
    owner.close()
    with stage.acquire_lock(tmp_path, 0):
        pass


def test_lock_rejects_symlink(stage, tmp_path):
    target = tmp_path/'outside'
    target.mkdir()
    (tmp_path/f'highway-env-lock-{os.getuid()}').symlink_to(target)
    with pytest.raises(RuntimeError, match='Unsafe'):
        stage.acquire_lock(tmp_path, 0)


def test_space_and_inode_guards(stage, tmp_path, monkeypatch):
    monkeypatch.setattr(stage.shutil, 'disk_usage', lambda root: SimpleNamespace(free=100))
    monkeypatch.setattr(stage.os, 'statvfs', lambda root: SimpleNamespace(f_files=1000, f_favail=10))
    with pytest.raises(RuntimeError, match='Insufficient'):
        stage.check_space(tmp_path, 101, 1)
    with pytest.raises(RuntimeError, match='Insufficient'):
        stage.check_space(tmp_path, 50, 11)
    stage.check_space(tmp_path, 50, 5)


@pytest.mark.parametrize('fail_child', [False, True])
def test_extract_without_archive_copy_and_cleanup(stage, tmp_path, monkeypatch, fail_child):
    bundle, local = tmp_path/'bundle', tmp_path/'local'
    bundle.mkdir(); local.mkdir()
    source = tmp_path/'pyvenv.cfg'
    source.write_text('test snapshot')
    with tarfile.open(bundle/'venv.tar', 'w') as tar:
        tar.add(source, arcname='pyvenv.cfg')
    (bundle/'venv.tar.sha256').write_text('checked by cluster sha256sum')
    real_run = stage.run_bounded
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] != 'sha256sum':
            return real_run(command, **kwargs)
    monkeypatch.setattr(stage, 'run_bounded', run)
    marker = tmp_path/'child-ok'
    script = '''import os,pathlib,sys
p=pathlib.Path(os.environ['VIRTUAL_ENV'])
assert (p/'pyvenv.cfg').read_text() == 'test snapshot'
assert not (p.parent/'venv.tar').exists()
assert os.environ['EXPECTED_LOCAL_VENV']==str(p)
assert os.environ['PATH'].startswith(str(p/'bin')+':')
assert 'PYTHONHOME' not in os.environ
pathlib.Path(sys.argv[1]).write_text(str(p))
sys.exit(int(sys.argv[2]))
'''
    args = ['--bundle',str(bundle),'--local-root',str(local),'--',sys.executable,
            '-c',script,str(marker),str(int(fail_child))]
    if fail_child:
        with pytest.raises(subprocess.CalledProcessError):
            stage.main(args)
    else:
        stage.main(args)
    assert marker.exists()
    assert not Path(marker.read_text()).exists()
    assert not list(local.glob('highway-env-job.*'))
    assert calls[0][0]=='sha256sum' and calls[1][0]=='tar'
    with stage.acquire_lock(local, 0):
        pass


def test_capacity_failure_never_extracts_or_runs_child(stage, tmp_path, monkeypatch):
    (tmp_path/'venv.tar').touch()
    (tmp_path/'venv.tar.sha256').touch()
    calls=[]
    monkeypatch.setattr(stage,'run_bounded',lambda *a,**kw: calls.append(a[0]))
    monkeypatch.setattr(stage,'required_space',lambda p:(10**30,1))
    with pytest.raises(RuntimeError,match='Insufficient'):
        stage.main(['--bundle',str(tmp_path),'--local-root',str(tmp_path),'--space-timeout','0','--','never-run'])
    assert len(calls)==1 and calls[0][0]=='sha256sum'
    assert not list(tmp_path.glob('highway-env-job.*'))


def test_child_timeout_reaped(stage):
    with pytest.raises(subprocess.TimeoutExpired):
        stage.run_bounded([sys.executable,'-c','import time; time.sleep(30)'], timeout=.05)


def test_recovery_maps_only_missing_indices():
    script=(ROOT/'scripts/katana_reward_fixed_sum_armB_recover.pbs').read_text()
    indices=[int(i) for i in script.split('missing=(',1)[1].split(')',1)[0].split()]
    completed={2,3,12,13,16,17,18,21,22,23,24,25,26}
    assert len(indices)==22 and set(indices)==set(range(35))-completed
    assert '#PBS -J 0-21' in script


@pytest.mark.parametrize('recover', [True, False])
def test_space_wait_rechecks_capacity_and_obeys_deadline(stage, tmp_path, monkeypatch, capsys, recover):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(stage.time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(stage.time, 'sleep', sleep)
    monkeypatch.setattr(stage.shutil, 'disk_usage', lambda p: SimpleNamespace(
        free=200 if recover and clock[0] >= 60 else 10))
    monkeypatch.setattr(stage.os, 'statvfs', lambda p: SimpleNamespace(f_files=1000, f_favail=100))
    if recover:
        stage.wait_for_space(tmp_path, 100, 10, timeout=90, interval=60)
        assert sleeps == [60]
    else:
        with pytest.raises(stage.InsufficientSpace, match='bounded wait'):
            stage.wait_for_space(tmp_path, 100, 10, timeout=90, interval=60)
        assert sleeps == [60, 30]
    output = capsys.readouterr().out
    assert 'node=' in output and 'available=' in output and 'free inodes=' in output
    assert 'remaining=90.0s' in output
    assert ('Space ready' if recover else 'Space wait timed out') in output


def test_space_wait_handles_inode_recovery(stage, tmp_path, monkeypatch):
    clock = [0]
    monkeypatch.setattr(stage.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(stage.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(stage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1000))
    monkeypatch.setattr(stage.os, 'statvfs', lambda p: SimpleNamespace(f_files=1000, f_favail=1 if clock[0]==0 else 100))
    stage.wait_for_space(tmp_path, 100, 10, timeout=120, interval=60)
    assert clock[0] == 60


def test_space_wait_does_not_hide_filesystem_errors(stage, tmp_path, monkeypatch):
    def broken(path):
        raise PermissionError('cannot inspect filesystem')
    monkeypatch.setattr(stage.shutil, 'disk_usage', broken)
    with pytest.raises(PermissionError):
        stage.wait_for_space(tmp_path, 1, 1)


@pytest.mark.parametrize('option,value', [('--space-timeout','-1'),('--space-timeout','nan'),
    ('--space-check-interval','0'),('--space-check-interval','inf'),('--lock-timeout','nan')])
def test_wait_arguments_reject_invalid_values(stage, tmp_path, option, value):
    with pytest.raises(SystemExit):
        stage.main(['--bundle',str(tmp_path),option,value,'--','never-run'])


def make_shared_bundle(tmp_path):
    import hashlib
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    source = tmp_path/'pyvenv.cfg'
    source.write_text('shared test venv')
    with tarfile.open(bundle/'venv.tar', 'w') as archive:
        archive.add(source, arcname='pyvenv.cfg')
    digest = hashlib.sha256((bundle/'venv.tar').read_bytes()).hexdigest()
    (bundle/'venv.tar.sha256').write_text(f'{digest}  venv.tar\n')
    return bundle, digest


def test_two_shared_jobs_overlap_and_keep_cache(tmp_path):
    import json
    bundle, digest = make_shared_bundle(tmp_path)
    local = tmp_path/'local'; local.mkdir()
    # macOS lacks sha256sum; provide an actual checksum verifier for this test.
    bins = tmp_path/'bin'; bins.mkdir()
    verifier = bins/'sha256sum'
    verifier.write_text(f'#!{sys.executable}\nimport hashlib,pathlib\n'
        'expected=pathlib.Path("venv.tar.sha256").read_text().split()[0]\n'
        'assert hashlib.sha256(pathlib.Path("venv.tar").read_bytes()).hexdigest()==expected\n')
    verifier.chmod(0o755)
    code = '''import os,pathlib,sys,time,json
root=pathlib.Path(sys.argv[1]); mine=sys.argv[2]; other=sys.argv[3]
assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
(root/mine).write_text(json.dumps({'venv':os.environ['VIRTUAL_ENV'],'tmp':os.environ['TMPDIR']}))
deadline=time.monotonic()+15
while not (root/other).exists():
    assert time.monotonic()<deadline, 'other child never ran concurrently'
    time.sleep(.02)
assert pathlib.Path(os.environ['VIRTUAL_ENV'],'pyvenv.cfg').exists()
'''
    env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'])
    processes=[]
    try:
        for mine,other in [('one','two'),('two','one')]:
            processes.append(subprocess.Popen([sys.executable,'-S',str(ROOT/'scripts/katana_local_env.py'),
                '--shared-env','--bundle',str(bundle),'--local-root',str(local),'--lock-timeout','5',
                '--',sys.executable,'-c',code,str(tmp_path),mine,other],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True))
        logs=[]
        for process in processes:
            output,_=process.communicate(timeout=25)
            assert process.returncode==0,output
            logs.append(output)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill();process.wait()
    a=json.loads((tmp_path/'one').read_text());b=json.loads((tmp_path/'two').read_text())
    assert a['venv']==b['venv'] and a['tmp']!=b['tmp']
    assert Path(a['venv']).exists() and not Path(a['tmp']).exists() and not Path(b['tmp']).exists()
    assert sum('cache miss' in log for log in logs)==1
    assert sum('cache hit' in log for log in logs)==1
    assert not list(local.glob('highway-shared-env-*/.staging-*'))


def test_shared_extraction_failure_never_publishes_cache(stage,tmp_path,monkeypatch):
    bundle,digest=make_shared_bundle(tmp_path)
    local=tmp_path/'local';local.mkdir()
    def run(command,**kw):
        if command[0]=='tar':raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(stage,'run_bounded',run)
    with pytest.raises(subprocess.CalledProcessError):
        stage.main(['--shared-env','--bundle',str(bundle),'--local-root',str(local),'--','never-run'])
    assert not (local/f'highway-shared-env-{os.getuid()}'/digest).exists()
    assert not list(local.glob('highway-shared-env-*/.staging-*'))
    with stage.acquire_lock(local,0):pass


def test_shared_child_failure_keeps_valid_cache(stage,tmp_path,monkeypatch):
    bundle,digest=make_shared_bundle(tmp_path)
    local=tmp_path/'local';local.mkdir()
    real=stage.run_bounded
    def run(command,**kw):
        if command[0]!='sha256sum':return real(command,**kw)
    monkeypatch.setattr(stage,'run_bounded',run)
    with pytest.raises(subprocess.CalledProcessError):
        stage.main(['--shared-env','--bundle',str(bundle),'--local-root',str(local),'--',
                    sys.executable,'-c','raise SystemExit(3)'])
    assert (local/f'highway-shared-env-{os.getuid()}'/digest/'ready.json').exists()
    assert not list(local.glob('highway-shared-job.*'))
    with stage.acquire_lock(local,0):pass
