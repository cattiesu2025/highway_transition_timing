"""Symmetric follow-up preserves the original sweep and pairs actual endpoints."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'experiments/reward_sensitivity'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    run = load('symmetric_test_run', 'run.py')
    monkeypatch.setitem(sys.modules, 'run', run)
    supplement = load('symmetric_test_supplement', 'run_symmetric.py')
    monkeypatch.setitem(sys.modules, 'run_symmetric', supplement)
    collector = load('symmetric_test_collect', 'collect.py')
    monkeypatch.setitem(sys.modules, 'collect', collector)
    paired = load('symmetric_test_paired', 'collect_symmetric.py')
    return run, supplement, paired


def test_frozen_manifest_and_supplement_commands(modules, tmp_path):
    run, supplement, paired = modules
    folder = ROOT/'experiments/reward_sensitivity'
    assert run.jobs() == json.loads((folder/'jobs.json').read_text())
    assert supplement.jobs() == json.loads((folder/'jobs_symmetric.json').read_text())
    jobs = supplement.jobs()
    assert len(jobs) == 20
    assert [(j['arm'], j['variant'], j['seed']) for j in jobs[::5]] == [
        ('A', 'R7', 3100), ('A', 'R8', 3100), ('C', 'R7', 4200), ('C', 'R8', 4200)]
    assert len({run.run_directory(tmp_path, j) for j in jobs}) == 20
    assert not ({run.run_directory(tmp_path, j) for j in jobs} &
                {run.run_directory(tmp_path, j) for j in run.jobs()})
    assert len(paired.tasks(tmp_path)) == 40
    for job in jobs:
        train = run.commands(job, tmp_path, 'train')[0]
        assert ('--target-lane-vehicle' in train) == (job['arm'] == 'C')
        assert train[train.index('--timesteps')+1] == '100000'
        assert train[train.index('--duration')+1] == '20'
        for phase in ('train', 'evaluate'):
            for cmd in run.commands(job, tmp_path, phase):
                assert not any('heldout' in x for x in cmd)
                if phase == 'train' or job['arm'] == 'A':
                    assert float(cmd[cmd.index('--speed-weight')+1]) == job['speed_weight']
                    assert float(cmd[cmd.index('--front-distance-weight')+1]) == job['front_distance_weight']
    run.main(['--manifest', str(tmp_path/'manifest.json')], task_list=jobs)
    assert json.loads((tmp_path/'manifest.json').read_text()) == jobs
    with pytest.raises(SystemExit):
        run.main(['--index', '20'], task_list=jobs)


def test_supplement_execution_and_protection(modules, tmp_path, monkeypatch):
    run, supplement, _ = modules
    jobs = supplement.jobs()
    calls = []
    monkeypatch.setattr(run.subprocess, 'run', lambda cmd, **kwargs: calls.append(cmd))
    args = ['--index', '19', '--out-root', str(tmp_path), '--execute']
    run.main(args, task_list=jobs)
    directory = run.run_directory(tmp_path, jobs[19])
    assert json.loads((directory/'sweep_job.json').read_text()) == jobs[19]
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        run.main(args, task_list=jobs)
    with pytest.raises(FileNotFoundError):
        run.main(args + ['--phase', 'evaluate'], task_list=jobs)
    (directory/'models').mkdir()
    (directory/'models/SP_main.zip').touch()
    run.main(args + ['--phase', 'evaluate'], task_list=jobs)
    assert len(calls) == 2
    (directory/'sweep_job.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        run.main(args + ['--phase', 'evaluate'], task_list=jobs)


def make_outcomes():
    return [dict(arm='C', variant=variant, agent=agent, seed=4200,
                 episode_id=f'{agent}_M{i:04d}_r0', episode_outcome='valid_onset',
                 collision=False, response_latency_seconds=latency)
            for variant, agent, latency in [('R7', 'FD', 9.), ('SP', 'SP', 6.)]
            for i in range(36)]


def test_paired_gaps_preserve_failure_and_collision(modules):
    _, _, paired = modules
    outcomes = make_outcomes()
    outcomes[0].update(episode_outcome='terminal_failure', collision=True, response_latency_seconds='')
    outcomes[1].update(episode_outcome='no_onset_censored', response_latency_seconds='')
    outcomes[2]['collision'] = True  # An observed onset may precede a later collision.
    ep, seeds, statuses = paired.paired_tables(list(reversed(outcomes)))
    assert len(ep) == 36 and len(seeds) == 1 and len(statuses) == 20
    assert seeds[0]['conditional_median_gap_seconds'] == -3
    assert seeds[0]['n_valid_pairs'] == seeds[0]['n_negative'] == 34
    assert seeds[0]['n_excluded_pairs'] == 2
    assert ep[0]['gap_seconds'] == '' and ep[0]['fd_collision'] is True
    assert ep[2]['gap_seconds'] == -3 and ep[2]['fd_collision'] is True
    assert sum(r['status'] == 'missing_condition' for r in statuses) == 19
    with pytest.raises(ValueError, match='Duplicate'):
        paired.paired_tables(outcomes + [outcomes[-1]])
    with pytest.raises(ValueError, match='scene IDs'):
        paired.paired_tables(outcomes[:-1])


def test_exposure_mismatch_is_rejected(modules, tmp_path):
    _, _, paired = modules
    task_list = []
    for variant, agent in [('R7', 'FD'), ('SP', 'SP')]:
        directory = tmp_path/variant
        (directory/'analysis').mkdir(parents=True)
        records = ['episode_id,agent_condition,exposure_id,exposure_seed,rollout_id,rollout_seed']
        records += [f'{agent}_M{i:04d}_r0,{agent},M{i:04d},{4200+i},r0,0' for i in range(36)]
        (directory/'analysis/episode_outcomes.csv').write_text('\n'.join(records)+'\n')
        task_list.append((dict(arm='C', variant=variant, agent=agent, seed=4200), directory))
    paired.verify_exposure_pairing(task_list, make_outcomes())
    file = tmp_path/'SP/analysis/episode_outcomes.csv'
    file.write_text(file.read_text().replace(',4200,', ',9999,'))
    with pytest.raises(ValueError, match='metadata differs'):
        paired.verify_exposure_pairing(task_list, make_outcomes())
