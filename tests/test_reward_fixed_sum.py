"""Fixed-sum wiring, reuse, data pairing, and overwrite protection."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'experiments/reward_sensitivity'


@pytest.fixture
def modules(monkeypatch):
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, FOLDER / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    sweep = load('run', 'run.py')
    load('collect', 'collect.py')
    load('run_symmetric', 'run_symmetric.py')
    pairs = load('collect_symmetric', 'collect_symmetric.py')
    fixed = load('run_fixed_sum', 'run_fixed_sum.py')
    collector = load('collect_fixed_sum', 'collect_fixed_sum.py')
    return sweep, fixed, collector, pairs


def test_design_mapping_reuse_and_commands(modules, tmp_path):
    sweep, fixed, collector, _ = modules
    assert fixed.jobs() == json.loads(fixed.MANIFEST.read_text())
    assert len(fixed.jobs()) == 70 and len(fixed.conditions()) == 90
    assert [j['variant'] for j in fixed.jobs()[::5]] == ['F1','F2','F4','F6','F7','F8','F9'] * 2
    assert fixed.jobs()[34]['seed'] == 3104
    assert fixed.jobs()[35]['arm'] == 'C' and fixed.jobs()[35]['seed'] == 4200
    assert all(sum(pair) == 140 for pair in fixed.COEFFICIENTS.values())
    assert list(fixed.COEFFICIENTS.values()) == [tuple(reversed(pair)) for pair in reversed(list(fixed.COEFFICIENTS.values()))]
    original_paths = {sweep.run_directory(tmp_path, j) for j in sweep.jobs()}
    for job in fixed.jobs():
        assert fixed.run_directory(tmp_path, job) not in original_paths
        calls = fixed.commands(job, tmp_path, 'train')
        command = calls[0]
        assert command[command.index('--out') + 1] == str(fixed.run_directory(tmp_path, job))
        assert command[command.index('--agents') + 1] == job['agent']
        assert float(command[command.index('--speed-weight') + 1]) == job['speed_weight']
        assert float(command[command.index('--front-distance-weight') + 1]) == job['front_distance_weight']
        assert command[command.index('--timesteps') + 1] == '100000'
        assert ('--target-lane-vehicle' in command) == (job['arm'] == 'C')
        for cmd in fixed.commands(job, tmp_path, 'evaluate'):
            assert str(fixed.run_directory(tmp_path, job)) in cmd
            assert not any('heldout' in value for value in cmd)
    sources = collector.tasks(tmp_path)
    assert len({path for _, path in sources}) == 90
    for job, path in sources:
        if job['variant'] == 'F3':
            assert '_R2_' in path.name and job['agent'] == 'FD'
        elif job['variant'] == 'F5':
            assert path.name.startswith(fixed.PREFIX[job['arm']]) and job['agent'] == 'BAL'
        else:
            assert path == fixed.run_directory(tmp_path, job)
    with pytest.raises(ValueError, match='Reused'):
        fixed.commands(fixed.condition('A', 'F3', 3100), tmp_path, 'train')


def test_execute_guards_and_dry_run(modules, monkeypatch, tmp_path):
    _, fixed, _, _ = modules
    calls = []
    monkeypatch.setattr(fixed.subprocess, 'run', lambda call, **kwargs: calls.append(call))
    args = ['--index', '69', '--out-root', str(tmp_path)]
    fixed.main(args)
    assert not list(tmp_path.iterdir()) and not calls
    with pytest.raises(SystemExit):
        fixed.main(['--index', '70'])
    fixed.main(args + ['--execute'])
    assert len(calls) == 1
    directory = fixed.run_directory(tmp_path, fixed.jobs()[69])
    assert json.loads((directory / 'sweep_job.json').read_text()) == fixed.jobs()[69]
    with pytest.raises(FileExistsError):
        fixed.main(args + ['--execute'])
    with pytest.raises(FileNotFoundError, match='Selected model'):
        fixed.main(args + ['--execute', '--phase', 'evaluate'])
    (directory / 'models').mkdir()
    (directory / 'models/SP_main.zip').touch()
    fixed.main(args + ['--execute', '--phase', 'evaluate'])
    assert len(calls) == 2
    (directory / 'sweep_job.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        fixed.main(args + ['--execute', '--phase', 'evaluate'])
    bad = tmp_path / 'bad_manifest.json'
    bad.write_text('[]')
    monkeypatch.setattr(fixed, 'MANIFEST', bad)
    with pytest.raises(ValueError, match='Frozen'):
        fixed.main(['--index', '0', '--out-root', str(tmp_path), '--execute'])
    assert not fixed.run_directory(tmp_path, fixed.jobs()[0]).exists()


def test_fixed_labels_check_training_prefix(modules, tmp_path):
    _, fixed, collector, _ = modules
    job = fixed.jobs()[0]
    directory = fixed.run_directory(tmp_path, job)
    baseline = tmp_path / f"{fixed.PREFIX['A']}_100k_seed3100"
    for folder in [directory, baseline]:
        folder.mkdir()
        (folder / 'training_scenarios.csv').write_text('agent_condition,reset_index,ego_speed\nFD,0,28\n')
    collector.verify_training(tmp_path, [(job, directory)])
    (directory / 'training_scenarios.csv').write_text('agent_condition,reset_index,ego_speed\nFD,0,29\n')
    with pytest.raises(ValueError, match='prefix differs'):
        collector.verify_training(tmp_path, [(job, directory)])


def test_mirror_gaps_keep_missing_onsets_and_collisions(modules):
    _, fixed, collector, pairs = modules
    outcomes = [dict(arm='C', variant=v, agent='BAL', seed=4200, episode_id=f'BAL_M{i:04d}_r0',
                     episode_outcome='valid_onset', collision=False, response_latency_seconds=latency)
                for v, latency in [('F4', 8), ('F6', 6)] for i in range(36)]
    outcomes[0].update(episode_outcome='no_onset_censored', response_latency_seconds='')
    outcomes[1].update(collision=True)
    ep, seeds, status = pairs.paired_tables(outcomes, pairs=collector.PAIRS, base_seeds=fixed.BASE_SEEDS)
    assert len(status) == 40 and len(seeds) == 1
    assert seeds[0]['conditional_median_gap_seconds'] == -2
    assert seeds[0]['n_valid_pairs'] == 35 and seeds[0]['n_excluded_pairs'] == 1
    assert ep[0]['gap_seconds'] == '' and ep[1]['fd_collision'] is True


def test_all_configuration_exposure_pairing(modules, tmp_path):
    _, fixed, collector, _ = modules
    tasks, outcomes = [], []
    # F3 and F5 are NOT a mirror pair: they must still share exposure metadata.
    for variant in ['F3', 'F5']:
        job = fixed.condition('A', variant, 3100)
        d = fixed.source_directory(tmp_path, job)
        (d / 'analysis').mkdir(parents=True)
        (d / 'models').mkdir()
        (d / 'models' / f"{job['agent']}_main.zip").touch()
        (d / 'training_scenarios.csv').touch()
        records = [dict(episode_id=f"{job['agent']}_S{i:04d}_r0", agent_condition=job['agent'],
                        exposure_id=f'S{i:04d}', exposure_seed=3100+i, rollout_id='r0', rollout_seed=0)
                   for i in range(36)]
        with (d / 'analysis/episode_outcomes.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
        tasks.append((job, d))
        outcomes.append(dict(arm='A', variant=variant, seed=3100))
    collector.verify_evaluations(tasks, outcomes)
    f = tasks[-1][1] / 'analysis/episode_outcomes.csv'
    f.write_text(f.read_text().replace(',3100,', ',9999,'))
    with pytest.raises(ValueError, match='Exposure metadata differs'):
        collector.verify_evaluations(tasks, outcomes)


def test_counterfactual_coverage_is_separate(modules, tmp_path):
    _, fixed, collector, _ = modules
    job = fixed.jobs()[0]; directory = fixed.run_directory(tmp_path, job)
    tasks = [(job, directory)]
    assert collector.counterfactual_status(tasks)[0]['status'] == 'missing_or_incomplete'
    path = directory / 'rollout_counterfactual_front_vehicle/counterfactual_rollout_summary.csv'
    path.parent.mkdir(parents=True)
    rows = [f'{name},FD,36' for name in sorted(collector.CF_VARIANTS)]
    path.write_text('counterfactual_variant,agent_condition,n\n' + '\n'.join(rows) + '\n')
    assert collector.counterfactual_status(tasks)[0]['status'] == 'complete'
    path.write_text(path.read_text().replace('no-front,FD,36', 'no-front,FD,35'))
    assert collector.counterfactual_status(tasks)[0]['status'] == 'missing_or_incomplete'
