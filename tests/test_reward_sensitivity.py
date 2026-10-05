"""Reward sweeps must change actual rewards, not just plot labels."""
import importlib.util
from pathlib import Path
import sys

import pytest

from highway_transition_timing.highway_adapter import weighted_reward
from highway_transition_timing.rewards import MAIN_REWARD_WEIGHTS, with_preference_weights

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sweep = load('experiments/reward_sensitivity/run.py', 'sweep_test')


@pytest.mark.parametrize('folder', ['single_lane_slow_front', 'multilane_open_lane_change'])
def test_cli_weights_reach_environment_and_report(folder, monkeypatch):
    module = load(f'experiments/{folder}/run.py', f'sweep_{folder}')
    args = module.build_parser().parse_args(['--out', '/tmp/unused', '--speed-weight', '.33', '--front-distance-weight', '.51'])
    config = module.config_from_args(args)
    if folder == 'single_lane_slow_front':
        weights = module.effective_reward_weights('FD', config)
        table = module.reward_config_table_with_strength_multiplier(0, 3, 1, speed_weight=.33, front_distance_weight=.51)
    else:
        pytest.importorskip('highway_env')
        env = module.make_env('FD', config)
        weights = env.weights
        env.close()
        table = module.reward_config_table_with_slow_down_penalty(.2, 3, speed_weight=.33, front_distance_weight=.51)
    assert (weights.speed_score, weights.front_distance_score) == (.33, .51)
    assert weights.collision_penalty == 2
    assert weights.collision_risk_penalty == 3
    assert table[0]['speed_score'] == .33
    components = dict(speed_score=.4, front_distance_score=.8, collision_penalty=0, lane_change_penalty=0, right_lane_score=0)
    assert weighted_reward(components, weights) == pytest.approx(.33*.4 + .51*.8)
    with pytest.raises(ValueError):
        module.ExperimentConfig(speed_weight=.3)


@pytest.mark.parametrize('values', [(None, .3), (.3, None), (0, .3), (-1, .3), (float('nan'), .3), (.3, float('inf'))])
def test_invalid_overrides(values):
    with pytest.raises(ValueError):
        with_preference_weights(MAIN_REWARD_WEIGHTS['FD'], *values)


def test_manifest_and_commands():
    assert sweep.reproduce_sample() == sweep.VARIANTS
    jobs = sweep.jobs()
    assert len(jobs) == 90
    assert len({str(sweep.run_directory('/tmp/out', j)) for j in jobs}) == 90
    for job in jobs:
        for phase in ('train', 'evaluate'):
            for command in sweep.commands(job, '/tmp/out', phase):
                assert not any('heldout' in arg for arg in command)
                if job['arm'] == 'A' or phase == 'train':
                    assert command[command.index('--speed-weight')+1] == str(job['speed_weight'])
        command = sweep.commands(job, '/tmp/out', 'train')[0]
        assert ('--target-lane-vehicle' in command) == (job['arm'] == 'C')


def test_counterfactual_restores_custom_weights(tmp_path):
    module = load('experiments/multilane_open_lane_change/run.py', 'sweep_restore_env')
    cf = load('experiments/multilane_open_lane_change/rollout_counterfactual.py', 'sweep_cf')
    (tmp_path/'training_runs.csv').write_text('seed,configured_speed_weight,configured_front_distance_weight\n4200,1.17,0.76\n')
    config = cf.config_from_training_run(module, tmp_path, 120)
    assert (config.speed_weight, config.front_distance_weight) == (1.17, .76)


def test_refuses_overwrite(tmp_path):
    run_dir = sweep.run_directory(tmp_path, sweep.jobs()[0])
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        sweep.main(['--index', '0', '--out-root', str(tmp_path), '--execute'])


def test_collector_uses_physical_onsets_and_keeps_censoring(tmp_path, monkeypatch):
    import csv
    monkeypatch.setitem(sys.modules, 'run', sweep)
    collector = load('experiments/reward_sensitivity/collect.py', 'sweep_collector')
    job = dict(arm='B', variant='R1', agent='FD', seed=4100,
               speed_weight=.33, front_distance_weight=.51)
    monkeypatch.setattr(sweep, 'jobs', lambda: [job])
    monkeypatch.setattr(sweep, 'BASE_SEEDS', {})
    directory = sweep.run_directory(tmp_path, job)
    def write(relative, rows):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write('training_runs.csv', [dict(agent_condition='FD', reward_speed_score=.33,
        reward_front_distance_score=.51, reward_collision_penalty=2, reward_collision_risk_penalty=3,
        reward_slow_down_penalty=.2, reward_lane_change_penalty=.2, seed=4100,total_timesteps=100000,
        training_duration_seconds=20,evaluation_duration_seconds=120,policy_frequency_hz=5,
        dqn_variant='double-dqn',training_scenario_profile='stratified',target_lane_vehicle=False,lanes_count=2)])
    write('training_scenarios.csv',[dict(agent_condition='FD',reset_index=0,ego_speed=28)])
    base=tmp_path/'multilane_twolane_open_fixed_100k_seed4100'
    base.mkdir()
    (base/'training_scenarios.csv').write_text((directory/'training_scenarios.csv').read_text())
    write('training_logs/FD/progress.csv', [{'time/total_timesteps':s,'rollout/ep_rew_mean':10} for s in (1000,100000)])
    write('analysis/episode_outcomes.csv',[dict(agent_condition='FD',episode_id=str(i),
        episode_outcome='valid_onset',collision_flag=False,response_latency=9) for i in range(36)])
    write('analysis/actual_lane_change_summary.csv',[dict(agent_condition='FD',episode_id=str(i),
        actual_lane_change_observed=i>0,first_actual_lane_change_seconds=2.4 if i else '') for i in range(36)])
    write('evaluation/steps.csv',[dict(agent_condition='FD',episode_id=str(i),ego_speed=28,nearest_front_distance='inf') for i in range(36)])
    out=tmp_path/'collected'
    collector.collect(tmp_path,out,allow_incomplete=True)
    summary=next(collector.rows(out/'seed_summary.csv'))
    assert float(summary['conditional_median_onset_seconds']) == 2.4
    assert int(summary['n_onset']) == 35
    assert int(summary['n_censored']) == 1
    assert summary['mean_episode_finite_front_distance_m'] == ''
    assert int(summary['n_episodes_with_finite_front_distance']) == 0
    (base/'training_scenarios.csv').write_text('agent_condition,reset_index,ego_speed\nFD,0,30\n')
    with pytest.raises(ValueError, match='scenario prefix'):
        collector.collect(tmp_path,out,allow_incomplete=True)
