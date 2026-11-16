"""Distance-only Arm C extension. Dry run by default; never mutates old runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
WEIGHTS = {'FD': ('F1', .28, 1.12), 'BAL': ('F4', .56, .84), 'SP': ('F7', .98, .42)}
VERSION = 'distance-affine-v1'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base():
    return load_module('arm_c_distance_base', ROOT / 'experiments/multilane_open_lane_change/run.py')


def config_for(module, agent, seed):
    _, speed, spacing = WEIGHTS[agent]
    return module.ExperimentConfig(duration=20, evaluation_duration=120, seed=seed,
        target_lane_vehicle=True, speed_weight=speed, front_distance_weight=spacing)


def transform(module, source):
    """Deterministic push-forward of the old scene; no resampling or rejection."""
    if not source.include_front_vehicle:
        return source
    if not 90 <= source.front_distance <= 195 or source.target_lane_gap is None:
        raise ValueError('Expected an original Arm C scene within its training support')
    if not 10 <= source.target_lane_gap <= 90:
        raise ValueError('Expected original target distance in [10,90]')
    # Retain source distance rank, including the former boundary family.
    distance = round(40 + (source.front_distance - 90) * 110 / 105, 3)
    extra_max = min(80, 195 - distance)
    extra = 30 + (source.target_lane_gap - 10) / 80 * (extra_max - 30)
    fields = asdict(source)
    for key in ('ttc_bin', 'required_deceleration_bin', 'visible_at_t0'):
        fields.pop(key)
    fields.update(front_distance=distance, target_lane_gap=round(distance + extra, 3))
    return module.build_open_lane_spec(**fields)


def install(module):
    """Override only this freshly loaded module, leaving the original files intact."""
    original_training = module.make_training_spec
    original_eval = module.make_eval_specs
    original_row = module.training_spec_row
    module.source_training_spec = original_training
    module.source_eval_specs = original_eval
    module.make_training_spec = lambda i, c: transform(module, original_training(i, c))
    module.make_eval_specs = lambda n, c: [transform(module, s) for s in original_eval(n, c)]

    def row(index, spec, config):
        source = original_training(index, config)
        result = original_row(index, spec, config)
        result.update(geometry_profile=VERSION, source_scenario_type=source.scenario_type,
            source_ttc_bin=source.ttc_bin,
            source_required_deceleration_bin=source.required_deceleration_bin,
            source_front_distance=source.front_distance,
            source_target_distance=source.target_lane_gap,
            target_lane_gap=spec.target_lane_gap, adjacent_lanes_open=spec.target_lane_gap is None)
        return result
    module.training_spec_row = row
    # Reuse existing fixed-vehicle logging without inheriting the pilot sampler.
    pilot = load_module('arm_c_distance_logging', ROOT / 'experiments/larger_target_gap/run.py')
    training, block = module.make_training_spec, module.STRATIFIED_TRAINING_BLOCK
    pilot.install(module)
    module.make_training_spec, module.STRATIFIED_TRAINING_BLOCK = training, block


def audit(module, config, out, count=2000):
    rows = []
    for i in range(count):
        source = module.source_training_spec(i, config)
        actual = module.make_training_spec(i, config)
        before, after = asdict(source), asdict(actual)
        allowed = {'front_distance', 'target_lane_gap', 'ttc_bin', 'required_deceleration_bin', 'visible_at_t0'}
        assert all(before[k] == after[k] for k in before.keys() - allowed)
        if actual.include_front_vehicle:
            assert 40 <= actual.front_distance <= 150
            assert actual.front_distance + 30 <= actual.target_lane_gap <= 195
        else:
            assert actual == source and actual.target_lane_gap is None
        rows.append(module.training_spec_row(i, actual, config))
    module.write_csv_rows(out / 'paired_training_audit.csv', rows)
    module.write_csv_rows(out / 'evaluation_scenes.csv', [asdict(s) for s in module.make_eval_specs(36, config)])
    return {'paired_resets': count, 'unchanged_non_distance_fields': True}


def preflight(module, config, out):
    """Check actual observations and short scripted lane changes, not learned behavior."""
    rows = []
    specs = module.make_eval_specs(36, config) + [module.make_training_spec(i, config) for i in range(20)]
    # Include the most compressed geometric/speed corner explicitly.
    specs.append(module.build_open_lane_spec(exposure_id='tight_corner', exposure_seed=9900000,
        ego_speed=31, front_speed=10, front_distance=40, target_lane_gap=70,
        scenario_type='preflight_corner', include_front_vehicle=True, ego_lane=1))
    for spec in specs:
        env = module.make_env('BAL', config, training=False)
        try:
            env.reset(seed=spec.exposure_seed)
            obs = module.apply_open_lane_scene(env, spec, include_front_vehicle=spec.include_front_vehicle)
            expected = 3 if spec.include_front_vehicle else 1
            assert int(obs[:, 0].sum()) == expected
            assert not (abs(obs[1:expected, 1]) >= 1).any()
            state = module.extract_diagnostics(env)
            if spec.include_front_vehicle:
                assert state['lg_original_visible'] and state['lg_target_visible']
                assert state['lg_target_more_space']
            for t in range(5):
                _, _, terminated, truncated, _ = env.step(0 if t == 0 else 1)
                if terminated or truncated:
                    break
            rows.append(dict(exposure_id=spec.exposure_id, observed_vehicles=expected,
                collision=bool(env.unwrapped.vehicle.crashed),
                lane_changed=env.unwrapped.vehicle.lane_index[2] != spec.ego_lane))
        finally:
            env.close()
    module.write_csv_rows(out / 'preflight.csv', rows)
    if any(r['collision'] or not r['lane_changed'] for r in rows):
        raise RuntimeError('Scripted preflight failed; inspect preflight.csv')
    return {'preflight_scenes': len(rows), 'scripted_collisions': 0}


def baseline_status(root, agent, seed):
    variant, speed, spacing = WEIGHTS[agent]
    prefix = 'reward_fixed_sum' if seed < 4205 else 'reward_selected_validation'
    directory = root / f'{prefix}_armC_{variant}_100k_seed{seed}'
    result = {'path': str(directory), 'status': 'missing'}
    if not directory.is_dir():
        return result
    try:
        with (directory / 'training_runs.csv').open() as handle:
            records = [r for r in csv.DictReader(handle) if r['agent_condition'] == agent]
        if len(records) != 1:
            raise ValueError('Expected one matching training record')
        row = records[0]
        required = dict(seed=seed, total_timesteps=100000, reward_speed_score=speed,
            reward_front_distance_score=spacing, training_duration_seconds=20,
            evaluation_duration_seconds=120, policy_frequency_hz=5, lanes_count=2,
            reward_slow_down_penalty=.2, reward_lane_change_penalty=.2,
            reward_collision_risk_penalty=3, reward_collision_penalty=2)
        for key, value in required.items():
            if not math.isclose(float(row[key]), value):
                raise ValueError(f'Incompatible {key}')
        if row['target_lane_vehicle'].lower() != 'true' or row['training_scenario_profile'] != 'stratified' or row['dqn_variant'] != 'double-dqn':
            raise ValueError('Incompatible baseline protocol')
        with (directory / 'model_selection.csv').open() as handle:
            selected = [r for r in csv.DictReader(handle) if r['agent_condition'] == agent and r['selected'].lower() == 'true' and r['eligible'].lower() == 'true']
        if len(selected) != 1 or not (directory / 'models' / f'{agent}_main.zip').is_file():
            raise ValueError('Eligible selected baseline missing')
        result['status'] = 'metadata_verified'
    except (OSError, ValueError, KeyError) as exc:
        result.update(status='unavailable_or_incompatible', reason=str(exc))
    return result


def run(args):
    module = load_base()
    config = config_for(module, args.agent, args.seed)
    install(module)
    variant = WEIGHTS[args.agent][0]
    name = f'arm_c_{VERSION}_{variant}_100k_seed{args.seed}'
    if args.mode != 'train':
        name += f'_{args.mode}'
    out = args.out_root.resolve() / name
    record = dict(experiment='arm_c_distance_variant', geometry_profile=VERSION,
        mode=args.mode, agent=args.agent, variant=variant, config=asdict(config),
        timesteps={'audit': 0, 'smoke': 1200, 'train': 100000}[args.mode],
        checkpoint_every=5000, num_exposures=36,
        baseline=baseline_status(args.baseline_root.resolve(), args.agent, args.seed),
        selection='latest_eligible' if args.mode == 'train' else 'not_scientific',
        output=str(out))
    print(json.dumps(record, indent=2), flush=True)
    if not args.execute:
        return 0
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'experiments/multilane_open_lane_change/run.py',
        ROOT / 'experiments/multilane_open_lane_change/rollout_counterfactual.py',
        ROOT / 'experiments/larger_target_gap/run.py'] + sorted((ROOT / 'src/highway_transition_timing').glob('*.py'))
    record['effective_learning_starts'] = min(config.learning_starts, max(1, record['timesteps'] // 10)) if record['timesteps'] else None
    record['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out / 'protocol.json').write_text(json.dumps(record, indent=2) + '\n')
    status = {'status': 'running', 'scientific_result': False}
    try:
        status.update(audit(module, config, out))
        status.update(preflight(module, config, out))
        if args.mode != 'audit':
            module.train_agents(out, [args.agent], record['timesteps'], config, verbose=0)
            if args.mode == 'train':
                module.select_models(out, [args.agent], 36, config)
                steps, exposures = module.evaluate_agents(out, [args.agent], 36, config)
                module.write_analysis(out, steps, exposures, config, bootstrap_samples=500, figures=False)
                cf = module.load_rollout_counterfactual_module()
                model = module.dqn_class_for_variant(config.dqn_variant).load(out / 'models' / f'{args.agent}_main.zip')
                for variant_name in module.GATE_VARIANTS:
                    steps, exposures = cf.rollout_variant(module, {args.agent: model}, [args.agent],
                        module.make_eval_specs(36, config), config, variant_name)
                    module.write_analysis(out / 'controls' / variant_name, steps, exposures, config,
                        bootstrap_samples=0, figures=False)
                status['scientific_result'] = True
            else:
                # Exercise loading, 120-second rollouts, logging and analysis for one scene.
                steps, exposures = module.evaluate_agents(out, [args.agent], 1, config)
                module.write_analysis(out, steps, exposures, config, bootstrap_samples=0, figures=False)
        status['status'] = 'complete'
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=WEIGHTS, default='BAL')
    parser.add_argument('--seed', type=int, choices=range(4200, 4220), default=4200)
    parser.add_argument('--mode', choices=('audit', 'smoke', 'train'), default='audit')
    parser.add_argument('--out-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--baseline-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--execute', action='store_true')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
