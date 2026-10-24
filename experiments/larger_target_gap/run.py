"""Independent larger-target-gap pilot; original experiment defaults stay intact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
SEEDS = (5200, 5201, 5202)
AGENTS = ('FD', 'BAL', 'SP')
JOBS = tuple(itertools.product(SEEDS, AGENTS))
# Each independently shuffled block gives identical reset-index schedules to all agents.
FAMILIES = ('slow',) * 6 + ('no_front',) * 2 + ('matched',) * 2
PROFILES = ('pilot-v1', 'broad-v2')


def profile_settings(profile):
    if profile not in PROFILES:
        raise ValueError(profile)
    broad = profile == 'broad-v2'
    return dict(prefix='larger_target_gap_broad_v2' if broad else 'larger_target_gap_pilot',
                front_range=(40, 150) if broad else (40, 80),
                eval_distances=(45, 60, 90, 120, 135, 150) if broad else (45, 60, 75),
                eval_extras=(30, 45) if broad else (35, 55, 75),
                max_target_distance=195 if broad else None)


def load_modules():
    path = ROOT / 'experiments/multilane_open_lane_change/rollout_counterfactual.py'
    spec = importlib.util.spec_from_file_location('larger_gap_counterfactual', path)
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf, cf.load_experiment_module()


def training_spec(module, reset_index, config, profile='pilot-v1'):
    block, slot = divmod(reset_index, len(FAMILIES))
    families = list(FAMILIES)
    random.Random(config.seed + 3_000_000 + block).shuffle(families)
    family = families[slot]
    rng = random.Random(config.seed + 4_000_000 + reset_index)
    settings = profile_settings(profile)
    distance = round(rng.uniform(*settings['front_range']), 3)
    extra_max = min(80, settings['max_target_distance'] - distance) if settings['max_target_distance'] else 80
    extra = round(rng.uniform(30, extra_max), 3)
    slow_speed = rng.uniform(10, 20)
    return module.build_open_lane_spec(
        exposure_id=f'train_reset_{reset_index:06d}',
        exposure_seed=config.seed + 4_000_000 + reset_index,
        ego_speed=28, front_distance=distance, front_speed=slow_speed if family == 'slow' else 28,
        scenario_type=f'larger_gap_{family}', include_front_vehicle=family != 'no_front',
        ego_lane=1, target_lane_gap=round(distance + extra, 3))


def eval_specs(module, config, profile='pilot-v1'):
    settings = profile_settings(profile)
    return [module.build_open_lane_spec(
        exposure_id=f'LG{i:03d}', exposure_seed=9_200_000 + i,
        ego_speed=28, front_distance=d, front_speed=v,
        scenario_type='larger_gap_slow', include_front_vehicle=True, ego_lane=1,
        target_lane_gap=d + extra)
        for i, (d, extra, v) in enumerate(itertools.product(settings['eval_distances'], settings['eval_extras'], (10, 14, 18)))]


def install(module, profile='pilot-v1'):
    module.make_training_spec = lambda i, c: training_spec(module, i, c, profile)
    module.STRATIFIED_TRAINING_BLOCK = tuple((f'larger_gap_{f}', '', '') for f in FAMILIES)
    original_scene = module.apply_open_lane_scene
    original_diagnostics = module.extract_diagnostics
    original_row = module.step_row_from_diagnostics

    def scene(env, spec, **kwargs):
        obs = original_scene(env, spec, **kwargs)
        env.unwrapped.larger_gap_vehicles = {
            'original': next((v for v in env.unwrapped.road.vehicles
                              if v is not env.unwrapped.vehicle and v.lane_index[2] == spec.ego_lane), None),
            'target': next((v for v in env.unwrapped.road.vehicles
                            if v is not env.unwrapped.vehicle and v.lane_index[2] == spec.ego_lane - 1), None)}
        return obs

    def diagnostics(env):
        result = original_diagnostics(env)
        ego = env.unwrapped.vehicle
        result['lg_ego_x'] = float(ego.position[0])
        for role, vehicle in getattr(env.unwrapped, 'larger_gap_vehicles', {}).items():
            prefix = f'lg_{role}_'
            result[prefix + 'present'] = vehicle is not None
            # Keep signed relative position even after the ego passes a vehicle.
            distance = float(vehicle.position[0] - ego.position[0]) if vehicle is not None else math.inf
            gap = distance - (vehicle.LENGTH + ego.LENGTH) / 2 if vehicle is not None else math.inf
            closing = float(ego.speed - vehicle.speed) if vehicle is not None else 0.
            visible = (vehicle in env.unwrapped.road.close_vehicles_to(
                ego, env.unwrapped.PERCEPTION_DISTANCE, count=3, see_behind=False, sort=True)
                if vehicle is not None else False)
            values = dict(x=float(vehicle.position[0]) if vehicle is not None else '',
                          speed=float(vehicle.speed) if vehicle is not None else '',
                          center_distance=distance, net_gap=gap, closing_speed=closing,
                          ttc=gap / closing if distance > 0 and closing > 0 else math.inf,
                          visible=visible)
            result.update({prefix + k: v for k, v in values.items()})
        a, b = result.get('lg_original_net_gap', math.inf), result.get('lg_target_net_gap', math.inf)
        result['lg_extra_gap'] = b - a if math.isfinite(a) and math.isfinite(b) else ''
        # Only compare two actual front vehicles; removal/passing is a separate state.
        result['lg_target_more_space'] = b > a if math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 else ''
        return result

    def row(**kwargs):
        result = original_row(**kwargs)
        for prefix, key in (('', 'diagnostics'), ('post_', 'post_step_diagnostics')):
            result.update({prefix + k: v for k, v in kwargs[key].items() if k.startswith('lg_')})
        return result

    module.apply_open_lane_scene = scene
    module.extract_diagnostics = diagnostics
    module.step_row_from_diagnostics = row


def preflight(module, config, output, profile='pilot-v1'):
    """Check all candidate initial scenes and scripted first-step lane changes."""
    rows = []
    specs = eval_specs(module, config, profile)
    specs.append(module.build_open_lane_spec(
        exposure_id='training_tight_corner', exposure_seed=9_300_000, ego_speed=28,
        front_distance=40, front_speed=10, scenario_type='corner', include_front_vehicle=True,
        ego_lane=1, target_lane_gap=70))
    for spec in specs:
        for policy in ('idle', 'left_once'):
            env = module.make_env('BAL', config, training=False)
            try:
                env.reset(seed=spec.exposure_seed)
                obs = module.apply_open_lane_scene(env, spec)
                state = module.extract_diagnostics(env)
                assert state['lg_extra_gap'] > 0
                assert int(obs[:, 0].sum()) == 3, (spec.exposure_id, obs)
                assert state['lg_original_visible'] and state['lg_target_visible']
                position_clipped = bool((abs(obs[1:3, 1]) >= 1).any())
                assert not position_clipped, (spec.exposure_id, obs)
                changed, collision, elapsed = False, False, 0.
                for t in range(config.evaluation_max_policy_steps):
                    _, _, term, trunc, _ = env.step(0 if policy == 'left_once' and t == 0 else 1)
                    changed |= env.unwrapped.vehicle.lane_index[2] != 1
                    collision |= bool(env.unwrapped.vehicle.crashed)
                    elapsed = (t + 1) / config.policy_frequency
                    if term or trunc:
                        break
                rows.append(dict(exposure_id=spec.exposure_id, policy=policy,
                                 front_distance=spec.front_distance, target_distance=spec.target_lane_gap,
                                 vehicles_observed=3, position_clipped=position_clipped,
                                 perception_distance=env.unwrapped.PERCEPTION_DISTANCE,
                                 front_distance_score=state['front_distance_score'],
                                 lane_changed=changed, collision=collision,
                                 elapsed_seconds=elapsed))
            finally:
                env.close()
    module.write_csv_rows(output / 'preflight.csv', rows)
    if any(r['collision'] or not r['lane_changed'] for r in rows if r['policy'] == 'left_once'):
        raise RuntimeError('Scripted lane-change preflight failed; inspect preflight.csv before training.')
    return rows


def evaluate(cf, module, model, agent, config, output, smoke=False, profile='pilot-v1'):
    specs = eval_specs(module, config, profile)
    if smoke:
        specs = specs[:3]
    all_summary = []
    for variant in ('original', 'no-front', 'open-target-lane', 'matched-speed-front'):
        steps, exposures = cf.rollout_variant(module, {agent: model}, [agent], specs, config, variant)
        directory = output / 'evaluation' / variant
        module.write_csv_rows(directory / 'steps.csv', steps)
        module.write_csv_rows(directory / 'exposures.csv', exposures)
        summaries = module.summarize_actual_lane_changes(steps, config.policy_frequency)
        for summary in summaries:
            group = [r for r in steps if r['episode_id'] == summary['episode_id']]
            first_change = next((r for r in group if r['post_ego_lane'] != 1), None)
            first_left = next((r for r in group if r['action'] == 'LANE_LEFT'), None)
            summary['variant'] = variant
            summary['seed'] = config.seed
            # Preserve existing endpoint while additionally identifying the sample boundary.
            summary['physical_change_sample_end_seconds'] = ((first_change['t'] + 1) / config.policy_frequency
                                                              if first_change else '')
            for name, onset in (('first_left', first_left), ('physical_change', first_change)):
                summary[name + '_target_more_space'] = onset['lg_target_more_space'] if onset else ''
                for key in ('lg_original_net_gap', 'lg_target_net_gap', 'lg_extra_gap', 'lg_original_ttc'):
                    summary[name + '_' + key] = onset[key] if onset else ''
        module.write_csv_rows(directory / 'episode_summary.csv', summaries)
        all_summary.extend(summaries)
    module.write_csv_rows(output / 'episode_summary.csv', all_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(9), default=0)
    parser.add_argument('--out-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--profile', choices=PROFILES, default='pilot-v1')
    args = parser.parse_args()
    seed, agent = JOBS[args.index]
    settings = profile_settings(args.profile)
    suffix = '_preflight' if args.preflight_only else '_smoke' if args.smoke else ''
    output = args.out_root.resolve() / f"{settings['prefix']}_{agent}_100k_seed{seed}{suffix}"
    protocol = dict(profile=args.profile, seed=seed, agent=agent, timesteps=200 if args.smoke else 100000,
                    initialization='from_scratch_no_checkpoint_loaded',
                    training_duration=20, evaluation_duration=20,
                    grid_scenes=len(settings['eval_distances'])*len(settings['eval_extras'])*3,
                    training_family_fractions={'slow': .6, 'no_front': .2, 'matched': .2},
                    front_distance_range=settings['front_range'], extra_gap_range=[30, 80], ego_speed=28,
                    max_target_distance=settings['max_target_distance'],
                    extra_gap_sampling='uniform(30, min(80, 195-front_distance))' if args.profile=='broad-v2' else 'uniform(30,80)',
                    eval_distances=settings['eval_distances'], eval_extras=settings['eval_extras'],
                    target_speed=29, slow_front_speed_range=[10, 20],
                    checkpoint='final_fixed_step_no_reselection', smoke=args.smoke,
                    preflight_only=args.preflight_only, output=str(output))
    print(json.dumps(protocol, indent=2), flush=True)
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cf, module = load_modules()
    module.configure_headless_runtime()
    install(module, args.profile)
    config = module.ExperimentConfig(seed=seed, duration=20, evaluation_duration=20,
                                     lanes_count=2, ego_lane=1, target_lane_vehicle=True)
    (output / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    sources = [Path(__file__), Path(__file__).with_name('DESIGN.md'),
               ROOT / 'experiments/multilane_open_lane_change/run.py',
               ROOT / 'experiments/multilane_open_lane_change/rollout_counterfactual.py',
               *sorted((ROOT / 'src/highway_transition_timing').glob('*.py'))]
    if args.profile == 'broad-v2':
        sources.append(Path(__file__).with_name('BROAD_V2.md'))
    provenance = dict(packages={n: importlib.metadata.version(n) for n in
                               ('torch', 'stable-baselines3', 'highway-env', 'gymnasium')},
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sources})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    module.write_csv_rows(output / 'evaluation_grid.csv', [asdict(s) for s in eval_specs(module, config, args.profile)])
    module.write_csv_rows(output / 'training_schedule_first100.csv',
                          [asdict(training_spec(module, i, config, args.profile)) for i in range(100)])
    preflight(module, config, output, args.profile)
    if args.preflight_only:
        return
    module.train_agents(output, [agent], protocol['timesteps'], config, 0,
                        checkpoint_every=100 if args.smoke else 5000)
    model = module.dqn_class_for_variant(config.dqn_variant).load(output / f'models/{agent}_main.zip', device='cpu')
    assert model.num_timesteps == protocol['timesteps']
    evaluate(cf, module, model, agent, config, output, args.smoke, args.profile)
    (output / 'complete.json').write_text(json.dumps(dict(complete=True, smoke=args.smoke)) + '\n')
    print('Completed', output, flush=True)


if __name__ == '__main__':
    main()
