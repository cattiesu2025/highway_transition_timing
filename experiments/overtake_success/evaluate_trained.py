"""Deterministic Arm B final-policy evaluation on development or training scenes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import csv
import importlib.metadata
import json
from pathlib import Path
from statistics import median

import pilot
from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.transitions import select_first_confirmed_lane_change_onset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exposures', type=int, default=36)
    parser.add_argument('--training-tail', action='store_true',
                        help='Replay completed visible slow-front scenes from the last 20K training steps for 20 s')
    args = parser.parse_args()
    if not 1 <= args.exposures <= 36:
        parser.error('exposures must be 1..36')
    run = args.run_dir
    if not json.loads((run / 'complete.json').read_text()).get('complete'):
        raise ValueError('Training is incomplete')
    summary = json.loads((run / 'training_summary.json').read_text())
    if summary['success_definition'] != pilot.SUCCESS_DEFINITION:
        raise ValueError('Wrong success definition')
    args.out.mkdir(parents=True, exist_ok=False)
    _, module = pilot.load_modules()
    import torch
    torch.set_num_threads(1)
    config = module.ExperimentConfig(**summary['config'])
    if summary['arm'] != 'B':
        raise ValueError('This evaluator currently supports Arm B only')
    if config.evaluation_duration != 120:
        raise ValueError('This report expects a 120 s evaluation window')
    # Training's __main__ replay subclass/schedule were cloudpickled on Python
    # 3.11. Replace executable training-only objects before loading on another
    # Python minor version; keep the saved network parameters unchanged.
    model = pilot.DoubleDQN.load(
        run / 'final_model.zip', device='cpu', buffer_size=1,
        custom_objects={'replay_buffer_class': pilot.AuditedReplayBuffer,
                        'learning_rate': config.learning_rate,
                        'lr_schedule': lambda _: config.learning_rate,
                        'exploration_schedule': lambda _: 0.0})
    import zipfile
    with zipfile.ZipFile(run / 'final_model.zip') as archive:
        with archive.open('policy.pth') as saved:
            weights = torch.load(saved, map_location='cpu', weights_only=True)
    loaded = model.policy.state_dict()
    if weights.keys() != loaded.keys() or not all(torch.equal(v, loaded[k]) for k, v in weights.items()):
        raise ValueError('Loaded policy weights differ from saved weights')
    versions = {name: importlib.metadata.version(name) for name in
                ('highway-env', 'gymnasium', 'stable-baselines3', 'torch')}
    training_manifest = json.loads((run / 'manifest.json').read_text())
    metadata = dict(success_definition=pilot.SUCCESS_DEFINITION,
                    evaluation_mode='deterministic_training_tail_20s' if args.training_tail else 'deterministic_120s_continuation',
                    saved_policy_weights_verified=True,
                    model_sha256=module.file_sha256(run / 'final_model.zip'),
                    versions=versions, training_versions=training_manifest['versions'],
                    source_hashes={p.name:module.file_sha256(p) for p in
                                   (Path(__file__),Path(pilot.__file__))},
                    arm=summary['arm'],agent=summary['agent'],seed=summary['seed'],
                    group=summary['group'],config=asdict(config))
    metadata['inference_load_overrides'] = 'local replay class and unused training schedules; buffer_size=1; saved weights unchanged'
    (args.out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    scenes = module.make_eval_specs(args.exposures, config)
    if args.training_tail:
        with (run / 'training_episodes.csv').open() as f:
            tail = [r for r in csv.DictReader(f) if int(r['end_step']) > summary['steps'] - 20000
                    and r['reason'] != 'budget_stop' and r['scenario_type'] == 'train_visible_slow_front']
        scenes = [module.make_training_spec(int(r['reset_index']), config) for r in tail]
        for scene, original in zip(scenes, tail):
            for key, value in asdict(scene).items():
                if str(value if value is not None else '') != original[key]:
                    raise ValueError(f'Training scene mismatch: {scene.exposure_id}/{key}')
    if not scenes:
        raise ValueError('No evaluation scenes')
    episodes, steps = [], []
    for scene in scenes:
        env = module.make_env(summary['agent'], config, training=args.training_tail)
        try:
            obs, _ = env.reset(seed=scene.exposure_seed)
            obs = module.apply_open_lane_scene(env, scene)
            tracker = pilot.SuccessTracker(env, scene.include_front_vehicle)
            collision_time = None
            scene_rows = []
            for t in range(config.evaluation_max_policy_steps):
                action, _ = model.predict(obs, deterministic=True)
                ego = env.unwrapped.vehicle
                pre_lane = ego.lane_index[2]
                pre_speed = float(ego.speed)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                first = tracker.finish_policy_step(other_terminal=terminated)
                if tracker.collision and collision_time is None:
                    collision_time = round(tracker.elapsed, 8)
                row = dict(episode_id=scene.exposure_id, t=t,
                           time_after_step=round(tracker.elapsed, 8),
                           ego_lane=pre_lane, post_ego_lane=ego.lane_index[2],
                           ego_speed=pre_speed, post_ego_speed=float(ego.speed),
                           action=module.action_name_from_env(env, int(action)),
                           clearance=tracker.clearance, ego_y=float(ego.position[1]),
                           success_event=first, reward_base=float(reward),
                           collision_flag=tracker.collision,
                           terminated=bool(terminated), truncated=bool(truncated))
                scene_rows.append(row)
                if terminated or truncated:
                    break
            onset = select_first_confirmed_lane_change_onset(scene_rows, AnalysisConfig())
            episodes.append(dict(**asdict(scene), **tracker.summary(),
                                 collision_time=collision_time,
                                 onset_seconds=None if onset is None else onset['onset_t']/config.policy_frequency,
                                 duration_observed=round(tracker.elapsed, 8),
                                 policy_steps=len(scene_rows)))
            steps.extend(scene_rows)
            print(f"{scene.exposure_id}: success={tracker.success_time}, collision={collision_time}", flush=True)
        finally:
            env.close()
    pilot.write_csv(args.out / 'episodes.csv', episodes)
    with gzip.open(args.out / 'steps.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)
    times = [r['success_time'] for r in episodes if r['success_time'] is not None]
    onsets = [r['onset_seconds'] for r in episodes if r['onset_seconds'] is not None]
    result = dict(total=len(episodes),
                  observation_horizon_seconds=config.duration if args.training_tail else config.evaluation_duration,
                  successes_observed=len(times),
                  success20=sum(r['success_by_20'] for r in episodes),
                  success120=None if args.training_tail else sum(r['success_by_120'] for r in episodes),
                  collisions_observed=sum(r['collision'] for r in episodes),
                  collisions120=None if args.training_tail else sum(r['collision'] for r in episodes),
                  collisions20=sum(r['collision_time'] is not None and r['collision_time']<=20+1e-8 for r in episodes),
                  onset_observed=len(onsets), median_onset_seconds=median(onsets) if onsets else None,
                  median_success_seconds=median(times) if times else None,
                  success_time_range=[min(times),max(times)] if times else None)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    (args.out / 'complete.json').write_text(json.dumps({'complete': True}))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
