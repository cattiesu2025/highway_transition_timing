"""Success-trigger diagnostic on unchanged scenes; not the formal experiment."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SUCCESS_DEFINITION = 'complete_overtake_return_v1'
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/overtake-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/overtake-cache')

import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from highway_transition_timing.double_dqn import DoubleDQN


def load_modules():
    path = ROOT / 'experiments/multilane_open_lane_change/rollout_counterfactual.py'
    spec = importlib.util.spec_from_file_location('overtake_pilot_cf', path)
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf, cf.load_experiment_module()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SuccessTracker:
    """Passive per-simulation-step tracker bound to the initial front vehicle."""

    def __init__(self, env, applicable=True):
        self.env = env.unwrapped
        self.ego = self.env.vehicle
        self.target = self.env.road.vehicles[1] if applicable else None
        self.original_lane = self.ego.lane_index
        self.left = (*self.original_lane[:2], self.original_lane[2] - 1)
        self.left_completed_time = None
        self.passed_in_left_time = None
        self.returned_ahead_time = None
        self.elapsed = 0.0
        self.since = None
        self.pending = False
        self.success_time = None
        self.clearance = None
        self.max_clearance = -math.inf
        self.collision = False
        self.offroad = False
        self.left_entered = False
        self.passed = False
        self.successful_lane_dwell = 0.0
        self.max_dwell = 0.0
        road = self.env.road
        original = road.step

        def observed_step(dt):
            original(dt)
            self.observe(dt)

        road.step = observed_step

    def observe(self, dt):
        self.elapsed += dt
        ego = self.ego
        self.collision |= bool(ego.crashed)
        self.offroad |= not bool(ego.on_road)
        self.left_entered |= ego.lane_index == self.left
        if self.target is None:
            return
        # Current experiment is a straight road aligned with world x.
        def half_extent(v):
            return (abs(math.cos(v.heading)) * v.LENGTH
                    + abs(math.sin(v.heading)) * v.WIDTH) / 2
        self.clearance = float(ego.position[0] - half_extent(ego)
                               - self.target.position[0] - half_extent(self.target))
        self.max_clearance = max(self.max_clearance, self.clearance)
        self.passed |= self.clearance >= 5.0
        lane = self.env.road.network.get_lane(self.left)
        _, offset = lane.local_coordinates(ego.position)
        safe = not self.offroad and not self.collision
        if ego.lane_index == self.left and abs(offset) <= 0.5 and safe:
            if self.left_completed_time is None:
                self.left_completed_time = self.elapsed
        if (self.left_completed_time is not None and ego.lane_index == self.left
                and self.clearance >= 5.0 and safe and self.passed_in_left_time is None):
            self.passed_in_left_time = self.elapsed
        original = self.env.road.network.get_lane(self.original_lane)
        _, return_offset = original.local_coordinates(ego.position)
        qualifies = (self.passed_in_left_time is not None
                     and ego.lane_index == self.original_lane and abs(return_offset) <= 0.5
                     and self.clearance >= 5.0 and safe)
        if qualifies:
            if self.returned_ahead_time is None:
                self.returned_ahead_time = self.elapsed
            if self.since is None:
                self.since = self.elapsed
            self.successful_lane_dwell = self.elapsed - self.since
            self.max_dwell = max(self.max_dwell, self.successful_lane_dwell)
            if self.successful_lane_dwell >= 1.0 - 1e-9:
                self.pending = True
        else:
            self.since = None
            self.successful_lane_dwell = 0.0

    def finish_policy_step(self, other_terminal=False):
        first = (self.pending and self.success_time is None
                 and not self.collision and not self.offroad and not other_terminal)
        if first:
            self.success_time = round(self.elapsed, 8)
        self.pending = False
        return bool(first)

    def summary(self):
        return dict(success_definition=SUCCESS_DEFINITION,
                    success_applicable=self.target is not None,
                    left_completed_time=self.left_completed_time,
                    passed_in_left_time=self.passed_in_left_time,
                    returned_ahead_time=self.returned_ahead_time,
                    success_time=self.success_time,
                    success_by_20=self.success_time is not None and self.success_time <= 20 + 1e-8,
                    success_by_120=self.success_time is not None and self.success_time <= 120 + 1e-8,
                    collision=self.collision, offroad=self.offroad, left_entered=self.left_entered,
                    ever_clearance_5m=self.passed,
                    max_clearance=None if self.target is None else self.max_clearance,
                    max_qualifying_dwell=self.max_dwell)


class TriggerWrapper(gym.Wrapper):
    """Log actual training success transitions, optionally deliver the bonus."""

    def __init__(self, env, beta=0, terminal=False):
        super().__init__(env)
        self.beta, self.terminal = beta, terminal
        self.rows = []
        self.global_step = 0
        self.tracker = None
        self.episode_steps = 0
        self.success_global_steps = []

    def reset(self, **kwargs):
        if self.tracker is not None and self.episode_steps:
            raise RuntimeError('Reset of unfinished diagnostic episode')
        obs, info = self.env.reset(**kwargs)
        scene = self.env.env.last_training_exposure
        self.scene = scene
        self.tracker = SuccessTracker(self.env, scene.include_front_vehicle)
        self.episode_steps = 0
        return obs, info

    def step(self, action):
        obs, base, terminated, truncated, info = self.env.step(action)
        self.global_step += 1
        self.episode_steps += 1
        first = self.tracker.finish_policy_step(other_terminal=terminated)
        info = dict(info, overtake_success_event=first)
        if first:
            self.success_global_steps.append(self.global_step)
            if self.terminal:
                terminated, truncated = True, False
        reward = base + self.beta * first
        if terminated or truncated:
            reason = ('collision' if self.tracker.collision else
                      'overtake_success' if first and self.terminal else
                      'other_terminal' if terminated else 'duration')
            self.record(reason)
            self.episode_steps = 0
        return obs, reward, terminated, truncated, info

    def record(self, reason):
        self.rows.append(dict(reset_index=len(self.rows), end_step=self.global_step,
                              episode_steps=self.episode_steps, reason=reason,
                              **asdict(self.scene), **self.tracker.summary()))

    def finish_log(self):
        if self.episode_steps:
            self.record('budget_stop')
            self.episode_steps = 0


class AuditedReplayBuffer(ReplayBuffer):
    """Track reward-bearing transitions actually sampled for training (one env)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError('Pilot replay audit requires one environment')
        self.events = np.zeros(self.buffer_size, dtype=bool)
        self.sampled = np.zeros(self.buffer_size, dtype=bool)
        self.event_inserts = self.event_draws = self.unique_events_sampled = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        event = bool(infos[0].get('overtake_success_event', False))
        self.events[self.pos] = event
        self.sampled[self.pos] = False
        self.event_inserts += int(event)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        hits = batch_inds[self.events[batch_inds]]
        self.event_draws += len(hits)
        unique = np.unique(hits)
        self.unique_events_sampled += int((~self.sampled[unique]).sum())
        self.sampled[unique] = True
        return super()._get_samples(batch_inds, env)


def evaluate_old(args, out, cf, module):
    rows = []
    for arm, seeds in [('B', (4100, 4101)), ('C', (4200, 4201))]:
        for seed in seeds:
            name = 'open' if arm == 'B' else 'occupied'
            run = ROOT / f'outputs/multilane_twolane_{name}_fixed_100k_seed{seed}'
            if not (run / 'training_runs.csv').exists():
                raise FileNotFoundError(run)
            config = cf.config_from_training_run(module, run, 120)
            specs = module.make_eval_specs(args.exposures, config)
            for agent in ('FD', 'BAL', 'SP'):
                # Reassess the policy previously reported, not an unverified final model.
                path = run / 'models' / f'{agent}_main.zip'
                model = DoubleDQN.load(path, device='cpu')
                model_hash = module.file_sha256(path)
                for scene in specs:
                    env = module.make_env(agent, config)
                    obs, _ = env.reset(seed=scene.exposure_seed)
                    obs = module.apply_open_lane_scene(env, scene)
                    tracker = SuccessTracker(env)
                    for step in range(600):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = env.step(int(action))
                        tracker.finish_policy_step(other_terminal=terminated)
                        if terminated or truncated or tracker.success_time is not None:
                            break
                    rows.append(dict(arm=arm, seed=seed, agent=agent,
                                     checkpoint='maintained_selected', model_sha256=model_hash,
                                     **asdict(scene), **tracker.summary(), steps=step+1))
                    env.close()
                write_csv(out / 'old_policy_episodes.csv', rows)
                print(f'old-policy {arm} {seed} {agent}: {len(rows)} episodes', flush=True)
    return rows


def train_one(args, out, module):
    config = module.ExperimentConfig(seed=args.seed, duration=20, evaluation_duration=120,
                                    target_lane_vehicle=args.arm == 'C')
    beta = 30 if args.group in ('S10', 'S11') else 0
    env = TriggerWrapper(module.make_env(args.agent, config, training=True),
                         beta, args.group in ('S01', 'S11'))
    model = DoubleDQN('MlpPolicy', env, learning_rate=config.learning_rate,
                      buffer_size=config.buffer_size, learning_starts=1000,
                      batch_size=config.batch_size, gamma=config.gamma,
                      train_freq=1, gradient_steps=1, target_update_interval=250,
                      exploration_fraction=.25, exploration_initial_eps=1,
                      exploration_final_eps=.05, seed=args.seed, device='cpu',
                      replay_buffer_class=AuditedReplayBuffer, verbose=0)
    # A short diagnostic is a prefix of the 100K exploration schedule, not
    # a compressed 2K training schedule (which would reach epsilon .05 at 500).
    model.exploration_schedule = lambda remaining: (
        1.0 - .95 * min(args.steps * (1.0 - remaining) / 25000.0, 1.0))
    model.learn(total_timesteps=args.steps)
    env.finish_log()
    write_csv(out / 'training_episodes.csv', env.rows)
    write_csv(out / 'training_scenarios.csv', env.env.env.training_spec_rows)
    model.save(out / 'final_model.zip')
    replay = model.replay_buffer
    report = dict(success_definition=SUCCESS_DEFINITION,
                  arm=args.arm, seed=args.seed, agent=args.agent, group=args.group,
                  steps=args.steps, config=asdict(config), beta=beta,
                  exploration_decay_steps=25000, final_epsilon=model.exploration_rate,
                  first_success_step=next(iter(env.success_global_steps), None),
                  success_steps=env.success_global_steps,
                  success_transitions_inserted=replay.event_inserts,
                  success_transition_sample_draws=replay.event_draws,
                  unique_success_transitions_sampled=replay.unique_events_sampled,
                  episodes=len(env.rows),
                  completed_episodes=sum(r['reason'] != 'budget_stop' for r in env.rows),
                  budget_stop_episodes=sum(r['reason'] == 'budget_stop' for r in env.rows))
    (out / 'training_summary.json').write_text(json.dumps(report, indent=2))
    env.close()
    print(json.dumps({k: v for k, v in report.items() if k not in ('config', 'success_steps')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['old-policy', 'train'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', choices=['B', 'C'], default='B')
    parser.add_argument('--agent', choices=['FD', 'BAL', 'SP'], default='BAL')
    parser.add_argument('--seed', type=int, default=4190)
    parser.add_argument('--group', choices=['S00', 'S10', 'S01', 'S11'], default='S10')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--exposures', type=int, default=36)
    args = parser.parse_args()
    if args.steps <= 0 or not 1 <= args.exposures <= 36:
        parser.error('steps must be positive and exposures in 1..36')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    cf, module = load_modules()
    import importlib.metadata
    manifest = dict(args=vars(args), success_definition=SUCCESS_DEFINITION,
                    stage='unfiltered_original_scene_trigger_diagnostic',
                    source_hashes={str(p.relative_to(ROOT)): module.file_sha256(p)
                                   for p in [Path(__file__), ROOT / 'experiments/multilane_open_lane_change/run.py',
                                             ROOT / 'src/highway_transition_timing/double_dqn.py',
                                             ROOT / 'src/highway_transition_timing/rewards.py']},
                    versions={p: importlib.metadata.version(p) for p in
                              ['highway-env', 'gymnasium', 'stable-baselines3', 'torch']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.mode == 'old-policy':
        evaluate_old(args, out, cf, module)
    else:
        train_one(args, out, module)
    (out / 'complete.json').write_text(json.dumps({'complete': True}))


if __name__ == '__main__':
    main()
