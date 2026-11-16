#!/usr/bin/env python3
"""Development-only clip/onset coverage; no training or human-study scoring.

Keep the historical P0 per-seed budget (five clips), both HIGHLIGHTS variants,
and both leader directions for each DISAGREEMENTS pair. Arm D is the independent
distance-affine-v1 experiment, not a relabelled original Arm C.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p0_selection_alignment import ARMS_BY_KEY, onset_by_episode, steps_by_agent
from p1_replay_check import ARMS as REPLAY_ARMS, config_for_arm, load_experiment_module, compare_step
from highway_transition_timing.summaries import context_window, select_highlights, state_importance

AGENTS = ('FD', 'BAL', 'SP')
VARIANTS = {'FD': 'F1', 'BAL': 'F4', 'SP': 'F7'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def run_path(root, arm, seed, agent):
    if arm == 'D':
        return root / f'arm_c_distance-affine-v1_{VARIANTS[agent]}_100k_seed{seed}'
    return root / f'{ARMS_BY_KEY[arm].run_prefix}{seed}'


def eligible(path, arm, agent):
    if arm == 'D':
        status = json.loads((path / 'status.json').read_text())
        gate_failure = any(message in status.get('error', '') for message in
                           ('No eligible checkpoint', 'No checkpoint passed the eligibility gate'))
        if status['status'] == 'failed' and gate_failure:
            selection = read_csv(path / 'model_selection.csv')
            if not selection or any(r['agent_condition'] != agent or
                    r['eligible'].lower() == 'true' or r['selected'].lower() == 'true'
                    for r in selection):
                raise ValueError(f'Inconsistent failed-gate metadata: {path}')
            return False
        if status['status'] != 'complete' or not status.get('scientific_result'):
            raise ValueError(f'Incomplete/non-scientific run: {path}')
    if not (path / 'models' / f'{agent}_main.zip').exists():
        raise FileNotFoundError(path / 'models' / f'{agent}_main.zip')
    return True


def get_onsets(path, arm):
    if arm != 'D':
        return onset_by_episode(path, ARMS_BY_KEY[arm])
    return {r['episode_id']: int(float(r['target_onset_t']))
            for r in read_csv(path / 'analysis' / 'episode_outcomes.csv')
            if r['analysis_target'] == 'lane_change_onset' and r['episode_outcome'] == 'valid_onset'}


def coverage_row(arm, seed, leader, follower, method, episode, step, length, onset, score,
                 native_start=None, native_end=None):
    window = context_window(step, length)
    return dict(arm=arm, seed=seed, leader=leader, follower=follower, method=method,
                episode_id=episode, step=step, importance=score, episode_length=length,
                onset_t='' if onset is None else onset,
                window_start=window.start, window_end_exclusive=window.stop,
                onset_in_context_window=onset is not None and onset in window,
                native_start='' if native_start is None else native_start,
                native_end_inclusive='' if native_end is None else native_end,
                onset_in_native_window='' if native_start is None else (
                    onset is not None and native_start <= onset <= native_end))


def select_disagreements(candidates, budget=5, overlap_limit=5):
    """Reference greedy union-overlap rule, stable ties and discarded fallback."""
    selected, discarded = [], []
    seen = defaultdict(set)
    for row in sorted(candidates, key=lambda r: -r['importance']):
        indices = set(range(row['native_start'], row['native_end_inclusive'] + 1))
        if len(seen[row['episode_id']] & indices) > overlap_limit:
            discarded.append(row)
            continue
        selected.append(row)
        seen[row['episode_id']].update(indices)
        if len(selected) == budget:
            break
    return (selected + discarded)[:budget]


def last_state_importance(leader_q, branch_q, low, high, same_image=False):
    """Both policies evaluate both terminal states, with trace-wise scaling."""
    if same_image:
        return 0.0
    scales = [hi - lo for lo, hi in zip(low, high)]
    if any(scale <= 0 for scale in scales):
        raise ValueError('Undefined constant-Q min-max normalization')
    return abs(sum((max(a) - max(b)) / scale
                   for a, b, scale in zip(leader_q, branch_q, scales)))


def load_scene(arm, seed, leader, path):
    if arm == 'D':
        extension = load_experiment_module('arm_c_distance_variant')
        module = extension.load_base()
        config = extension.config_for(module, leader, seed)
        extension.install(module)
        # Refuse silently reconstructed settings inconsistent with the run.
        from dataclasses import asdict
        recorded = json.loads((path / 'protocol.json').read_text())['config']
        if json.loads(json.dumps(asdict(config))) != recorded:
            raise ValueError('Arm D recorded configuration differs from reconstruction')
        return module, config, module.apply_open_lane_scene
    arm_spec = REPLAY_ARMS[arm]
    module = load_experiment_module(arm_spec.experiment)
    return module, config_for_arm(module, arm_spec, path), getattr(module, arm_spec.apply_scene)


def replay_candidates(module, config, apply_scene, spec, leader, models, stored, onsets, arm, seed):
    """Verify full leader replay, then fork each disagreement for ten steps.

    Native last-state scoring includes the terminal state, matching upstream.
    Coverage is measured on the real leader episode, never by transferring its
    onset to the synthetic follower branch. Common-window coverage matches P0;
    native-window coverage is a separately labelled sensitivity check.
    """
    import numpy as np
    import torch
    from highway_transition_timing.highway_adapter import action_name_from_env, extract_diagnostics
    followers = [a for a in models if a != leader]
    episode = f'{leader}_{spec.exposure_id}_r0'
    saved = {int(r['t']): r for r in stored if r['episode_id'] == episode}
    if not saved or set(saved) != set(range(len(saved))):
        raise ValueError(f'Missing/noncontiguous stored episode {episode}')

    def query(agent, obs):
        with torch.inference_mode():
            tensor, _ = models[agent].policy.obs_to_tensor(obs)
            return models[agent].q_net(tensor).cpu().numpy()[0]

    def frame_digest(env):
        # Preserve upstream's image-equality guard, without storing frame arrays.
        env.unwrapped.render_mode = 'rgb_array'
        env.unwrapped.config['offscreen_rendering'] = True
        frame = env.unwrapped.render()
        digest = hashlib.sha256(frame.tobytes()).digest()
        # A pygame viewer cannot be deep-copied; no simulator state is changed.
        if env.unwrapped.viewer is not None:
            env.unwrapped.viewer.close()
            env.unwrapped.viewer = None
        env.unwrapped.enable_auto_render = False
        env.unwrapped.render_mode = None
        return digest

    # Evaluation wrappers only recalculate rewards; no rewards are consumed
    # here. Copy the simulator itself (custom wrappers recurse on deepcopy).
    env = module.make_env(leader, config, training=False).unwrapped
    q_values, images, branches = [], [], defaultdict(list)
    action_ties = 0
    try:
        env.reset(seed=spec.exposure_seed)
        obs = apply_scene(env, spec)
        for t in range(len(saved) + 1):
            q = {a: query(a, obs) for a in models}
            q_values.append(q)
            images.append(frame_digest(env))
            if t == len(saved):
                break
            action = int(np.argmax(q[leader]))
            mismatches = compare_step(saved[t], action_name_from_env(env, action), extract_diagnostics(env))
            if mismatches:
                # Keep stored actions to reproduce the corpus; only numerical
                # argmax ties are admissible, and they remain in the audit.
                names = env.unwrapped.action_type.actions
                recorded_action = next(i for i, name in names.items() if name == saved[t]['action'])
                if mismatches != ['action'] or abs(float(q[leader][action] - q[leader][recorded_action])) > 1e-5:
                    raise ValueError(f'Replay mismatch {episode}/{t}: {mismatches}')
                action_ties += 1
                action = recorded_action
            for follower in followers:
                if action == int(np.argmax(q[follower])):
                    continue
                branch = deepcopy(env)
                branch_obs = obs.copy()
                branch_min, branch_max = float('inf'), 0.0
                end_t = t
                aligned_q, aligned_image = None, None
                try:
                    for j in range(1, 11):
                        bq = query(follower, branch_obs)
                        branch_obs, _, terminated, truncated, _ = branch.step(int(np.argmax(bq)))
                        end_t = t + j
                        after = query(follower, branch_obs)
                        branch_min = min(branch_min, float(after.min()))
                        branch_max = max(branch_max, float(after.max()))
                        # If the original leader ended first, the upstream
                        # leader slice is shorter than the counterfactual.
                        # Score the follower at that same index, while still
                        # retaining its full rollout for normalization bounds.
                        if j == min(10, len(saved) - t) or terminated or truncated:
                            if aligned_q is None:
                                aligned_q = {a: query(a, branch_obs) for a in (leader, follower)}
                                aligned_image = frame_digest(branch)
                        if terminated or truncated:
                            break
                    # Upstream aligns the leader endpoint to the follower
                    # branch length (which may stop early after collision).
                    branches[follower].append(dict(step=t, end=end_t,
                        q=aligned_q, image=aligned_image, low=branch_min, high=branch_max))
                finally:
                    branch.close()
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated or truncated) != (t == len(saved) - 1):
                raise ValueError(f'Replay termination mismatch {episode}/{t}')
    finally:
        env.close()
    result = {}
    for follower in followers:
        low = [min(float(q[a].min()) for q in q_values) for a in (leader, follower)]
        high = [max(0.0, max(float(q[a].max()) for q in q_values)) for a in (leader, follower)]
        for b in branches[follower]:
            low[1] = min(low[1], b['low'])
            high[1] = max(high[1], b['high'])
        rows = []
        for b in branches[follower]:
            end = min(b['end'], len(saved))
            score = last_state_importance([q_values[end][a] for a in (leader, follower)],
                [b['q'][a] for a in (leader, follower)], low, high, images[end] == b['image'])
            rows.append(coverage_row(arm, seed, leader, follower, 'disagreements_last_state',
                episode, b['step'], len(saved), onsets.get(episode), score,
                max(0, b['step'] - 9), end))
        result[follower] = rows
    audit = dict(arm=arm, seed=seed, leader=leader, episode_id=episode,
                 stored_steps=len(saved), replayed_steps=len(q_values)-1,
                 numerical_action_ties=action_ties, verified=True)
    return result, audit


def run(args):
    from highway_transition_timing.highway_adapter import configure_headless_runtime, dqn_class_for_variant
    import torch
    torch.set_num_threads(1)
    configure_headless_runtime()
    out = args.out / f'arm{args.arm}_seed{args.seed}_{args.leader}'
    out.mkdir(parents=True, exist_ok=True)
    paths = {a: run_path(args.outputs_root, args.arm, args.seed, a) for a in AGENTS}
    available = {a: p for a, p in paths.items() if eligible(p, args.arm, a)}
    if args.leader not in available:
        (out / 'status.json').write_text(json.dumps(dict(status='ineligible', arm=args.arm,
            seed=args.seed, leader=args.leader, reason='No eligible checkpoint'), indent=2))
        return
    path = available[args.leader]
    inputs = {str(p.relative_to(args.outputs_root)): sha256(p)
              for a, d in available.items()
              for p in (d / 'models' / f'{a}_main.zip',)}
    outcome_dir = 'analysis' if args.arm == 'D' else ARMS_BY_KEY[args.arm].analysis_dir
    for p in (path / 'evaluation/steps.csv', path / outcome_dir / 'episode_outcomes.csv'):
        inputs[str(p.relative_to(args.outputs_root))] = sha256(p)
    source_paths = [Path(__file__), ROOT / 'scripts/p1_replay_check.py',
        ROOT / 'scripts/p0_selection_alignment.py'] + list((ROOT / 'src/highway_transition_timing').glob('*.py'))
    source_paths += [ROOT / 'experiments' / e / 'run.py' for e in
                     ('single_lane_slow_front', 'multilane_open_lane_change', 'arm_c_distance_variant', 'larger_target_gap')]
    manifest = dict(inputs=inputs, sources={str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                    arm=args.arm, seed=args.seed, leader=args.leader, max_exposures=args.max_exposures)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Input/source manifest changed; use a fresh --out directory')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    onsets = get_onsets(path, args.arm)
    per_agent, lengths = steps_by_agent(path)
    states = per_agent[args.leader]
    selections = []
    for variant in ('max_min', 'max_second'):
        scored = [((e, t), state_importance(q, variant)) for e, t, q in states]
        importance = dict(scored)
        for e, t in select_highlights(scored):
            selections.append(coverage_row(args.arm, args.seed, args.leader, '',
                'highlights_' + variant, e, t, lengths[e], onsets.get(e), importance[e, t]))
    write_csv(out / 'highlights.csv', selections)
    if args.highlights_only:
        return
    module, config, apply_scene = load_scene(args.arm, args.seed, args.leader, path)
    models = {a: dqn_class_for_variant(config.dqn_variant).load(p / 'models' / f'{a}_main.zip', device='cpu')
              for a, p in available.items()}
    stored = read_csv(path / 'evaluation' / 'steps.csv')
    specs = module.make_eval_specs(36, config)
    if args.max_exposures:
        specs = specs[:args.max_exposures]
    all_candidates, audit = defaultdict(list), []
    start = time.monotonic()
    for i, spec in enumerate(specs):
        cache = out / f'episode_{spec.exposure_id}.json'
        if cache.exists():
            payload = json.loads(cache.read_text())
            candidates, record = payload['candidates'], payload['audit']
        else:
            candidates, record = replay_candidates(module, config, apply_scene, spec,
                args.leader, models, stored, onsets, args.arm, args.seed)
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(candidates=candidates, audit=record)))
            temporary.replace(cache)
        audit.append(record)
        for follower, rows in candidates.items():
            all_candidates[follower].extend(rows)
        print(f'{args.arm}/{args.seed}/{args.leader} {i+1}/{len(specs)} episodes; '
              f'{time.monotonic()-start:.1f}s', flush=True)
    for follower, candidates in all_candidates.items():
        selections.extend(select_disagreements(candidates))
    write_csv(out / 'selections.csv', selections)
    write_csv(out / 'replay_audit.csv', audit)
    write_csv(out / 'candidate_counts.csv', [dict(arm=args.arm, seed=args.seed,
        leader=args.leader, follower=f, candidates=len(rows), selected=len(select_disagreements(rows)))
        for f, rows in all_candidates.items()])
    (out / 'status.json').write_text(json.dumps(dict(status='complete', arm=args.arm,
        seed=args.seed, leader=args.leader, exposures=len(specs), partial=bool(args.max_exposures),
        available=list(available), elapsed_seconds=time.monotonic()-start), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=list('ABCD'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--leader', choices=AGENTS, required=True)
    p.add_argument('--outputs-root', type=Path, default=ROOT / 'outputs')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/summary_comparison/clip_onset_coverage')
    p.add_argument('--max-exposures', type=int)
    p.add_argument('--highlights-only', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
