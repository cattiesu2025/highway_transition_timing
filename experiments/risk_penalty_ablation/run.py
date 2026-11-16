"""Isolated C/R6 risk-reward pilot. Does not alter maintained experiment defaults."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
MODES = ('baseline', 'early30', 'smooth6')
SEEDS = (4201, 4202)


def risk_score(distance, ego_speed, front_speed, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if mode == 'baseline':
        from highway_transition_timing.highway_adapter import closing_collision_risk_score
        return closing_collision_risk_score(distance, ego_speed, front_speed)
    if math.isinf(distance):
        return 0.0
    closing = max(0.0, ego_speed - (ego_speed if front_speed is None else front_speed))
    close = max(0.0, min(1.0, (24.0 - distance) / 24.0))
    risk = 0.0
    if closing > 0.1:
        if mode == 'smooth6':
            risk = 6.0 * closing / (distance + 6.0 * closing)
        else:
            threshold = 30.0 if mode == 'early30' else 6.0
            risk = max(0.0, min(1.0, 1.0 - distance / closing / threshold))
    return max(close, risk)


def load_modules():
    path = ROOT / 'experiments/multilane_open_lane_change/rollout_counterfactual.py'
    spec = importlib.util.spec_from_file_location('risk_ablation_counterfactual', path)
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf, cf.load_experiment_module()


def install_reward_and_logger(module, mode, output):
    # Only the privately loaded experiment module in this process is adapted.
    original_diagnostics = module.extract_diagnostics
    original_wrapper = module.OpenLaneRewardWrapper

    def diagnostics(env):
        result = original_diagnostics(env)
        result['collision_risk_score'] = risk_score(
            result['nearest_front_distance'], result['ego_speed'],
            result['front_vehicle_speed'], mode)
        return result

    module.extract_diagnostics = diagnostics
    fields = ['reset_index', 'scenario_type', 'episode_step', 'seconds', 'action',
              'risk_score', 'baseline_risk_score', 'risk_reward', 'reward_total',
              'collision', 'terminated', 'truncated']

    class LoggedReward(original_wrapper):
        def __init__(self, env, agent_condition, weights):
            super().__init__(env, agent_condition, weights)
            self.audit_handle = None
            if isinstance(env, module.OpenLaneTrainingResetWrapper):
                self.audit_handle = gzip.open(output / 'training_risk_steps.csv.gz', 'wt', newline='')
                self.audit_writer = csv.DictWriter(self.audit_handle, fieldnames=fields)
                self.audit_writer.writeheader()

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if self.audit_handle is not None:
                components = self.last_reward_components
                scene = self.env.last_training_exposure
                self.audit_writer.writerow(dict(
                    reset_index=self.env.reset_count - 1, scenario_type=scene.scenario_type,
                    episode_step=self.env.episode_step_count,
                    seconds=self.env.episode_step_count / self.env.config.policy_frequency,
                    action=module.action_name_from_env(self.env, int(action)),
                    risk_score=-components['collision_risk_penalty'],
                    baseline_risk_score=original_diagnostics(self.env)['collision_risk_score'],
                    risk_reward=self.weights.collision_risk_penalty * components['collision_risk_penalty'],
                    reward_total=reward, collision=bool(self.unwrapped.vehicle.crashed),
                    terminated=terminated, truncated=truncated))
                if terminated or truncated:
                    self.audit_handle.flush()
            return obs, reward, terminated, truncated, info

        def close(self):
            if self.audit_handle is not None:
                self.audit_handle.close()
                self.audit_handle = None
            return super().close()

    module.OpenLaneRewardWrapper = LoggedReward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(6), required=True)
    parser.add_argument('--out-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='200 training steps, 3 scenes, 4 s evaluation; no scientific claims')
    args = parser.parse_args()
    mode, seed = [(m, s) for m in MODES for s in SEEDS][args.index]
    name = f'risk_penalty_armC_R6_{mode}_100k_seed{seed}' + ('_smoke' if args.smoke else '')
    output = args.out_root.resolve() / name
    protocol = dict(mode=mode, seed=seed, arm='C', variant='R6', speed_weight=1.03,
                    front_distance_weight=.48, risk_weight=3., distance_threshold_m=24.,
                    ttc_threshold_or_scale_s=30. if mode == 'early30' else 6.,
                    smooth_formula='6*c/(d+6*c), c>0.1' if mode == 'smooth6' else None,
                    gamma=.95, training_duration_s=20, evaluation_duration_s=4 if args.smoke else 120,
                    timesteps=200 if args.smoke else 100000, evaluation_scenes=3 if args.smoke else 36,
                    primary_checkpoint='final_fixed_step', gate_role='audit_only_no_reselection',
                    smoke=args.smoke, output=str(output))
    print(json.dumps(protocol, indent=2), flush=True)
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cf, module = load_modules()
    module.configure_headless_runtime()
    install_reward_and_logger(module, mode, output)
    config = module.ExperimentConfig(seed=seed, duration=20, evaluation_duration=protocol['evaluation_duration_s'],
        lanes_count=2, ego_lane=1, target_lane_vehicle=True, speed_weight=1.03,
        front_distance_weight=.48, collision_risk_penalty=3., slow_down_penalty=.2, lane_change_penalty=.2)
    assert config.gamma == .95
    (output / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    import importlib.metadata
    source_paths = [Path(__file__), ROOT/'experiments/multilane_open_lane_change/run.py',
                    ROOT/'experiments/multilane_open_lane_change/rollout_counterfactual.py',
                    *sorted((ROOT/'src/highway_transition_timing').glob('*.py'))]
    provenance = dict(packages={n:importlib.metadata.version(n) for n in ['torch','stable-baselines3','highway-env','gymnasium']},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    module.train_agents(output, ['SP'], protocol['timesteps'], config, 0,
                        checkpoint_every=100 if args.smoke else 5000)
    model = module.dqn_class_for_variant(config.dqn_variant).load(output/'models/SP_main.zip', device='cpu')
    assert model.num_timesteps == protocol['timesteps']
    specs = module.make_eval_specs(protocol['evaluation_scenes'], config)
    summaries, gate = [], {}
    for variant in ['original', 'no-front', 'matched-speed-front', 'far-front']:
        steps, exposures = cf.rollout_variant(module, {'SP':model}, ['SP'], specs, config, variant)
        directory = output/'evaluation_controls'/variant
        module.write_csv_rows(directory/'steps.csv', steps)
        module.write_csv_rows(directory/'exposures.csv', exposures)
        episodes = module.summarize_actual_lane_changes(steps, config.policy_frequency)
        module.write_csv_rows(directory/'episode_summary.csv', episodes)
        summaries.extend(cf.summarize_variant(episodes, steps, variant))
        analysis = module.run_pipeline(steps, exposures, module.AnalysisConfig(bootstrap_samples=0), module.LANE_CHANGE_ONSET_TARGET)
        gate[variant] = module.variant_outcome(analysis.episode_outcomes, 'SP', config.policy_step_seconds)
    from highway_transition_timing.model_selection import evaluate_gate
    module.write_csv_rows(output/'control_summary.csv', summaries)
    (output/'final_checkpoint_gate.json').write_text(json.dumps(asdict(evaluate_gate(gate)),indent=2)+'\n')
    (output/'complete.json').write_text(json.dumps(dict(complete=True,smoke=args.smoke))+'\n')
    print('Completed', output, flush=True)

if __name__ == '__main__':
    main()
