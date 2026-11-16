"""Distance-only invariants against the unchanged source sampler and environment."""
import importlib.util
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('distance_variant', ROOT / 'experiments/arm_c_distance_variant/run.py')
variant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant)


@pytest.mark.parametrize('seed', [4200, 4209, 4219])
def test_only_distances_change_and_source_quota_is_preserved(seed):
    module = variant.load_base()
    config = variant.config_for(module, 'BAL', seed)
    original = [module.make_training_spec(i, config) for i in range(200)]
    module.validate_stratified_training_specs(original)
    variant.install(module)
    changed = [module.make_training_spec(i, config) for i in range(200)]
    allowed = {'front_distance', 'target_lane_gap', 'ttc_bin', 'required_deceleration_bin', 'visible_at_t0'}
    for i, (before, after) in enumerate(zip(original, changed)):
        b, a = asdict(before), asdict(after)
        assert {k: v for k, v in b.items() if k not in allowed} == {k: v for k, v in a.items() if k not in allowed}
        if before.include_front_vehicle:
            assert 40 <= after.front_distance <= 150
            assert after.front_distance + 30 <= after.target_lane_gap <= 195
            assert after.ttc_bin == module.classify_ttc(after.ttc_seconds, after.closing_speed)
        else:
            assert before == after and after.target_lane_gap is None
        row = module.training_spec_row(i, after, config)
        assert row['source_ttc_bin'] == before.ttc_bin
    for start in range(0, 200, 20):
        assert Counter(s.scenario_type for s in original[start:start+20]) == Counter(s.scenario_type for s in changed[start:start+20])


def test_reward_conditions_share_scenes_and_configs_match_baseline():
    module = variant.load_base()
    variant.install(module)
    samples = []
    for agent in variant.WEIGHTS:
        config = variant.config_for(module, agent, 4200)
        baseline = module.ExperimentConfig(duration=20, evaluation_duration=120, seed=4200,
            target_lane_vehicle=True, speed_weight=variant.WEIGHTS[agent][1],
            front_distance_weight=variant.WEIGHTS[agent][2])
        assert config == baseline
        samples.append([module.make_training_spec(i, config) for i in range(40)])
    assert samples[0] == samples[1] == samples[2]


def test_eval_preserves_source_pairing_and_old_module_is_unchanged():
    old = variant.load_base()
    config = variant.config_for(old, 'FD', 4200)
    expected = old.make_eval_specs(36, config)
    new = variant.load_base()
    variant.install(new)
    actual = new.make_eval_specs(36, config)
    assert old.make_eval_specs(36, config) == expected
    assert len({(s.front_distance, s.front_speed, s.target_lane_gap) for s in actual}) == 36
    for a, b in zip(expected, actual):
        assert (a.ego_speed, a.front_speed, a.exposure_seed, a.exposure_id) == (b.ego_speed, b.front_speed, b.exposure_seed, b.exposure_id)
        assert b.target_lane_gap > b.front_distance
    cf = new.load_rollout_counterfactual_module()
    assert cf.counterfactual_settings(actual[0], 'no-front')['target_lane_gap'] == actual[0].target_lane_gap
    assert cf.counterfactual_settings(actual[0], 'far-front')['front_distance'] == 360


def test_geometry_endpoints_and_reject_outside_support():
    module = variant.load_base()
    config = variant.config_for(module, 'SP', 4200)
    source = module.make_eval_specs(1, config)[0]
    for d, target, expected in [(90, 10, (40, 70)), (195, 90, (150, 195))]:
        changed = variant.transform(module, replace(source, front_distance=d, target_lane_gap=target))
        assert (changed.front_distance, changed.target_lane_gap) == expected
    with pytest.raises(ValueError):
        variant.transform(module, replace(source, front_distance=200))


def test_dry_run_does_not_create_output_and_missing_baseline_is_explicit(tmp_path):
    assert variant.main(['--out-root', str(tmp_path), '--baseline-root', str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []
    assert variant.baseline_status(tmp_path, 'FD', 4200)['status'] == 'missing'


def test_audit_status_and_overwrite_protection(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(variant, 'preflight', lambda *args: {'preflight_scenes': 57})
    args = ['--out-root', str(tmp_path), '--baseline-root', str(tmp_path), '--execute']
    assert variant.main(args) == 0
    directory = next(tmp_path.iterdir())
    status = json.loads((directory / 'status.json').read_text())
    assert status['status'] == 'complete'
    assert status['paired_resets'] == 2000
    assert status['scientific_result'] is False
    with pytest.raises(FileExistsError):
        variant.main(args)


def test_failed_gate_never_runs_final_evaluation(tmp_path, monkeypatch):
    import json
    module = variant.load_base()
    monkeypatch.setattr(variant, 'load_base', lambda: module)
    monkeypatch.setattr(variant, 'audit', lambda *args: {})
    monkeypatch.setattr(variant, 'preflight', lambda *args: {})
    monkeypatch.setattr(module, 'train_agents', lambda *args, **kwargs: None)
    def failed_gate(*args):
        raise RuntimeError('No checkpoint passed eligibility')
    monkeypatch.setattr(module, 'select_models', failed_gate)
    def forbidden_evaluation(*args):
        pytest.fail('Evaluation must not run after failed selection')
    monkeypatch.setattr(module, 'evaluate_agents', forbidden_evaluation)
    with pytest.raises(RuntimeError, match='No checkpoint passed'):
        variant.main(['--mode', 'train', '--out-root', str(tmp_path), '--execute'])
    status = json.loads(next(tmp_path.glob('*/status.json')).read_text())
    assert status['status'] == 'failed'
    assert status['scientific_result'] is False
