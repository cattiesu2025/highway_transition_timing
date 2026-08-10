import unittest
from pathlib import Path

from highway_transition_timing.model_selection import (
    GATE_VARIANTS,
    MIN_VALID_ONSETS,
    VariantOutcome,
    available_checkpoints,
    checkpoint_path,
    evaluate_gate,
    select_latest_eligible_checkpoint,
    variant_outcome,
)


def _outcomes(
    valid=36,
    terminal=0,
    original_median=2.0,
    no_front_valid=0,
    control_median=8.0,
    control_valid=36,
):
    return {
        "original": VariantOutcome(valid, 36 - valid - terminal, terminal, original_median),
        "no-front": VariantOutcome(no_front_valid, 36 - no_front_valid, 0, None),
        "matched-speed-front": VariantOutcome(control_valid, 0, 0, control_median),
        "far-front": VariantOutcome(control_valid, 0, 0, control_median),
    }


class GateTests(unittest.TestCase):
    def test_healthy_checkpoint_is_eligible(self):
        gate = evaluate_gate(_outcomes())

        self.assertTrue(gate.eligible)
        self.assertTrue(all(gate.criteria.values()))

    def test_collisions_above_ceiling_fail_safety(self):
        gate = evaluate_gate(_outcomes(valid=15, terminal=21))

        self.assertFalse(gate.eligible)
        self.assertFalse(gate.criteria["safety"])

    def test_too_few_onsets_fail_estimability(self):
        gate = evaluate_gate(_outcomes(valid=MIN_VALID_ONSETS - 1))

        self.assertFalse(gate.eligible)
        self.assertFalse(gate.criteria["estimability"])
        self.assertTrue(gate.criteria["safety"])

    def test_no_front_onset_fails_specificity(self):
        gate = evaluate_gate(_outcomes(no_front_valid=1))

        self.assertFalse(gate.eligible)
        self.assertFalse(gate.criteria["no_front_specificity"])

    def test_control_without_delay_fails_specificity(self):
        gate = evaluate_gate(_outcomes(original_median=2.0, control_median=2.5))

        self.assertFalse(gate.eligible)
        self.assertFalse(gate.criteria["matched_speed_front_specificity"])
        self.assertFalse(gate.criteria["far_front_specificity"])

    def test_control_without_onsets_passes_specificity(self):
        gate = evaluate_gate(_outcomes(control_valid=0, control_median=None))

        self.assertTrue(gate.eligible)

    def test_missing_variant_is_rejected(self):
        outcomes = _outcomes()
        del outcomes["far-front"]

        with self.assertRaises(ValueError):
            evaluate_gate(outcomes)

    def test_gate_covers_every_declared_variant(self):
        self.assertEqual(set(GATE_VARIANTS), set(_outcomes()))

    def test_thresholds_scale_to_a_smaller_grid(self):
        # A six-exposure smoke grid keeps the 36-scene rates: no terminal
        # failure is tolerated and at least ceil(27/36 * 6) = 5 onsets are
        # required.
        outcomes = {
            "original": VariantOutcome(5, 1, 0, 2.0),
            "no-front": VariantOutcome(0, 6, 0, None),
            "matched-speed-front": VariantOutcome(6, 0, 0, 8.0),
            "far-front": VariantOutcome(6, 0, 0, 8.0),
        }

        self.assertTrue(evaluate_gate(outcomes).eligible)

        outcomes["original"] = VariantOutcome(4, 1, 1, 2.0)
        gate = evaluate_gate(outcomes)

        self.assertFalse(gate.criteria["safety"])
        self.assertFalse(gate.criteria["estimability"])

    def test_thresholds_match_the_approved_counts_on_the_full_grid(self):
        self.assertTrue(evaluate_gate(_outcomes(valid=27, terminal=1)).eligible)
        self.assertFalse(evaluate_gate(_outcomes(valid=26, terminal=1)).eligible)
        self.assertFalse(evaluate_gate(_outcomes(valid=27, terminal=2)).eligible)

    def test_empty_original_variant_is_rejected(self):
        outcomes = _outcomes()
        outcomes["original"] = VariantOutcome(0, 0, 0, None)

        with self.assertRaises(ValueError):
            evaluate_gate(outcomes)


class VariantOutcomeTests(unittest.TestCase):
    def test_counts_and_median_use_only_the_requested_agent(self):
        rows = [
            {"agent_condition": "FD", "episode_outcome": "valid_onset", "response_latency": 5},
            {"agent_condition": "FD", "episode_outcome": "valid_onset", "response_latency": 15},
            {"agent_condition": "FD", "episode_outcome": "no_onset_censored"},
            {"agent_condition": "FD", "episode_outcome": "terminal_failure"},
            {"agent_condition": "SP", "episode_outcome": "valid_onset", "response_latency": 99},
        ]

        outcome = variant_outcome(rows, "FD", policy_step_seconds=0.2)

        self.assertEqual(outcome.valid_onsets, 2)
        self.assertEqual(outcome.censored, 1)
        self.assertEqual(outcome.terminal_failures, 1)
        self.assertAlmostEqual(outcome.median_onset_seconds, 2.0)

    def test_median_is_none_without_valid_onsets(self):
        rows = [{"agent_condition": "FD", "episode_outcome": "no_onset_censored"}]

        self.assertIsNone(variant_outcome(rows, "FD", 0.2).median_onset_seconds)


class SelectionTests(unittest.TestCase):
    def _checkpoints(self, steps):
        return [(step, Path(f"/tmp/BAL_step_{step:06d}.zip")) for step in steps]

    def test_latest_eligible_checkpoint_stops_the_walk(self):
        eligible_steps = {5_000, 10_000, 15_000}
        examined = []

        def evaluate(path):
            step = int(path.stem.rsplit("_", 1)[-1])
            examined.append(step)
            return _outcomes(valid=36 if step in eligible_steps else 0)

        selected, records = select_latest_eligible_checkpoint(
            "BAL",
            self._checkpoints([5_000, 10_000, 15_000, 20_000]),
            evaluate,
        )

        self.assertEqual(selected[0], 15_000)
        # Walks newest first and stops as soon as one passes.
        self.assertEqual(examined, [20_000, 15_000])
        self.assertEqual([record["checkpoint_step"] for record in records], [20_000, 15_000])
        self.assertEqual([record["selected"] for record in records], [False, True])

    def test_final_checkpoint_is_kept_when_it_passes(self):
        selected, records = select_latest_eligible_checkpoint(
            "FD",
            self._checkpoints([5_000, 10_000]),
            lambda path: _outcomes(),
        )

        self.assertEqual(selected[0], 10_000)
        self.assertEqual(len(records), 1)

    def test_no_eligible_checkpoint_returns_none_with_full_audit(self):
        selected, records = select_latest_eligible_checkpoint(
            "SP",
            self._checkpoints([5_000, 10_000]),
            lambda path: _outcomes(valid=0, terminal=36),
        )

        self.assertIsNone(selected)
        self.assertEqual(len(records), 2)
        self.assertFalse(any(record["selected"] for record in records))

    def test_empty_checkpoint_list_is_rejected(self):
        with self.assertRaises(ValueError):
            select_latest_eligible_checkpoint("FD", [], lambda path: _outcomes())

    def test_records_carry_per_variant_counts(self):
        _selected, records = select_latest_eligible_checkpoint(
            "FD",
            self._checkpoints([5_000]),
            lambda path: _outcomes(),
        )

        self.assertEqual(records[0]["original_valid_onsets"], 36)
        self.assertEqual(records[0]["no_front_valid_onsets"], 0)
        self.assertEqual(records[0]["far_front_median_onset_seconds"], 8.0)


class CheckpointDiscoveryTests(unittest.TestCase):
    def test_checkpoints_are_sorted_and_filtered_by_agent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            for agent, step in (("FD", 10_000), ("FD", 5_000), ("BAL", 5_000)):
                checkpoint_path(directory, agent, step).write_text("")
            (directory / "FD_step_notanumber.zip").write_text("")

            found = available_checkpoints(directory, "FD")

        self.assertEqual([step for step, _path in found], [5_000, 10_000])


if __name__ == "__main__":
    unittest.main()
