import unittest

from highway_transition_timing.highway_adapter import (
    dqn_class_for_variant,
    q_value_diagnostics,
)


class HighwayAdapterTests(unittest.TestCase):
    def test_unknown_dqn_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            dqn_class_for_variant("not-a-dqn")

    def test_legacy_single_dqn_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            dqn_class_for_variant("dqn")

    def test_q_value_diagnostics_reports_action_margins(self):
        summary = q_value_diagnostics([1.0, 2.0, 3.5, 3.0, 0.5])

        self.assertEqual(summary["q_argmax_action"], "LANE_RIGHT")
        self.assertEqual(summary["q_top_margin"], 0.5)
        self.assertEqual(summary["lane_change_q_advantage"], 0.5)
        self.assertEqual(summary["slower_q_advantage"], -3.0)


if __name__ == "__main__":
    unittest.main()
