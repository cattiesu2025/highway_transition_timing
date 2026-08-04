import unittest

from highway_transition_timing.highway_adapter import (
    closing_collision_risk_score,
    reward_components,
    weighted_reward,
)
from highway_transition_timing.rewards import (
    RewardWeights,
    reward_config_table_with_strength_multiplier,
    reward_config_table_with_slow_down_penalty,
    with_reward_strength_multiplier,
)


class RewardTests(unittest.TestCase):
    def test_collision_risk_score_responds_to_closing_speed(self):
        self.assertGreater(closing_collision_risk_score(50.0, 30.0, 20.0), 0.0)
        self.assertEqual(closing_collision_risk_score(50.0, 20.0, 20.0), 0.0)
        self.assertGreater(closing_collision_risk_score(10.0, 20.0, 20.0), 0.0)

    def test_collision_risk_penalty_reduces_reward_before_crash(self):
        components = reward_components(
            {
                "speed_score": 0.0,
                "front_distance_score": 0.0,
                "collision_risk_score": 0.5,
                "collision_flag": False,
            }
        )
        weights = RewardWeights(
            speed_score=0.0,
            front_distance_score=0.0,
            collision_penalty=0.0,
            lane_change_penalty=0.0,
            collision_risk_penalty=2.0,
        )

        self.assertEqual(components["collision_penalty"], 0.0)
        self.assertEqual(components["collision_risk_penalty"], -0.5)
        self.assertAlmostEqual(weighted_reward(components, weights), -1.0)

    def test_reward_config_records_common_collision_risk_penalty(self):
        rows = reward_config_table_with_slow_down_penalty(
            common_slow_down_penalty=0.2,
            common_collision_risk_penalty=1.5,
        )

        self.assertEqual({row["slow_down_penalty"] for row in rows}, {0.2})
        self.assertEqual({row["collision_risk_penalty"] for row in rows}, {1.5})

    def test_reward_config_can_override_common_collision_penalty_only(self):
        rows = reward_config_table_with_slow_down_penalty(
            common_slow_down_penalty=0.0,
            common_collision_risk_penalty=3.0,
            common_collision_penalty=10.0,
        )

        self.assertEqual({row["collision_penalty"] for row in rows}, {10.0})
        self.assertEqual(
            {row["speed_score"] for row in rows},
            {0.45, 0.70, 1.00},
        )
        self.assertEqual(
            {row["front_distance_score"] for row in rows},
            {1.00, 0.70, 0.25},
        )

    def test_reward_config_can_override_common_lane_change_penalty(self):
        rows = reward_config_table_with_slow_down_penalty(
            common_slow_down_penalty=0.2,
            common_collision_risk_penalty=3.0,
            common_lane_change_penalty=0.2,
        )

        self.assertEqual({row["lane_change_penalty"] for row in rows}, {0.2})
        self.assertEqual(
            {row["front_distance_score"] for row in rows},
            {1.00, 0.70, 0.25},
        )

    def test_reward_strength_scales_preference_terms_and_bal_control(self):
        base = RewardWeights(
            speed_score=0.5,
            front_distance_score=1.0,
            collision_penalty=2.0,
            lane_change_penalty=0.1,
            slow_down_penalty=0.2,
            right_lane_score=0.3,
            collision_risk_penalty=3.0,
        )

        fd = with_reward_strength_multiplier("FD", base, 2.0)
        sp = with_reward_strength_multiplier("SP", base, 4.0)
        bal = with_reward_strength_multiplier("BAL", base, 2.0)

        self.assertEqual(fd.front_distance_score, 2.0)
        self.assertEqual(fd.speed_score, 0.5)
        self.assertEqual(fd.collision_risk_penalty, 3.0)
        self.assertEqual(sp.speed_score, 2.0)
        self.assertEqual(sp.front_distance_score, 1.0)
        self.assertEqual(sp.collision_penalty, 2.0)
        self.assertEqual(
            bal,
            RewardWeights(
                speed_score=1.0,
                front_distance_score=2.0,
                collision_penalty=4.0,
                lane_change_penalty=0.2,
                slow_down_penalty=0.4,
                right_lane_score=0.6,
                collision_risk_penalty=6.0,
            ),
        )

    def test_reward_strength_rejects_invalid_inputs(self):
        weights = RewardWeights(1.0, 1.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            with_reward_strength_multiplier("UNKNOWN", weights, 2.0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            with_reward_strength_multiplier("FD", weights, 0.0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            with_reward_strength_multiplier("FD", weights, float("nan"))

    def test_reward_strength_table_records_effective_weights_and_rule(self):
        rows = reward_config_table_with_strength_multiplier(
            common_slow_down_penalty=0.0,
            common_collision_risk_penalty=3.0,
            reward_strength_multiplier=2.0,
        )
        by_agent = {row["agent_condition"]: row for row in rows}

        self.assertEqual(by_agent["FD"]["front_distance_score"], 2.0)
        self.assertEqual(by_agent["FD"]["collision_risk_penalty"], 3.0)
        self.assertEqual(by_agent["SP"]["speed_score"], 2.0)
        self.assertEqual(by_agent["SP"]["collision_risk_penalty"], 3.0)
        self.assertEqual(by_agent["BAL"]["speed_score"], 1.4)
        self.assertEqual(by_agent["BAL"]["collision_risk_penalty"], 6.0)
        self.assertEqual(
            by_agent["BAL"]["reward_strength_rule"],
            "all_effective_reward_weights",
        )


if __name__ == "__main__":
    unittest.main()
