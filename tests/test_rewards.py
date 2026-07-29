import unittest

from highway_transition_timing.highway_adapter import (
    closing_collision_risk_score,
    reward_components,
    weighted_reward,
)
from highway_transition_timing.rewards import (
    RewardWeights,
    reward_config_table_with_slow_down_penalty,
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


if __name__ == "__main__":
    unittest.main()
