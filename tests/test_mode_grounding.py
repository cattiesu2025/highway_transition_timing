import unittest

from highway_transition_timing.constants import (
    HIGH_SPEED_CRUISE,
    LANE_KEEPING_CRUISE,
    TRAFFIC_SPACING_ADJUSTMENT,
)
from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.mode_grounding import (
    bridge_short_interruptions,
    build_mode_table_for_episode,
)


class ModeGroundingTests(unittest.TestCase):
    def test_bridge_short_interruption(self):
        labels = [
            LANE_KEEPING_CRUISE,
            LANE_KEEPING_CRUISE,
            HIGH_SPEED_CRUISE,
            LANE_KEEPING_CRUISE,
            LANE_KEEPING_CRUISE,
        ]

        smoothed, status = bridge_short_interruptions(labels, max_gap=1)

        self.assertEqual(smoothed, [LANE_KEEPING_CRUISE] * 5)
        self.assertEqual(status[2], "bridged_gap_len_1")

    def test_does_not_bridge_long_interruption(self):
        labels = [
            LANE_KEEPING_CRUISE,
            HIGH_SPEED_CRUISE,
            HIGH_SPEED_CRUISE,
            LANE_KEEPING_CRUISE,
        ]

        smoothed, _ = bridge_short_interruptions(labels, max_gap=1)

        self.assertEqual(smoothed, labels)

    def test_does_not_bridge_through_terminal_none(self):
        labels = [
            TRAFFIC_SPACING_ADJUSTMENT,
            None,
            TRAFFIC_SPACING_ADJUSTMENT,
        ]

        smoothed, _ = bridge_short_interruptions(labels, max_gap=1)

        self.assertEqual(smoothed, labels)

    def test_idle_continues_high_speed_when_speed_stays_high(self):
        rows = [
            _step(0, "IDLE", ego_speed=28.8, speed_score=0.58),
            _step(1, "FASTER", ego_speed=29.3, speed_score=0.62),
            _step(2, "IDLE", ego_speed=29.3, speed_score=0.62),
        ]

        modes = build_mode_table_for_episode(rows, AnalysisConfig(bridge_max_gap=0))

        self.assertEqual(modes[1]["mode_label"], HIGH_SPEED_CRUISE)
        self.assertEqual(modes[2]["mode_label"], HIGH_SPEED_CRUISE)
        self.assertEqual(modes[2]["smoothing_status"], f"idle_continuation:{HIGH_SPEED_CRUISE}")

    def test_idle_continues_spacing_when_front_vehicle_remains_close(self):
        rows = [
            _step(0, "SLOWER", ego_speed=24.0, nearest_front_distance=14.0),
            _step(1, "IDLE", ego_speed=23.8, nearest_front_distance=13.5),
            _step(2, "IDLE", ego_speed=23.8, nearest_front_distance=13.0),
        ]

        modes = build_mode_table_for_episode(rows, AnalysisConfig(bridge_max_gap=0))

        self.assertEqual(
            [row["mode_label"] for row in modes],
            [TRAFFIC_SPACING_ADJUSTMENT] * 3,
        )
        self.assertEqual(
            modes[1]["smoothing_status"],
            f"idle_continuation:{TRAFFIC_SPACING_ADJUSTMENT}",
        )


def _step(
    t,
    action,
    ego_speed=25.0,
    speed_score=0.3,
    nearest_front_distance=80.0,
):
    return {
        "episode_id": "ep0",
        "agent_condition": "SP",
        "policy_id": "SP_main",
        "exposure_id": "H0000",
        "rollout_id": "r0",
        "rollout_seed": "0",
        "t": t,
        "ego_lane": 1,
        "ego_speed": ego_speed,
        "action": action,
        "nearest_front_distance": nearest_front_distance,
        "closest_k_distances": [nearest_front_distance, nearest_front_distance + 5],
        "speed_score": speed_score,
        "front_distance_score": min(1.0, nearest_front_distance / 60.0),
        "closest_k_distance_score": min(1.0, nearest_front_distance / 50.0),
        "collision_flag": False,
        "exposure_t": 0,
    }


if __name__ == "__main__":
    unittest.main()
