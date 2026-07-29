import unittest

from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.constants import (
    FIRST_STABLE_MODE_ONSET_TARGET,
    LANE_CHANGE_ONSET_TARGET,
    LANE_CHANGE_TARGET_LABEL,
    LANE_KEEPING_CRUISE,
    NO_ONSET_CENSORED,
    SLOWDOWN_ONSET_TARGET,
    SLOWDOWN_TARGET_LABEL,
    TERMINAL_FAILURE,
    TRAFFIC_SPACING_ADJUSTMENT,
    VALID_ONSET,
)
from highway_transition_timing.transitions import (
    classify_episode_outcomes,
    extract_transitions_for_episode,
)


class TransitionTests(unittest.TestCase):
    def test_transition_onset_and_confirmation(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = [_step("e0", t) for t in range(6)]
        modes = [
            _mode("e0", t, LANE_KEEPING_CRUISE if t < 3 else TRAFFIC_SPACING_ADJUSTMENT)
            for t in range(6)
        ]

        transitions = extract_transitions_for_episode(modes, steps, config)

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["mode_before"], LANE_KEEPING_CRUISE)
        self.assertEqual(transitions[0]["mode_after"], TRAFFIC_SPACING_ADJUSTMENT)
        self.assertEqual(transitions[0]["onset_t"], 3)
        self.assertEqual(transitions[0]["confirmation_t"], 5)
        self.assertEqual(transitions[0]["response_latency"], 3)

    def test_episode_outcomes_include_valid_censored_and_failure(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = (
            [_step("valid", t) for t in range(6)]
            + [_step("censored", t) for t in range(6)]
            + [_step("failure", t, collision=(t == 4)) for t in range(6)]
        )
        transitions = [
            {
                "transition_id": "valid:tr000",
                "episode_id": "valid",
                "mode_before": LANE_KEEPING_CRUISE,
                "mode_after": TRAFFIC_SPACING_ADJUSTMENT,
                "onset_t": 3,
                "confirmation_t": 5,
                "response_latency": 3,
            }
        ]

        outcomes = classify_episode_outcomes(steps, transitions, config)
        by_episode = {row["episode_id"]: row for row in outcomes}

        self.assertEqual(by_episode["valid"]["episode_outcome"], VALID_ONSET)
        self.assertEqual(by_episode["valid"]["response_latency"], 3)
        self.assertEqual(
            by_episode["censored"]["episode_outcome"], NO_ONSET_CENSORED
        )
        self.assertEqual(by_episode["failure"]["episode_outcome"], TERMINAL_FAILURE)
        self.assertTrue(by_episode["failure"]["collision_flag"])

    def test_first_stable_mode_onset_includes_latency_zero(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = [_step("immediate", t) for t in range(5)]
        modes = [
            _mode("immediate", t, TRAFFIC_SPACING_ADJUSTMENT)
            for t in range(5)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=FIRST_STABLE_MODE_ONSET_TARGET,
            mode_rows=modes,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_record_type"], "stable_mode_onset")
        self.assertEqual(outcomes[0]["target_mode_after"], TRAFFIC_SPACING_ADJUSTMENT)
        self.assertEqual(outcomes[0]["target_onset_t"], 0)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 2)
        self.assertEqual(outcomes[0]["response_latency"], 0)

    def test_mode_onset_target_can_select_specific_mode(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = [_step("specific", t) for t in range(6)]
        modes = [
            _mode("specific", t, LANE_KEEPING_CRUISE if t < 3 else TRAFFIC_SPACING_ADJUSTMENT)
            for t in range(6)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=f"mode_onset:{TRAFFIC_SPACING_ADJUSTMENT}",
            mode_rows=modes,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_mode_after"], TRAFFIC_SPACING_ADJUSTMENT)
        self.assertEqual(outcomes[0]["response_latency"], 3)

    def test_lane_change_onset_includes_latency_zero(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = [_step("lane", t, action="LANE_RIGHT") for t in range(5)]

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_record_type"], "stable_action_onset")
        self.assertEqual(outcomes[0]["target_mode_after"], LANE_CHANGE_TARGET_LABEL)
        self.assertEqual(outcomes[0]["target_onset_t"], 0)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 2)
        self.assertEqual(outcomes[0]["response_latency"], 0)

    def test_lane_change_onset_waits_for_stable_action_run(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = (
            [_step("delayed_lane", t, action="IDLE") for t in range(2)]
            + [_step("delayed_lane", t, action="LANE_RIGHT") for t in range(2, 6)]
        )

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_onset_t"], 2)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 4)
        self.assertEqual(outcomes[0]["response_latency"], 2)

    def test_lane_change_onset_ignores_pre_exposure_action_run(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = (
            [
                _step("pre_exposure_lane", t, action="LANE_RIGHT", exposure_t=3)
                for t in range(3)
            ]
            + [
                _step("pre_exposure_lane", t, action="IDLE", exposure_t=3)
                for t in range(3, 6)
            ]
        )

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], NO_ONSET_CENSORED)

    def test_lane_change_onset_resegments_at_exposure(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = [
            _step("resegmented_lane", t, action="LANE_RIGHT", exposure_t=3)
            for t in range(6)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_onset_t"], 3)
        self.assertEqual(outcomes[0]["response_latency"], 0)

    def test_slowdown_onset_uses_stable_slow_action(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        steps = (
            [_step("slow_action", t, action="IDLE", ego_speed=30.0) for t in range(2)]
            + [_step("slow_action", t, action="SLOWER", ego_speed=30.0) for t in range(2, 6)]
        )

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=SLOWDOWN_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_mode_after"], SLOWDOWN_TARGET_LABEL)
        self.assertEqual(outcomes[0]["target_onset_t"], 2)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 4)
        self.assertEqual(outcomes[0]["response_latency"], 2)

    def test_slowdown_onset_uses_stable_speed_drop(self):
        config = AnalysisConfig(persistence_k=3, bridge_max_gap=0)
        speeds = [30.0, 30.0, 29.5, 29.0, 28.5, 28.0]
        steps = [
            _step("speed_drop", t, action="IDLE", ego_speed=speed)
            for t, speed in enumerate(speeds)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            transition_rows=[],
            config=config,
            analysis_target=SLOWDOWN_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_mode_after"], SLOWDOWN_TARGET_LABEL)
        self.assertEqual(outcomes[0]["target_onset_t"], 2)
        self.assertEqual(outcomes[0]["response_latency"], 2)


def _step(
    episode_id,
    t,
    collision=False,
    action="IDLE",
    ego_lane=1,
    ego_speed=28.0,
    exposure_t=0,
):
    return {
        "episode_id": episode_id,
        "agent_condition": "FD",
        "policy_id": "FD_main",
        "exposure_id": "E0",
        "rollout_id": "r0",
        "rollout_seed": "0",
        "t": t,
        "action": action,
        "ego_lane": ego_lane,
        "ego_speed": ego_speed,
        "collision_flag": collision,
        "termination_reason": "collision" if collision else "",
        "exposure_t": exposure_t,
    }


def _mode(episode_id, t, label):
    return {
        "episode_id": episode_id,
        "agent_condition": "FD",
        "policy_id": "FD_main",
        "exposure_id": "E0",
        "rollout_id": "r0",
        "rollout_seed": "0",
        "t": t,
        "mode_label": label,
        "mode_confidence": 0.9,
    }


if __name__ == "__main__":
    unittest.main()
