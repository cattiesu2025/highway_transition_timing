import unittest

from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.constants import (
    LANE_CHANGE_ONSET_TARGET,
    LANE_CHANGE_TARGET_LABEL,
    NO_ONSET_CENSORED,
    SLOWDOWN_ONSET_TARGET,
    SLOWDOWN_TARGET_LABEL,
    TERMINAL_FAILURE,
    VALID_ONSET,
)
from highway_transition_timing.transitions import classify_episode_outcomes


class TransitionTests(unittest.TestCase):
    def test_episode_outcomes_include_valid_censored_and_failure(self):
        config = AnalysisConfig(persistence_k=3)
        steps = (
            [_step("valid", t, action="IDLE" if t < 3 else "SLOWER") for t in range(6)]
            + [_step("censored", t) for t in range(6)]
            + [_step("failure", t, collision=(t == 4)) for t in range(6)]
        )

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=SLOWDOWN_ONSET_TARGET,
        )
        by_episode = {row["episode_id"]: row for row in outcomes}

        self.assertEqual(by_episode["valid"]["episode_outcome"], VALID_ONSET)
        self.assertEqual(by_episode["valid"]["response_latency"], 3)
        self.assertEqual(
            by_episode["censored"]["episode_outcome"], NO_ONSET_CENSORED
        )
        self.assertEqual(by_episode["failure"]["episode_outcome"], TERMINAL_FAILURE)
        self.assertTrue(by_episode["failure"]["collision_flag"])

    def test_lane_change_onset_includes_latency_zero(self):
        config = AnalysisConfig(persistence_k=3)
        steps = [
            _step("lane", t, action="LANE_RIGHT", ego_lane=1 if t < 2 else 2)
            for t in range(5)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(
            outcomes[0]["target_record_type"], "confirmed_lane_change_onset"
        )
        self.assertEqual(outcomes[0]["target_mode_after"], LANE_CHANGE_TARGET_LABEL)
        self.assertEqual(outcomes[0]["target_onset_t"], 0)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 2)
        self.assertEqual(outcomes[0]["response_latency"], 0)

    def test_lane_change_onset_confirms_single_command_after_gap(self):
        # One meta-action, then lateral motion, then the lane index changes.
        # This is the observed highway-env pattern that command persistence
        # could not confirm.
        config = AnalysisConfig(persistence_k=3)
        actions = ["IDLE", "IDLE", "LANE_RIGHT", "FASTER", "FASTER", "FASTER"]
        lanes = [1, 1, 1, 1, 2, 2]
        steps = [
            _step("gapped_lane", t, action=action, ego_lane=lane)
            for t, (action, lane) in enumerate(zip(actions, lanes, strict=True))
        ]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_onset_t"], 2)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 4)
        self.assertEqual(outcomes[0]["response_latency"], 2)

    def test_lane_change_onset_requires_physical_completion(self):
        config = AnalysisConfig(persistence_k=3)
        steps = [_step("blocked_lane", t, action="LANE_RIGHT") for t in range(8)]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], NO_ONSET_CENSORED)

    def test_lane_change_onset_ignores_opposite_direction_command(self):
        # LANE_RIGHT from the rightmost lane is a no-op the simulator executes
        # without moving the vehicle. Only the LANE_LEFT command that caused the
        # realised change may date the onset.
        config = AnalysisConfig(lane_change_confirmation_window=15)
        actions = ["IDLE", "LANE_RIGHT", "LANE_RIGHT", "IDLE", "LANE_LEFT", "IDLE"]
        lanes = [1, 1, 1, 1, 1, 0]
        steps = [
            _step("noop_right", t, action=action, ego_lane=lane)
            for t, (action, lane) in enumerate(zip(actions, lanes, strict=True))
        ]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["response_latency"], 4)
        self.assertEqual(
            outcomes[0]["target_record_type"], "confirmed_lane_change_onset"
        )

    def test_lane_change_onset_ignores_pre_exposure_change(self):
        config = AnalysisConfig(persistence_k=3)
        steps = [
            _step("pre_exposure_lane", 0, action="LANE_RIGHT", ego_lane=1, exposure_t=3)
        ] + [
            _step("pre_exposure_lane", t, action="IDLE", ego_lane=2, exposure_t=3)
            for t in range(1, 6)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], NO_ONSET_CENSORED)

    def test_lane_change_onset_resegments_at_exposure(self):
        config = AnalysisConfig(persistence_k=3)
        steps = [
            _step(
                "resegmented_lane",
                t,
                action="LANE_RIGHT",
                ego_lane=1 if t < 5 else 2,
                exposure_t=3,
            )
            for t in range(6)
        ]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_onset_t"], 3)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 5)
        self.assertEqual(outcomes[0]["response_latency"], 0)

    def test_lane_change_onset_falls_back_when_command_outside_window(self):
        config = AnalysisConfig(
            persistence_k=3,
            lane_change_confirmation_window=2,
        )
        actions = ["LANE_RIGHT"] + ["IDLE"] * 6
        lanes = [1, 1, 1, 1, 1, 2, 2]
        steps = [
            _step("late_completion", t, action=action, ego_lane=lane)
            for t, (action, lane) in enumerate(zip(actions, lanes, strict=True))
        ]

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=LANE_CHANGE_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(
            outcomes[0]["target_record_type"],
            "confirmed_lane_change_onset_without_command",
        )
        self.assertEqual(outcomes[0]["target_onset_t"], 5)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 5)

    def test_slowdown_onset_uses_stable_slow_action(self):
        config = AnalysisConfig(persistence_k=3)
        steps = (
            [_step("slow_action", t, action="IDLE", ego_speed=30.0) for t in range(2)]
            + [_step("slow_action", t, action="SLOWER", ego_speed=30.0) for t in range(2, 6)]
        )

        outcomes = classify_episode_outcomes(
            steps,
            config=config,
            analysis_target=SLOWDOWN_ONSET_TARGET,
        )

        self.assertEqual(outcomes[0]["episode_outcome"], VALID_ONSET)
        self.assertEqual(outcomes[0]["target_mode_after"], SLOWDOWN_TARGET_LABEL)
        self.assertEqual(outcomes[0]["target_onset_t"], 2)
        self.assertEqual(outcomes[0]["target_confirmation_t"], 4)
        self.assertEqual(outcomes[0]["response_latency"], 2)

    def test_slowdown_onset_uses_stable_speed_drop(self):
        config = AnalysisConfig(persistence_k=3)
        speeds = [30.0, 30.0, 29.5, 29.0, 28.5, 28.0]
        steps = [
            _step("speed_drop", t, action="IDLE", ego_speed=speed)
            for t, speed in enumerate(speeds)
        ]

        outcomes = classify_episode_outcomes(
            steps,
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


