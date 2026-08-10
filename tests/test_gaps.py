import unittest

from highway_transition_timing.config import AnalysisConfig
from highway_transition_timing.constants import (
    TERMINAL_FAILURE,
    TERMINAL_FAILURE_PAIR,
    VALID_ONSET,
    VALID_PAIR,
)
from highway_transition_timing.gaps import compute_timing_gaps, summarize_gaps


class GapTests(unittest.TestCase):
    def test_valid_pair_gap(self):
        outcomes = [
            _outcome("FD", "E0", VALID_ONSET, 3),
            _outcome("SP", "E0", VALID_ONSET, 8),
        ]

        gaps = compute_timing_gaps(outcomes, agent_pairs=[("FD", "SP")])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["gap_status"], VALID_PAIR)
        self.assertEqual(gaps[0]["gap_b_minus_a"], 5.0)

    def test_failure_pair_has_no_numeric_gap(self):
        outcomes = [
            _outcome("FD", "E0", VALID_ONSET, 3),
            _outcome("SP", "E0", TERMINAL_FAILURE, ""),
        ]

        gaps = compute_timing_gaps(outcomes, agent_pairs=[("FD", "SP")])

        self.assertEqual(gaps[0]["gap_status"], TERMINAL_FAILURE_PAIR)
        self.assertEqual(gaps[0]["gap_b_minus_a"], "")

    def test_gap_summary_median(self):
        gaps = [
            {
                "analysis_target": "slowdown_onset",
                "agent_a": "FD",
                "agent_b": "SP",
                "gap_status": VALID_PAIR,
                "gap_b_minus_a": value,
            }
            for value in [3, 5, 9]
        ]

        summary = summarize_gaps(gaps, AnalysisConfig(bootstrap_samples=10))

        self.assertEqual(summary[0]["median_gap"], 5)
        self.assertEqual(summary[0]["n_valid_pairs"], 3)


def _outcome(agent, exposure_id, outcome, latency):
    return {
        "episode_id": f"{agent}_{exposure_id}",
        "agent_condition": agent,
        "policy_id": f"{agent}_main",
        "exposure_seed": "seed0",
        "exposure_id": exposure_id,
        "rollout_id": "r0",
        "rollout_seed": "0",
        "analysis_target": "slowdown_onset",
        "episode_outcome": outcome,
        "response_latency": latency,
    }


if __name__ == "__main__":
    unittest.main()
