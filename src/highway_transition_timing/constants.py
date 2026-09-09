"""Shared vocabulary for the transition-timing analysis."""

AGENTS = ("FD", "BAL", "SP")

LANE_CHANGE_ONSET_TARGET = "lane_change_onset"
LANE_CHANGE_TARGET_LABEL = "lane_change_action"
SLOWDOWN_ONSET_TARGET = "slowdown_onset"
SLOWDOWN_TARGET_LABEL = "slowdown_action"

VALID_ONSET = "valid_onset"
NO_ONSET_CENSORED = "no_onset_censored"
TERMINAL_FAILURE = "terminal_failure"

VALID_PAIR = "valid_pair"
CENSORED_PAIR = "censored_pair"
TERMINAL_FAILURE_PAIR = "terminal_failure_pair"

LANE_CHANGE_ACTIONS = {"LANE_LEFT", "LANE_RIGHT", "LEFT", "RIGHT"}
LEFT_LANE_CHANGE_ACTIONS = {"LANE_LEFT", "LEFT"}
RIGHT_LANE_CHANGE_ACTIONS = {"LANE_RIGHT", "RIGHT"}
SLOW_ACTIONS = {"SLOWER", "DECELERATE", "BRAKE"}
FAST_ACTIONS = {"FASTER", "ACCELERATE"}
IDLE_ACTIONS = {"IDLE", "KEEP", "MAINTAIN"}

FRONT_DISTANCE_SCORE_HORIZON = 120.0
