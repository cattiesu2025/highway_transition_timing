#!/bin/bash
# Reuse the shared snapshot tool; print the submission command for this experiment.
# Run from the repository root. Optional args: source venv, archive root.
set -euo pipefail
bash scripts/katana_pack_training_env.sh "$@" |
  sed 's|,SWEEP_INDEX=2 scripts/katana_reward_local_retry.pbs| experiments/overtake_success/scripts/katana_overtake_local_retry.pbs|'
