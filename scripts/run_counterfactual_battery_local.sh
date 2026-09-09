#!/bin/bash
#
# Run the selection-independent counterfactual battery locally.
#
# Same work as the three katana_arm*_counterfactual.pbs arrays, driven with a
# fixed number of parallel workers instead of a scheduler. Inference only: no
# training, no checkpoint selection, development grid throughout, and the sealed
# held-out grid is never loaded.
#
# Resumable. A seed whose summary output already exists is skipped, so an
# interrupted run can be restarted with the same command.
#
# Usage:
#   scripts/run_counterfactual_battery_local.sh [A|B|C|all] [jobs] [python]

set -uo pipefail

ARM="${1:-all}"
JOBS="${2:-4}"
PYTHON_BIN="${3:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS="${REPO_ROOT}/outputs"
LOG_DIR="${REPO_ROOT}/logs/counterfactual_battery"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONUNBUFFERED=1

run_single_lane() {
  local seed="$1"
  local run_dir="${OUTPUTS}/single_lane_slow_front_fixed_100k_seed${seed}"
  local log="${LOG_DIR}/armA_seed${seed}.log"
  [[ -d "${run_dir}/models" ]] || { echo "MISSING ${run_dir}"; return 1; }
  if [[ -f "${run_dir}/rollout_counterfactual_front_vehicle/counterfactual_rollout_summary.csv" \
     && -f "${run_dir}/counterfactual_front_vehicle/counterfactual_action_summary.csv" ]]; then
    echo "SKIP armA seed ${seed} (already complete)"
    return 0
  fi

  # The scenario flags must match the training array: the evaluation grid and
  # the exposure seeds are derived from them, and this entry point builds its
  # config from the CLI rather than from training_runs.csv.
  local common=(
    --run-dir "${run_dir}"
    --agents FD BAL SP
    --num-exposures 36
    --seed "${seed}"
    --evaluation-duration 120
    --duration 20
    --collision-risk-penalty 3.0
    --reward-strength-multiplier 1
    --counterfactual-variants original no-front matched-speed-front far-front
  )
  {
    "${PYTHON_BIN}" "${REPO_ROOT}/experiments/single_lane_slow_front/run.py" \
      rollout-counterfactual "${common[@]}" --bootstrap-samples 500 --no-figures \
    && "${PYTHON_BIN}" "${REPO_ROOT}/experiments/single_lane_slow_front/run.py" \
      counterfactual "${common[@]}"
  } > "${log}" 2>&1
  local status=$?
  [[ ${status} -eq 0 ]] && echo "OK   armA seed ${seed}" || echo "FAIL armA seed ${seed} (see ${log})"
  return ${status}
}

run_two_lane() {
  local arm="$1" seed="$2" prefix="$3"
  shift 3
  local variants=("$@")
  local run_dir="${OUTPUTS}/${prefix}_seed${seed}"
  local log="${LOG_DIR}/arm${arm}_seed${seed}.log"
  [[ -d "${run_dir}/models" ]] || { echo "MISSING ${run_dir}"; return 1; }
  if [[ -f "${run_dir}/rollout_counterfactual_open_lane/opening_action_summary.csv" ]]; then
    echo "SKIP arm${arm} seed ${seed} (already complete)"
    return 0
  fi

  # The scenario geometry is restored from training_runs.csv, so no scenario
  # flags are passed here.
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/multilane_open_lane_change/rollout_counterfactual.py" \
    --run-dir "${run_dir}" \
    --agents FD BAL SP \
    --num-exposures 36 \
    --evaluation-duration 120 \
    --eval-grid development \
    --variants "${variants[@]}" \
    > "${log}" 2>&1
  local status=$?
  [[ ${status} -eq 0 ]] && echo "OK   arm${arm} seed ${seed}" || echo "FAIL arm${arm} seed ${seed} (see ${log})"
  return ${status}
}

# macOS ships bash 3.2, which has no `wait -n`, so the throttle polls instead of
# blocking on the next child to exit.
throttle() {
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 2; done
}

started=$(date +%s)

if [[ "${ARM}" == "A" || "${ARM}" == "all" ]]; then
  for index in $(seq 0 19); do
    throttle
    run_single_lane $((3100 + index)) &
  done
fi

if [[ "${ARM}" == "B" || "${ARM}" == "all" ]]; then
  for index in $(seq 0 19); do
    throttle
    run_two_lane B $((4100 + index)) multilane_twolane_open_fixed_100k \
      original no-front matched-speed-front far-front &
  done
fi

if [[ "${ARM}" == "C" || "${ARM}" == "all" ]]; then
  for index in $(seq 0 19); do
    throttle
    run_two_lane C $((4200 + index)) multilane_twolane_occupied_fixed_100k \
      original no-front matched-speed-front far-front open-target-lane &
  done
fi

wait
echo "battery finished in $(( ($(date +%s) - started) / 60 )) minutes; logs in ${LOG_DIR}"
