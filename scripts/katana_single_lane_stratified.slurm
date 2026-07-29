#!/bin/bash
#SBATCH --job-name=single-lane-stratified
#SBATCH --output=katana-single-lane-%A_%a.out
#SBATCH --error=katana-single-lane-%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

RUN_SEED="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="single_lane_slow_front_stratified_100k_seed${RUN_SEED}"

PYTHONPATH=src python experiments/single_lane_slow_front/run.py audit \
  --out "outputs/${RUN_NAME}_audit" \
  --num-resets 2000 \
  --seed "${RUN_SEED}"

PYTHONPATH=src python experiments/single_lane_slow_front/run.py \
  --out "outputs/${RUN_NAME}" \
  --agents FD BAL SP \
  --timesteps 100000 \
  --num-exposures 36 \
  --duration 120 \
  --evaluation-duration 120 \
  --seed "${RUN_SEED}" \
  --bootstrap-samples 500 \
  --verbose 0
