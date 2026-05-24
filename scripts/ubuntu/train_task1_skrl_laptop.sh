#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Optional:
# export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
# export G1_TASK1_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt"

python src/g1_rl/tasks/task1/task1_train.py \
  --num-envs 256 \
  --total-env-steps 300000000 \
  --rollouts 64 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --max-lr 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --kl-threshold 0.015 \
  --entropy-coef 0.002 \
  --value-coef 2.0 \
  --init-log-std -1.35 \
  --summary-interval 10 \
  --tb-log-interval-steps 50 \
  --skrl-write-interval 1000000 \
  --skrl-checkpoint-interval 0 \
  --save-freq-env-steps 20000000 \
  --headless \
  --device cuda:0
