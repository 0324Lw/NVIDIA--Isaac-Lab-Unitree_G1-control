#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Optional:
# export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
# export G1_TASK2_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"
# export G1_TASK1_PRETRAINED="/home/lw/unitree_g1_isaaclab_rl/logs/task1/<run>/final_checkpoint/g1_task1_model.pt"

ARGS=(
  src/g1_rl/tasks/task2/task2_train.py
  --num-envs 256
  --total-env-steps 500000000
  --rollouts 64
  --learning-epochs 5
  --mini-batches 8
  --lr 1e-4
  --min-lr 7e-5
  --max-lr 2e-4
  --gamma 0.99
  --gae-lambda 0.95
  --kl-threshold 0.015
  --entropy-coef 0.0025
  --value-coef 2.0
  --init-log-std -1.35
  --summary-interval 10
  --tb-log-interval-steps 50
  --skrl-write-interval 1000000
  --skrl-checkpoint-interval 0
  --save-freq-env-steps 20000000
  --headless
  --device cuda:0
)

if [ -n "${G1_USD_PATH:-}" ]; then
  ARGS+=(--usd-path "$G1_USD_PATH")
fi

if [ -n "${G1_TASK2_MOTION_FILE:-}" ]; then
  ARGS+=(--motion-file "$G1_TASK2_MOTION_FILE")
fi

if [ -n "${G1_TASK1_PRETRAINED:-}" ]; then
  ARGS+=(--pretrained-task1 "$G1_TASK1_PRETRAINED")
fi

python "${ARGS[@]}"
