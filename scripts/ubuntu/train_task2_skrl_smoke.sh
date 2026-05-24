#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Optional overrides:
# export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
# export G1_TASK2_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"
# export G1_TASK1_PRETRAINED="/home/lw/unitree_g1_isaaclab_rl/logs/task1/<run>/final_checkpoint/g1_task1_model.pt"

echo "============================================================"
echo "G1 Task2 Omni pure-RL skrl PPO smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "G1_USD_PATH=${G1_USD_PATH:-<default from Task2Config>}"
echo "G1_TASK2_MOTION_FILE=${G1_TASK2_MOTION_FILE:-<default from Task2Config>}"
echo "G1_TASK1_PRETRAINED=${G1_TASK1_PRETRAINED:-<none>}"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

import torch
print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda:", torch.cuda.is_available())

import isaaclab
print("[CHECK] isaaclab: ok")

import skrl
print("[CHECK] skrl:", getattr(skrl, "__version__", "unknown"))
PY

ARGS=(
  src/g1_rl/tasks/task2/task2_train.py
  --num-envs 8
  --total-env-steps 16384
  --rollouts 16
  --learning-epochs 3
  --mini-batches 2
  --lr 1e-4
  --min-lr 7e-5
  --max-lr 2e-4
  --summary-interval 1
  --tb-log-interval-steps 10
  --skrl-write-interval 1000000
  --skrl-checkpoint-interval 0
  --save-freq-env-steps 16384
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
