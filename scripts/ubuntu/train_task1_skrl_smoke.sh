#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Optional overrides:
# export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
# export G1_TASK1_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt"

echo "============================================================"
echo "G1 Task1 pure-RL skrl PPO smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "G1_USD_PATH=${G1_USD_PATH:-<default from Task1Config>}"
echo "G1_TASK1_MOTION_FILE=${G1_TASK1_MOTION_FILE:-<default from Task1Config>}"
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

python src/g1_rl/tasks/task1/task1_train.py \
  --num-envs 8 \
  --total-env-steps 16384 \
  --rollouts 16 \
  --learning-epochs 3 \
  --mini-batches 2 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --max-lr 3e-4 \
  --summary-interval 1 \
  --tb-log-interval-steps 10 \
  --skrl-write-interval 1000000 \
  --skrl-checkpoint-interval 0 \
  --save-freq-env-steps 16384 \
  --headless \
  --device cuda:0
