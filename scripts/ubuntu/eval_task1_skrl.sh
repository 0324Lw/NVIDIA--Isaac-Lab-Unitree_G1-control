#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ubuntu/eval_task1_skrl.sh /path/to/g1_task1_model.pt [start_k]"
  echo "Example:"
  echo "  bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/g1_task1_model.pt 0.10"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Optional overrides:
# export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
# export G1_TASK1_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt"

CKPT="$1"
START_K="${2:-0.10}"

echo "============================================================"
echo "G1 Task1 skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
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

python src/g1_rl/tasks/task1/task1_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 2000 \
  --start-k "${START_K}" \
  --print-interval 100 \
  --headless \
  --device cuda:0
