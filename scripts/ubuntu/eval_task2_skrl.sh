#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ubuntu/eval_task2_skrl.sh /path/to/g1_task2_omni_model.pt [start_k]"
  echo "Example:"
  echo "  bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/g1_task2_omni_model.pt 0.10"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-0.10}"

echo "============================================================"
echo "G1 Task2 Omni skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
echo "PYTHON=$(which python)"
echo "G1_USD_PATH=${G1_USD_PATH:-<default from Task2Config>}"
echo "G1_TASK2_MOTION_FILE=${G1_TASK2_MOTION_FILE:-<default from Task2Config>}"
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
  src/g1_rl/tasks/task2/task2_model_test.py
  --checkpoint "${CKPT}"
  --num-envs 4
  --steps 2000
  --start-k "${START_K}"
  --print-interval 100
  --headless
  --device cuda:0
)

if [ -n "${G1_USD_PATH:-}" ]; then
  ARGS+=(--usd-path "$G1_USD_PATH")
fi

if [ -n "${G1_TASK2_MOTION_FILE:-}" ]; then
  ARGS+=(--motion-file "$G1_TASK2_MOTION_FILE")
fi

python "${ARGS[@]}"
