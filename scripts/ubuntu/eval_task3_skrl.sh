#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ubuntu/eval_task3_skrl.sh /path/to/g1_task3_whole_body_model.pt [start_k]"
  echo "Example:"
  echo "  bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/g1_task3_whole_body_model.pt 1.0"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-1.0}"

echo "============================================================"
echo "G1 Task3 Whole-Body skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
echo "PYTHON=$(which python)"
echo "G1_USD_PATH=${G1_USD_PATH:-<default from Task3Config>}"
echo "G1_TASK3_MOTION_FILE=${G1_TASK3_MOTION_FILE:-<default from Task3Config>}"
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
  src/g1_rl/tasks/task3/task3_model_test.py
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

if [ -n "${G1_TASK3_MOTION_FILE:-}" ]; then
  ARGS+=(--motion-file "$G1_TASK3_MOTION_FILE")
fi

python "${ARGS[@]}"
