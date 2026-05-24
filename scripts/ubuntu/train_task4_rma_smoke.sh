#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "G1 Task4 RMA smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "G1_USD_PATH=${G1_USD_PATH:-<default from Task4Config>}"
echo "G1_TASK4_MOTION_FILE=${G1_TASK4_MOTION_FILE:-<default from Task4Config>}"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)
import torch
print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda:", torch.cuda.is_available())
import isaaclab
print("[CHECK] isaaclab: ok")
PY

ARGS=(
  src/g1_rl/tasks/task4/task4_train.py
  --num-envs 8
  --total-env-steps 16384
  --rollouts 16
  --epochs 3
  --mini-batches 2
  --summary-interval 1
  --save-freq-env-steps 16384
  --headless
  --device cuda:0
)

if [ -n "${G1_USD_PATH:-}" ]; then
  ARGS+=(--usd-path "$G1_USD_PATH")
fi

if [ -n "${G1_TASK4_MOTION_FILE:-}" ]; then
  ARGS+=(--motion-file "$G1_TASK4_MOTION_FILE")
fi

python "${ARGS[@]}"
