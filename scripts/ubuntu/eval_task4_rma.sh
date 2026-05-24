#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task4_rma.sh /path/to/checkpoint_or_final_checkpoint_dir [start_k]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/eval_task4_rma.sh logs/task4/<run>/final_checkpoint/g1_task4_student_deploy.pt 1.0"
  echo "  bash scripts/ubuntu/eval_task4_rma.sh logs/task4/<run>/final_checkpoint 1.0"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-1.0}"

echo "============================================================"
echo "G1 Task4 RMA model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
echo "PYTHON=$(which python)"
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
  src/g1_rl/tasks/task4/task4_model_test.py
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

if [ -n "${G1_TASK4_MOTION_FILE:-}" ]; then
  ARGS+=(--motion-file "$G1_TASK4_MOTION_FILE")
fi

python "${ARGS[@]}"
