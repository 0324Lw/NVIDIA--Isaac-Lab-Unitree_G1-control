#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

ARGS=(
  src/g1_rl/tasks/task4/task4_train.py
  --num-envs 128
  --total-env-steps 700000000
  --rollouts 64
  --epochs 5
  --mini-batches 8
  --lr 5e-5
  --min-lr 2e-5
  --max-lr 1e-4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-range 0.2
  --target-kl 0.015
  --hard-kl-stop 0.08
  --entropy-coef 0.001
  --value-coef 2.0
  --grad-clip 0.5
  --init-log-std -1.6
  --min-log-std -4.0
  --max-log-std -0.8
  --summary-interval 1
  --save-freq-env-steps 20000000
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
