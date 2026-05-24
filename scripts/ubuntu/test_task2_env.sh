#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Optional overrides:
# export G1_USD_PATH="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
# export G1_TASK2_MOTION_FILE="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"

echo "============================================================"
echo "G1 Task2 Omni Env Test"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "G1_USD_PATH=${G1_USD_PATH:-<default from Task2Config>}"
echo "G1_TASK2_MOTION_FILE=${G1_TASK2_MOTION_FILE:-<default from Task2Config>}"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

try:
    import torch
    print("[CHECK] torch:", torch.__version__)
    print("[CHECK] cuda available:", torch.cuda.is_available())
except Exception as e:
    raise RuntimeError("Current Python cannot import torch. Please activate conda env: isaaclab") from e

try:
    import isaaclab
    print("[CHECK] isaaclab: ok")
except Exception as e:
    raise RuntimeError("Current Python cannot import isaaclab. Please activate IsaacLab conda env.") from e
PY

python tests/task2/task2_env_test.py \
  --num-envs 8 \
  --steps 240 \
  --collect-interval 40 \
  --headless \
  --test-device cuda:0 \
  ${G1_USD_PATH:+--usd-path "$G1_USD_PATH"} \
  ${G1_TASK2_MOTION_FILE:+--motion-file "$G1_TASK2_MOTION_FILE"}
