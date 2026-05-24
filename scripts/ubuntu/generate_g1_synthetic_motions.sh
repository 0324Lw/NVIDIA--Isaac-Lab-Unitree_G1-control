#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

OUTPUT_DIR="${1:-${PROJECT_ROOT}/assets/motions}"

echo "============================================================"
echo "Generate G1 synthetic motion references"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

import torch
print("[CHECK] torch:", torch.__version__)

import numpy as np
print("[CHECK] numpy:", np.__version__)
PY

python src/g1_rl/data/g1_synthetic_motions.py all \
  --output-dir "${OUTPUT_DIR}" \
  --task1-file g1_walk.pt \
  --task2-file g1_omni_walk.pt \
  --fps 50.0 \
  --task1-frames 600 \
  --frames-per-mode 600 \
  --gait-freq 1.45 \
  --target-vx 0.50 \
  --fade-ratio 0.08

python src/g1_rl/data/g1_synthetic_motions.py validate \
  --file "${OUTPUT_DIR}/g1_walk.pt"

python src/g1_rl/data/g1_synthetic_motions.py validate \
  --file "${OUTPUT_DIR}/g1_omni_walk.pt"

echo ""
echo "✅ G1 synthetic motion generation completed."
echo ""
echo "Recommended exports:"
echo "  export G1_TASK1_MOTION_FILE=\"${OUTPUT_DIR}/g1_walk.pt\""
echo "  export G1_TASK2_MOTION_FILE=\"${OUTPUT_DIR}/g1_omni_walk.pt\""
echo ""
