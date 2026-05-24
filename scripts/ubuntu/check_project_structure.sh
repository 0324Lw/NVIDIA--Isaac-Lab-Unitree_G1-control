#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "G1 project structure check"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
from pathlib import Path

from g1_rl.common.paths import project_root
from g1_rl.tasks.task1.task1_config import Task1Config

root = project_root()

required = [
    root / "configs" / "task1_assisted_locomotion.yaml",
    root / "src" / "g1_rl" / "common" / "paths.py",
    root / "src" / "g1_rl" / "common" / "info_utils.py",
    root / "src" / "g1_rl" / "tasks" / "task1" / "task1_config.py",
    root / "tests" / "task1",
    root / "scripts" / "ubuntu",
    root / "scripts" / "windows",
    root / "assets" / "gifs",
    root / "assets" / "images",
]

missing = [str(p) for p in required if not p.exists()]
if missing:
    raise RuntimeError("Missing files/directories:\n" + "\n".join(missing))

cfg = Task1Config()
cfg.validate()

print("[OK] project_root:", root)
print("[OK] Task1Config validated")
print("[OK] num_actions:", cfg.num_actions)
print("[OK] single_obs_dim:", cfg.num_observations)
print("[OK] stacked_obs_dim:", cfg.stacked_obs_dim)
print("[OK] usd_path:", cfg.usd_path)
print("[OK] motion_file:", cfg.motion_file)
print("[OK] framework check passed")
PY
