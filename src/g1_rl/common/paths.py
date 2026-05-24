from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return repository root.

    File location:
        src/g1_rl/common/paths.py

    parents:
        0 -> common
        1 -> g1_rl
        2 -> src
        3 -> project root
    """
    return Path(__file__).resolve().parents[3]


def src_root() -> Path:
    return project_root() / "src"


def default_log_root(task_name: str) -> str:
    """Default log root for a task.

    Can be overridden by:
        RT_G1_LOG_ROOT
        RT_G1_TASK1_LOG_ROOT
        RT_G1_TASK2_LOG_ROOT
        ...
    """
    task_name = str(task_name).strip().lower()
    env_key = f"RT_G1_{task_name.upper()}_LOG_ROOT"

    if os.environ.get(env_key):
        return os.environ[env_key]

    if os.environ.get("RT_G1_LOG_ROOT"):
        return str(Path(os.environ["RT_G1_LOG_ROOT"]) / task_name)

    return str(project_root() / "logs" / task_name)


def asset_path_from_env(env_key: str, default_path: str) -> str:
    """Resolve asset path with environment-variable override."""
    return os.environ.get(env_key, default_path)


def ensure_project_on_pythonpath() -> None:
    """Add src/ to sys.path when a script is launched directly."""
    import sys

    src = str(src_root())
    if src not in sys.path:
        sys.path.insert(0, src)
