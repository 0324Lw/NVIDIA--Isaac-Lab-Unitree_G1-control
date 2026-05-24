from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from g1_rl.data.g1_motion_constants import (
    CONTROLLED_JOINT_NAMES,
    G1_JOINT_NAMES,
    JOINT_ID,
    SENSOR_JOINT_NAMES,
    TASK1_REQUIRED_KEYS,
    TASK2_REQUIRED_KEYS,
)


@dataclass
class G1SyntheticWalkConfig:
    input_file: str = ""
    output_file: str = "assets/motions/g1_walk.pt"

    default_num_frames: int = 600
    default_fps: float = 50.0

    gait_freq_hz: float = 1.45
    target_vx: float = 0.50

    hip_pitch_amp: float = 0.32
    knee_amp: float = 0.55
    ankle_pitch_amp: float = 0.22
    hip_roll_amp: float = 0.045
    ankle_roll_amp: float = 0.045
    hip_yaw_amp: float = 0.035

    shoulder_pitch_amp: float = 0.24
    shoulder_roll_amp: float = 0.035
    shoulder_yaw_amp: float = 0.050
    elbow_amp: float = 0.18
    wrist_roll_amp: float = 0.04
    waist_yaw_amp: float = 0.045

    contact_duty_ratio: float = 0.62
    fade_ratio: float = 0.08


@dataclass
class G1OmniSyntheticConfig:
    output_file: str = "assets/motions/g1_omni_walk.pt"

    fps: float = 50.0
    frames_per_mode: int = 600
    gait_freq_hz: float = 1.45
    contact_duty_ratio: float = 0.62
    fade_ratio: float = 0.08

    hip_pitch_amp: float = 0.30
    knee_amp: float = 0.50
    ankle_pitch_amp: float = 0.20
    hip_roll_amp: float = 0.055
    ankle_roll_amp: float = 0.050
    hip_yaw_amp: float = 0.050
    waist_yaw_amp: float = 0.055

    shoulder_pitch_amp: float = 0.22
    shoulder_roll_amp: float = 0.035
    shoulder_yaw_amp: float = 0.045
    elbow_amp: float = 0.16
    wrist_roll_amp: float = 0.035

    lateral_roll_scale: float = 1.25
    turn_yaw_scale: float = 1.35
    mixed_scale: float = 0.80


def _unwrap_numpy_object(raw: np.ndarray) -> Any:
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        return raw.item()
    return raw


def _safe_len_from_value(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return int(value.shape[0])

    if isinstance(value, torch.Tensor) and value.ndim >= 1:
        return int(value.shape[0])

    if isinstance(value, (list, tuple)):
        return len(value)

    if isinstance(value, dict) and len(value) > 0:
        for v in value.values():
            n = _safe_len_from_value(v)
            if n is not None and n > 0:
                return n

    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.reshape(-1)[0])
            return default

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.reshape(-1)[0].item())
            return default

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    except Exception:
        pass

    return default


def load_source_metadata(input_file: str, default_frames: int, default_fps: float) -> Tuple[int, float, Dict[str, Any]]:
    """Best-effort metadata reader.

    This generator does not perform AMP/AMASS retargeting. The optional input file
    is only used to infer sequence length and FPS if available.
    """
    if not input_file:
        return default_frames, default_fps, {"source_available": False, "frame_source_key": "default_no_input"}

    if not os.path.exists(input_file):
        print(f" ⚠️ 未找到输入文件: {input_file}")
        print(f" 将使用默认 num_frames={default_frames}, fps={default_fps}")
        return default_frames, default_fps, {"source_available": False, "frame_source_key": "default_missing_input"}

    try:
        raw = np.load(input_file, allow_pickle=True)
        data = _unwrap_numpy_object(raw)
    except Exception as exc:
        print(f" ⚠️ 读取输入文件失败: {type(exc).__name__}: {exc}")
        print(f" 将使用默认 num_frames={default_frames}, fps={default_fps}")
        return default_frames, default_fps, {"source_available": False, "load_error": str(exc)}

    meta = {"source_available": True, "source_type": type(data).__name__}

    if isinstance(data, dict):
        fps = _safe_float(data.get("fps", default_fps), default_fps)

        candidate_keys = [
            "root_translation",
            "root_pos",
            "translation",
            "rotation",
            "pose",
            "dof_pos",
            "joint_pos",
            "qpos",
        ]

        num_frames = None

        for key in candidate_keys:
            if key in data:
                num_frames = _safe_len_from_value(data[key])
                if num_frames is not None and num_frames > 0:
                    meta["frame_source_key"] = key
                    break

        if num_frames is None:
            for key, value in data.items():
                num_frames = _safe_len_from_value(value)
                if num_frames is not None and num_frames > 0:
                    meta["frame_source_key"] = key
                    break

        if num_frames is None or num_frames <= 0:
            num_frames = default_frames
            meta["frame_source_key"] = "default"

        return int(num_frames), float(fps), meta

    num_frames = _safe_len_from_value(data)
    if num_frames is None or num_frames <= 0:
        num_frames = default_frames
        meta["frame_source_key"] = "default"

    return int(num_frames), float(default_fps), meta


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smooth_fade(num_frames: int, fade_ratio: float) -> np.ndarray:
    envelope = np.ones(num_frames, dtype=np.float32)

    fade_len = int(max(1, round(num_frames * fade_ratio)))
    fade_len = min(fade_len, max(1, num_frames // 2))

    x = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_in = smoothstep(x)
    fade_out = fade_in[::-1].copy()

    envelope[:fade_len] *= fade_in
    envelope[-fade_len:] *= fade_out

    return envelope


def add_joint(pos: np.ndarray, name: str, value: np.ndarray) -> None:
    pos[:, JOINT_ID[name]] += value.astype(np.float32)


def set_joint(pos: np.ndarray, name: str, value: float) -> None:
    pos[:, JOINT_ID[name]] = float(value)


def finite_difference(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)

    if pos.shape[0] <= 1:
        return vel

    vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt

    return vel.astype(np.float32)


def build_contact_reference(phase: np.ndarray, duty_ratio: float, stand: bool = False) -> np.ndarray:
    if stand:
        return np.ones((len(phase), 2), dtype=np.float32)

    left_phase = phase
    right_phase = (phase + 0.5) % 1.0

    left_contact = (left_phase < duty_ratio).astype(np.float32)
    right_contact = (right_phase < duty_ratio).astype(np.float32)

    no_contact = (left_contact + right_contact) < 0.5
    left_contact[no_contact] = 1.0

    return np.stack([left_contact, right_contact], axis=-1).astype(np.float32)


def generate_task1_walk(num_frames: int, fps: float, cfg: G1SyntheticWalkConfig) -> Dict[str, Any]:
    dt = 1.0 / float(fps)
    t = np.arange(num_frames, dtype=np.float32) * dt

    phase = (cfg.gait_freq_hz * t) % 1.0
    omega = 2.0 * math.pi * cfg.gait_freq_hz

    s_l = np.sin(omega * t)
    c_l = np.cos(omega * t)
    s_r = np.sin(omega * t + math.pi)
    c_r = np.cos(omega * t + math.pi)

    swing_l = np.maximum(0.0, s_l)
    swing_r = np.maximum(0.0, s_r)

    fade = smooth_fade(num_frames, cfg.fade_ratio)

    pos = np.zeros((num_frames, len(G1_JOINT_NAMES)), dtype=np.float32)

    add_joint(pos, "left_hip_pitch_joint", cfg.hip_pitch_amp * s_l * fade)
    add_joint(pos, "right_hip_pitch_joint", cfg.hip_pitch_amp * s_r * fade)

    add_joint(pos, "left_knee_joint", cfg.knee_amp * swing_l * fade)
    add_joint(pos, "right_knee_joint", cfg.knee_amp * swing_r * fade)

    add_joint(pos, "left_ankle_pitch_joint", (-cfg.ankle_pitch_amp * swing_l + 0.06 * c_l) * fade)
    add_joint(pos, "right_ankle_pitch_joint", (-cfg.ankle_pitch_amp * swing_r + 0.06 * c_r) * fade)

    add_joint(pos, "left_hip_roll_joint", cfg.hip_roll_amp * c_l * fade)
    add_joint(pos, "right_hip_roll_joint", -cfg.hip_roll_amp * c_r * fade)

    add_joint(pos, "left_ankle_roll_joint", -cfg.ankle_roll_amp * c_l * fade)
    add_joint(pos, "right_ankle_roll_joint", cfg.ankle_roll_amp * c_r * fade)

    add_joint(pos, "left_hip_yaw_joint", cfg.hip_yaw_amp * s_l * fade)
    add_joint(pos, "right_hip_yaw_joint", cfg.hip_yaw_amp * s_r * fade)

    add_joint(pos, "waist_yaw_joint", cfg.waist_yaw_amp * np.sin(omega * t + math.pi / 2.0) * fade)

    add_joint(pos, "left_shoulder_pitch_joint", cfg.shoulder_pitch_amp * s_r * fade)
    add_joint(pos, "right_shoulder_pitch_joint", cfg.shoulder_pitch_amp * s_l * fade)

    add_joint(pos, "left_shoulder_roll_joint", cfg.shoulder_roll_amp * c_r * fade)
    add_joint(pos, "right_shoulder_roll_joint", -cfg.shoulder_roll_amp * c_l * fade)

    add_joint(pos, "left_shoulder_yaw_joint", cfg.shoulder_yaw_amp * s_r * fade)
    add_joint(pos, "right_shoulder_yaw_joint", cfg.shoulder_yaw_amp * s_l * fade)

    add_joint(pos, "left_elbow_joint", 0.10 + cfg.elbow_amp * np.maximum(0.0, -s_r) * fade)
    add_joint(pos, "right_elbow_joint", 0.10 + cfg.elbow_amp * np.maximum(0.0, -s_l) * fade)

    add_joint(pos, "left_wrist_roll_joint", cfg.wrist_roll_amp * c_r * fade)
    add_joint(pos, "right_wrist_roll_joint", -cfg.wrist_roll_amp * c_l * fade)

    set_joint(pos, "xl330_joint", 0.0)
    set_joint(pos, "d455_joint", 0.0)

    vel = finite_difference(pos, dt)

    for name in SENSOR_JOINT_NAMES:
        jid = JOINT_ID[name]
        pos[:, jid] = 0.0
        vel[:, jid] = 0.0

    contact_ref = build_contact_reference(phase, cfg.contact_duty_ratio, stand=False)

    root_lin_vel_ref = np.zeros((num_frames, 3), dtype=np.float32)
    root_lin_vel_ref[:, 0] = float(cfg.target_vx)

    return {
        "pos": torch.tensor(pos, dtype=torch.float32),
        "vel": torch.tensor(vel, dtype=torch.float32),
        "num_frames": int(num_frames),
        "joint_names": list(G1_JOINT_NAMES),
        "controlled_joint_names": list(CONTROLLED_JOINT_NAMES),
        "sensor_joint_names": list(SENSOR_JOINT_NAMES),
        "fps": float(fps),
        "dt": float(dt),
        "phase": torch.tensor(phase, dtype=torch.float32),
        "contact_ref": torch.tensor(contact_ref, dtype=torch.float32),
        "root_lin_vel_ref": torch.tensor(root_lin_vel_ref, dtype=torch.float32),
        "motion_type": "synthetic_g1_walk_reference",
        "description": (
            "Joint-name aligned synthetic G1 walking reference. "
            "This is not strict AMP/AMASS retargeting."
        ),
        "metadata": {
            "num_joints": len(G1_JOINT_NAMES),
            "num_controlled_joints": len(CONTROLLED_JOINT_NAMES),
            "gait_freq_hz": float(cfg.gait_freq_hz),
            "target_vx": float(cfg.target_vx),
            "contact_ref_order": ["left", "right"],
            "generator": "g1_synthetic_motions.py:task1",
        },
    }


def build_omni_modes() -> List[Dict[str, Any]]:
    return [
        {"name": "Stand", "cmd": [0.00, 0.00, 0.00], "kind": "stand"},
        {"name": "Forward_Slow", "cmd": [0.20, 0.00, 0.00], "kind": "forward"},
        {"name": "Forward_Fast", "cmd": [0.50, 0.00, 0.00], "kind": "forward"},
        {"name": "Backward", "cmd": [-0.25, 0.00, 0.00], "kind": "backward"},
        {"name": "Leftward", "cmd": [0.00, 0.18, 0.00], "kind": "lateral"},
        {"name": "Rightward", "cmd": [0.00, -0.18, 0.00], "kind": "lateral"},
        {"name": "Turn_Left", "cmd": [0.00, 0.00, 0.35], "kind": "turn"},
        {"name": "Turn_Right", "cmd": [0.00, 0.00, -0.35], "kind": "turn"},
        {"name": "Forward_Turn_Left", "cmd": [0.35, 0.00, 0.25], "kind": "mixed"},
        {"name": "Forward_Turn_Right", "cmd": [0.35, 0.00, -0.25], "kind": "mixed"},
        {"name": "Left_Turn_Left", "cmd": [0.10, 0.15, 0.20], "kind": "mixed"},
        {"name": "Right_Turn_Right", "cmd": [0.10, -0.15, -0.20], "kind": "mixed"},
    ]


def synthesize_omni_segment(mode: Dict[str, Any], cfg: G1OmniSyntheticConfig) -> Dict[str, np.ndarray]:
    T = int(cfg.frames_per_mode)
    dt = 1.0 / float(cfg.fps)

    t = np.arange(T, dtype=np.float32) * dt
    cmd = np.asarray(mode["cmd"], dtype=np.float32)

    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    kind = str(mode["kind"])

    phase = (cfg.gait_freq_hz * t) % 1.0
    omega = 2.0 * math.pi * cfg.gait_freq_hz

    s_l = np.sin(omega * t)
    c_l = np.cos(omega * t)
    s_r = np.sin(omega * t + math.pi)
    c_r = np.cos(omega * t + math.pi)

    swing_l = np.maximum(0.0, s_l)
    swing_r = np.maximum(0.0, s_r)

    fade = smooth_fade(T, cfg.fade_ratio)
    pos = np.zeros((T, len(G1_JOINT_NAMES)), dtype=np.float32)

    forward_scale = np.clip(abs(vx) / 0.50, 0.0, 1.0)
    lateral_scale = np.clip(abs(vy) / 0.25, 0.0, 1.0)
    yaw_scale = np.clip(abs(wz) / 0.40, 0.0, 1.0)

    forward_dir = 1.0 if vx >= 0.0 else -1.0
    lateral_dir = 1.0 if vy >= 0.0 else -1.0
    yaw_dir = 1.0 if wz >= 0.0 else -1.0

    is_stand = kind == "stand"

    if kind in ["forward", "backward", "mixed"]:
        scale = forward_scale if kind != "mixed" else max(forward_scale, 0.40) * cfg.mixed_scale

        add_joint(pos, "left_hip_pitch_joint", cfg.hip_pitch_amp * scale * forward_dir * s_l * fade)
        add_joint(pos, "right_hip_pitch_joint", cfg.hip_pitch_amp * scale * forward_dir * s_r * fade)

        add_joint(pos, "left_knee_joint", cfg.knee_amp * scale * swing_l * fade)
        add_joint(pos, "right_knee_joint", cfg.knee_amp * scale * swing_r * fade)

        add_joint(pos, "left_ankle_pitch_joint", (-cfg.ankle_pitch_amp * scale * swing_l + 0.04 * scale * c_l) * fade)
        add_joint(pos, "right_ankle_pitch_joint", (-cfg.ankle_pitch_amp * scale * swing_r + 0.04 * scale * c_r) * fade)

    if kind in ["lateral", "mixed"]:
        scale = lateral_scale if kind != "mixed" else max(lateral_scale, 0.35) * cfg.mixed_scale

        add_joint(pos, "left_hip_roll_joint", cfg.hip_roll_amp * cfg.lateral_roll_scale * scale * lateral_dir * c_l * fade)
        add_joint(pos, "right_hip_roll_joint", -cfg.hip_roll_amp * cfg.lateral_roll_scale * scale * lateral_dir * c_r * fade)

        add_joint(pos, "left_ankle_roll_joint", -cfg.ankle_roll_amp * cfg.lateral_roll_scale * scale * lateral_dir * c_l * fade)
        add_joint(pos, "right_ankle_roll_joint", cfg.ankle_roll_amp * cfg.lateral_roll_scale * scale * lateral_dir * c_r * fade)

        add_joint(pos, "left_hip_pitch_joint", 0.10 * scale * s_l * fade)
        add_joint(pos, "right_hip_pitch_joint", 0.10 * scale * s_r * fade)

        add_joint(pos, "left_knee_joint", 0.18 * scale * swing_l * fade)
        add_joint(pos, "right_knee_joint", 0.18 * scale * swing_r * fade)

    if kind in ["turn", "mixed"]:
        scale = yaw_scale if kind != "mixed" else max(yaw_scale, 0.35) * cfg.mixed_scale

        add_joint(
            pos,
            "waist_yaw_joint",
            cfg.waist_yaw_amp * cfg.turn_yaw_scale * scale * yaw_dir * np.sin(omega * t + math.pi / 2.0) * fade,
        )
        add_joint(pos, "left_hip_yaw_joint", cfg.hip_yaw_amp * cfg.turn_yaw_scale * scale * yaw_dir * s_l * fade)
        add_joint(pos, "right_hip_yaw_joint", cfg.hip_yaw_amp * cfg.turn_yaw_scale * scale * yaw_dir * s_r * fade)

        add_joint(pos, "left_knee_joint", 0.16 * scale * swing_l * fade)
        add_joint(pos, "right_knee_joint", 0.16 * scale * swing_r * fade)

        add_joint(pos, "left_ankle_pitch_joint", -0.06 * scale * swing_l * fade)
        add_joint(pos, "right_ankle_pitch_joint", -0.06 * scale * swing_r * fade)

    if is_stand:
        add_joint(pos, "waist_yaw_joint", 0.015 * np.sin(omega * t * 0.5) * fade)
        add_joint(pos, "left_hip_roll_joint", 0.010 * np.sin(omega * t * 0.5) * fade)
        add_joint(pos, "right_hip_roll_joint", -0.010 * np.sin(omega * t * 0.5) * fade)

    locomotion_intensity = max(forward_scale, lateral_scale * 0.7, yaw_scale * 0.6)

    if kind == "mixed":
        locomotion_intensity = max(locomotion_intensity, 0.45) * cfg.mixed_scale

    if kind == "stand":
        locomotion_intensity = 0.10

    arm_forward_sign = forward_dir if abs(vx) > 1e-5 else 1.0

    add_joint(pos, "left_shoulder_pitch_joint", cfg.shoulder_pitch_amp * locomotion_intensity * arm_forward_sign * s_r * fade)
    add_joint(pos, "right_shoulder_pitch_joint", cfg.shoulder_pitch_amp * locomotion_intensity * arm_forward_sign * s_l * fade)

    add_joint(
        pos,
        "left_shoulder_roll_joint",
        cfg.shoulder_roll_amp * locomotion_intensity * (0.5 * c_r + 0.5 * lateral_dir * lateral_scale) * fade,
    )
    add_joint(
        pos,
        "right_shoulder_roll_joint",
        -cfg.shoulder_roll_amp * locomotion_intensity * (0.5 * c_l + 0.5 * lateral_dir * lateral_scale) * fade,
    )

    add_joint(
        pos,
        "left_shoulder_yaw_joint",
        cfg.shoulder_yaw_amp * locomotion_intensity * (s_r + 0.5 * yaw_dir * yaw_scale) * fade,
    )
    add_joint(
        pos,
        "right_shoulder_yaw_joint",
        cfg.shoulder_yaw_amp * locomotion_intensity * (s_l - 0.5 * yaw_dir * yaw_scale) * fade,
    )

    add_joint(pos, "left_elbow_joint", 0.08 + cfg.elbow_amp * locomotion_intensity * np.maximum(0.0, -s_r) * fade)
    add_joint(pos, "right_elbow_joint", 0.08 + cfg.elbow_amp * locomotion_intensity * np.maximum(0.0, -s_l) * fade)

    add_joint(pos, "left_wrist_roll_joint", cfg.wrist_roll_amp * locomotion_intensity * c_r * fade)
    add_joint(pos, "right_wrist_roll_joint", -cfg.wrist_roll_amp * locomotion_intensity * c_l * fade)

    set_joint(pos, "xl330_joint", 0.0)
    set_joint(pos, "d455_joint", 0.0)

    cmd_ref = np.repeat(cmd.reshape(1, 3), T, axis=0).astype(np.float32)

    root_lin_vel_ref = np.zeros((T, 3), dtype=np.float32)
    root_ang_vel_ref = np.zeros((T, 3), dtype=np.float32)
    root_lin_vel_ref[:, 0] = vx
    root_lin_vel_ref[:, 1] = vy
    root_ang_vel_ref[:, 2] = wz

    contact_ref = build_contact_reference(phase, cfg.contact_duty_ratio, stand=is_stand)

    return {
        "pos": pos,
        "cmd": cmd_ref,
        "phase": phase.astype(np.float32),
        "contact_ref": contact_ref,
        "root_lin_vel_ref": root_lin_vel_ref,
        "root_ang_vel_ref": root_ang_vel_ref,
    }


def generate_task2_omni(cfg: G1OmniSyntheticConfig) -> Dict[str, Any]:
    modes = build_omni_modes()
    mode_names = [m["name"] for m in modes]

    pos_list = []
    cmd_list = []
    phase_list = []
    contact_list = []
    root_lin_list = []
    root_ang_list = []
    mode_id_list = []

    print("\n" + "=" * 90)
    print("Building G1 Task2 synthetic omni reference library")
    print("=" * 90)

    for mode_id, mode in enumerate(modes):
        seg = synthesize_omni_segment(mode, cfg)

        pos_list.append(seg["pos"])
        cmd_list.append(seg["cmd"])
        phase_list.append(seg["phase"])
        contact_list.append(seg["contact_ref"])
        root_lin_list.append(seg["root_lin_vel_ref"])
        root_ang_list.append(seg["root_ang_vel_ref"])
        mode_id_list.append(np.full((cfg.frames_per_mode,), mode_id, dtype=np.int64))

        print(
            f" -> mode {mode_id:02d} {mode['name']:<20} "
            f"cmd={mode['cmd']} frames={cfg.frames_per_mode}"
        )

    pos = np.concatenate(pos_list, axis=0).astype(np.float32)
    cmd = np.concatenate(cmd_list, axis=0).astype(np.float32)
    phase = np.concatenate(phase_list, axis=0).astype(np.float32)
    contact_ref = np.concatenate(contact_list, axis=0).astype(np.float32)
    root_lin_vel_ref = np.concatenate(root_lin_list, axis=0).astype(np.float32)
    root_ang_vel_ref = np.concatenate(root_ang_list, axis=0).astype(np.float32)
    mode_id = np.concatenate(mode_id_list, axis=0).astype(np.int64)

    dt = 1.0 / float(cfg.fps)
    vel = finite_difference(pos, dt)

    for i in range(1, len(modes)):
        boundary = i * cfg.frames_per_mode
        vel[boundary - 1] = 0.0
        vel[boundary] = 0.0

    for name in SENSOR_JOINT_NAMES:
        jid = JOINT_ID[name]
        pos[:, jid] = 0.0
        vel[:, jid] = 0.0

    num_frames = int(pos.shape[0])

    return {
        "pos": torch.tensor(pos, dtype=torch.float32),
        "vel": torch.tensor(vel, dtype=torch.float32),
        "cmd": torch.tensor(cmd, dtype=torch.float32),
        "num_frames": int(num_frames),
        "phase": torch.tensor(phase, dtype=torch.float32),
        "contact_ref": torch.tensor(contact_ref, dtype=torch.float32),
        "mode_id": torch.tensor(mode_id, dtype=torch.long),
        "mode_names": list(mode_names),
        "root_lin_vel_ref": torch.tensor(root_lin_vel_ref, dtype=torch.float32),
        "root_ang_vel_ref": torch.tensor(root_ang_vel_ref, dtype=torch.float32),
        "joint_names": list(G1_JOINT_NAMES),
        "controlled_joint_names": list(CONTROLLED_JOINT_NAMES),
        "sensor_joint_names": list(SENSOR_JOINT_NAMES),
        "fps": float(cfg.fps),
        "dt": float(dt),
        "motion_type": "synthetic_g1_omni_reference",
        "description": (
            "Joint-name aligned synthetic omnidirectional G1 reference library. "
            "This is not strict AMP/AMASS retargeting."
        ),
        "metadata": {
            "num_joints": len(G1_JOINT_NAMES),
            "num_controlled_joints": len(CONTROLLED_JOINT_NAMES),
            "frames_per_mode": int(cfg.frames_per_mode),
            "num_modes": len(modes),
            "gait_freq_hz": float(cfg.gait_freq_hz),
            "contact_ref_order": ["left", "right"],
            "cmd_order": ["vx", "vy", "wz"],
            "mode_names": list(mode_names),
            "generator": "g1_synthetic_motions.py:task2",
        },
    }


def validate_dataset(data: Dict[str, Any], require_cmd: bool) -> None:
    required = TASK2_REQUIRED_KEYS if require_cmd else TASK1_REQUIRED_KEYS

    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"motion file missing required keys: {missing}")

    pos = data["pos"]
    vel = data["vel"]
    phase = data["phase"]
    contact_ref = data["contact_ref"]
    num_frames = int(data["num_frames"])
    joint_names = list(data["joint_names"])

    if not isinstance(pos, torch.Tensor):
        raise RuntimeError("pos must be torch.Tensor")

    if not isinstance(vel, torch.Tensor):
        raise RuntimeError("vel must be torch.Tensor")

    if tuple(pos.shape) != tuple(vel.shape):
        raise RuntimeError(f"pos/vel shape mismatch: {tuple(pos.shape)} vs {tuple(vel.shape)}")

    if tuple(pos.shape) != (num_frames, len(G1_JOINT_NAMES)):
        raise RuntimeError(
            f"pos shape should be [{num_frames}, {len(G1_JOINT_NAMES)}], got {tuple(pos.shape)}"
        )

    if joint_names != G1_JOINT_NAMES:
        raise RuntimeError("joint_names must exactly match G1_JOINT_NAMES order")

    if tuple(phase.shape) != (num_frames,):
        raise RuntimeError(f"phase shape should be [{num_frames}], got {tuple(phase.shape)}")

    if tuple(contact_ref.shape) != (num_frames, 2):
        raise RuntimeError(f"contact_ref shape should be [{num_frames}, 2], got {tuple(contact_ref.shape)}")

    for name, tensor in [
        ("pos", pos),
        ("vel", vel),
        ("phase", phase),
        ("contact_ref", contact_ref),
    ]:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or Inf")

    sensor_ids = [JOINT_ID[name] for name in SENSOR_JOINT_NAMES]

    if torch.abs(pos[:, sensor_ids]).max().item() > 1e-6:
        raise RuntimeError("sensor joint pos should be exactly zero")

    if torch.abs(vel[:, sensor_ids]).max().item() > 1e-6:
        raise RuntimeError("sensor joint vel should be exactly zero")

    if require_cmd:
        cmd = data["cmd"]
        mode_id = data["mode_id"]
        mode_names = list(data["mode_names"])

        if tuple(cmd.shape) != (num_frames, 3):
            raise RuntimeError(f"cmd shape should be [{num_frames}, 3], got {tuple(cmd.shape)}")

        if tuple(mode_id.shape) != (num_frames,):
            raise RuntimeError(f"mode_id shape should be [{num_frames}], got {tuple(mode_id.shape)}")

        if not torch.isfinite(cmd).all():
            raise RuntimeError("cmd contains NaN or Inf")

        unique_cmd = torch.unique(cmd, dim=0)
        unique_mode = torch.unique(mode_id)

        if unique_cmd.shape[0] < 4:
            raise RuntimeError("Task2 omni dataset should contain multiple commands")

        if len(mode_names) != int(unique_mode.numel()):
            raise RuntimeError("mode_names length should match unique mode_id count")


def save_motion(data: Dict[str, Any], path: str, require_cmd: bool) -> None:
    validate_dataset(data, require_cmd=require_cmd)

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, str(out))
    print_motion_report(data, str(out), require_cmd=require_cmd)


def print_motion_report(data: Dict[str, Any], output_file: str, require_cmd: bool) -> None:
    pos = data["pos"]
    vel = data["vel"]
    contact_ref = data["contact_ref"]

    print("\n" + "=" * 90)
    print(" G1 synthetic reference generated")
    print("=" * 90)
    print(f"output_file       : {output_file}")
    print(f"motion_type       : {data.get('motion_type')}")
    print(f"num_frames        : {data['num_frames']}")
    print(f"fps               : {data['fps']}")
    print(f"dt                : {data['dt']:.6f}")
    print(f"num_joints        : {pos.shape[1]}")
    print(f"num_controlled    : {len(data['controlled_joint_names'])}")
    print(f"pos shape         : {tuple(pos.shape)}")
    print(f"vel shape         : {tuple(vel.shape)}")
    print(f"phase shape       : {tuple(data['phase'].shape)}")
    print(f"contact_ref shape : {tuple(contact_ref.shape)}")
    print("-" * 90)
    print(f"pos abs max       : {pos.abs().max().item():.6f}")
    print(f"vel abs max       : {vel.abs().max().item():.6f}")
    print(f"left contact mean : {contact_ref[:, 0].float().mean().item():.6f}")
    print(f"right contact mean: {contact_ref[:, 1].float().mean().item():.6f}")

    if require_cmd:
        cmd = data["cmd"]
        mode_id = data["mode_id"]
        mode_names = list(data["mode_names"])

        unique_cmd = torch.unique(cmd, dim=0)
        unique_mode = torch.unique(mode_id)

        print("-" * 90)
        print(f"cmd shape         : {tuple(cmd.shape)}")
        print(f"mode_id shape     : {tuple(mode_id.shape)}")
        print(f"unique commands   : {len(unique_cmd)}")
        print(f"unique modes      : {len(unique_mode)}")
        print(f"cmd min           : {cmd.min(dim=0).values.tolist()}")
        print(f"cmd max           : {cmd.max(dim=0).values.tolist()}")
        print("-" * 90)
        print("modes:")
        for i, name in enumerate(mode_names):
            mask = mode_id == i
            if mask.any():
                mean_cmd = cmd[mask].mean(dim=0).tolist()
                print(f"  {i:02d}: {name:<20} frames={int(mask.sum().item()):5d} cmd={mean_cmd}")

    print("-" * 90)
    print("joint_names:")
    for i, name in enumerate(data["joint_names"]):
        tag = " [sensor]" if name in SENSOR_JOINT_NAMES else ""
        print(f"  {i:02d}: {name}{tag}")

    print("=" * 90 + "\n")


def load_and_validate(path: str) -> None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"motion file not found: {p}")

    data = torch.load(str(p), map_location="cpu")
    require_cmd = "cmd" in data

    validate_dataset(data, require_cmd=require_cmd)
    print_motion_report(data, str(p), require_cmd=require_cmd)


def command_task1(args) -> None:
    cfg = G1SyntheticWalkConfig(
        input_file=args.input,
        output_file=args.output,
        default_num_frames=args.default_frames,
        default_fps=args.default_fps,
        gait_freq_hz=args.gait_freq,
        target_vx=args.target_vx,
        fade_ratio=args.fade_ratio,
    )

    print("\n" + "=" * 90)
    print("G1 Task1 synthetic walking reference generator")
    print("注意：这是正弦合成参考步态，不是严格 AMP/AMASS retargeting。")
    print("=" * 90)

    num_frames, fps, source_meta = load_source_metadata(
        cfg.input_file,
        cfg.default_num_frames,
        cfg.default_fps,
    )

    if num_frames < 120:
        print(f" ⚠️ 输入帧数过短: {num_frames}，自动扩展到 300 帧以保证参考稳定。")
        num_frames = 300

    motion = generate_task1_walk(num_frames, fps, cfg)
    motion["metadata"]["source_meta"] = source_meta

    save_motion(motion, cfg.output_file, require_cmd=False)


def command_task2(args) -> None:
    cfg = G1OmniSyntheticConfig(
        output_file=args.output,
        fps=args.fps,
        frames_per_mode=args.frames_per_mode,
        gait_freq_hz=args.gait_freq,
        fade_ratio=args.fade_ratio,
    )

    print("\n" + "=" * 90)
    print("G1 Task2 synthetic omnidirectional reference generator")
    print("注意：这是全向正弦合成参考库，不是严格 AMP/AMASS retargeting。")
    print("=" * 90)

    motion = generate_task2_omni(cfg)
    save_motion(motion, cfg.output_file, require_cmd=True)


def command_all(args) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task1_out = out_dir / args.task1_file
    task2_out = out_dir / args.task2_file

    class Obj:
        pass

    a1 = Obj()
    a1.input = args.input
    a1.output = str(task1_out)
    a1.default_frames = args.task1_frames
    a1.default_fps = args.fps
    a1.gait_freq = args.gait_freq
    a1.target_vx = args.target_vx
    a1.fade_ratio = args.fade_ratio

    a2 = Obj()
    a2.output = str(task2_out)
    a2.fps = args.fps
    a2.frames_per_mode = args.frames_per_mode
    a2.gait_freq = args.gait_freq
    a2.fade_ratio = args.fade_ratio

    command_task1(a1)
    command_task2(a2)

    print("\n" + "=" * 90)
    print("G1 synthetic motion files generated")
    print("=" * 90)
    print(f"Task1 motion: {task1_out}")
    print(f"Task2 motion: {task2_out}")
    print("")
    print("Use these environment variables for tests/training:")
    print(f'export G1_TASK1_MOTION_FILE="{task1_out}"')
    print(f'export G1_TASK2_MOTION_FILE="{task2_out}"')
    print("=" * 90 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic G1 Task1/Task2 reference motions.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("task1", help="Generate Task1 g1_walk.pt")
    p1.add_argument("--input", type=str, default="")
    p1.add_argument("--output", type=str, default="assets/motions/g1_walk.pt")
    p1.add_argument("--default-frames", type=int, default=600)
    p1.add_argument("--default-fps", type=float, default=50.0)
    p1.add_argument("--gait-freq", type=float, default=1.45)
    p1.add_argument("--target-vx", type=float, default=0.50)
    p1.add_argument("--fade-ratio", type=float, default=0.08)
    p1.set_defaults(func=command_task1)

    p2 = sub.add_parser("task2", help="Generate Task2 g1_omni_walk.pt")
    p2.add_argument("--output", type=str, default="assets/motions/g1_omni_walk.pt")
    p2.add_argument("--fps", type=float, default=50.0)
    p2.add_argument("--frames-per-mode", type=int, default=600)
    p2.add_argument("--gait-freq", type=float, default=1.45)
    p2.add_argument("--fade-ratio", type=float, default=0.08)
    p2.set_defaults(func=command_task2)

    pall = sub.add_parser("all", help="Generate both Task1 and Task2 motion files")
    pall.add_argument("--output-dir", type=str, default="assets/motions")
    pall.add_argument("--task1-file", type=str, default="g1_walk.pt")
    pall.add_argument("--task2-file", type=str, default="g1_omni_walk.pt")
    pall.add_argument("--input", type=str, default="")
    pall.add_argument("--fps", type=float, default=50.0)
    pall.add_argument("--task1-frames", type=int, default=600)
    pall.add_argument("--frames-per-mode", type=int, default=600)
    pall.add_argument("--gait-freq", type=float, default=1.45)
    pall.add_argument("--target-vx", type=float, default=0.50)
    pall.add_argument("--fade-ratio", type=float, default=0.08)
    pall.set_defaults(func=command_all)

    pval = sub.add_parser("validate", help="Validate an existing G1 motion .pt file")
    pval.add_argument("--file", type=str, required=True)
    pval.set_defaults(func=lambda args: load_and_validate(args.file))

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
