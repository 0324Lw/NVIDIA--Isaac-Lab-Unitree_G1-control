# Copyright (c) 2026
# Unitree G1 Task1: assisted locomotion pure-RL baseline config.
#
# This project intentionally keeps a pure-RL humanoid baseline for learning.
# It is not presented as the final/professional route for complex humanoid skills.
# For high-quality humanoid motion, imitation learning / retargeted motion data /
# motion priors are usually required.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from g1_rl.common.paths import asset_path_from_env


@dataclass
class Task1Config:
    """G1 Task1 assisted locomotion environment config.

    Design notes:
        - No separate world file.
        - IsaacLab scene and robot are created in task1_env.py.
        - Policy controls 23 joints.
        - Two sensor joints are fixed: xl330_joint and d455_joint.
        - Single-frame observation dim is 123.
        - Training script will later apply 5-frame stacking: 123 * 5 = 615.
    """

    # ----------------------------- Basic -----------------------------
    num_envs: int = 1024
    device: str = "cuda:0"

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length: int = 1000
    env_spacing: float = 2.0

    usd_path: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_USD_PATH",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd",
        )
    )

    motion_file: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_TASK1_MOTION_FILE",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt",
        )
    )

    # ----------------------------- Joint names -----------------------------
    # 25 joints in USD. Two sensor joints are excluded from policy control.
    all_joint_names: Tuple[str, ...] = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "xl330_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "d455_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
    )

    sensor_joint_names: Tuple[str, ...] = ("xl330_joint", "d455_joint")

    controlled_joint_names: Tuple[str, ...] = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
    )

    leg_joint_names: Tuple[str, ...] = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    )

    waist_joint_names: Tuple[str, ...] = ("waist_yaw_joint",)

    arm_joint_names: Tuple[str, ...] = (
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
    )

    foot_body_names: Tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

    # ----------------------------- Dimensions -----------------------------
    num_actions: int = 23
    num_observations: int = 123

    frame_stack: int = 5

    # ----------------------------- Control -----------------------------
    ema_alpha: float = 0.55
    leg_action_scale: float = 0.25
    waist_action_scale: float = 0.10
    arm_action_scale: float = 0.08
    wrist_action_scale: float = 0.05
    max_joint_vel_abs: float = 50.0

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 300_000_000

    # Stage target:
    # 0: stand
    # 1: marching
    # 2: assisted slow walking
    # 3: normal walking
    # 4: reduced-harness reference-style walking
    target_vx_final: float = 0.50
    target_vy: float = 0.0
    target_yaw_rate: float = 0.0

    target_height: float = 0.75
    fall_height: float = 0.52
    jump_height: float = 1.05
    bad_orientation_xy: float = 0.70

    harness_start: float = 0.80
    harness_end: float = 0.0
    harness_body_id: int = 0

    gait_freq_hz: float = 1.45
    reference_reset_scale_max: float = 0.45

    # ----------------------------- Contact -----------------------------
    contact_force_threshold: float = 8.0
    foot_clearance_target: float = 0.055

    # ----------------------------- Reward weights -----------------------------
    # Locomotion / gait
    w_vx: float = 0.26
    w_yaw: float = 0.04
    w_cmd_lat: float = 0.04
    w_phase_contact: float = 0.10
    w_air_time: float = 0.085
    w_clearance: float = 0.065
    w_double_contact: float = 0.055

    # Trunk / stability
    w_upright: float = 0.135
    w_height: float = 0.125
    w_base_ang_vel: float = 0.045
    w_base_acc: float = 0.0010
    w_com_support: float = 0.025

    # Safety / efficiency / style
    w_default_pose: float = 0.020
    w_ref_pose: float = 0.025
    w_ref_vel: float = 0.010
    w_alive: float = 0.002
    w_joint_limit: float = 0.05
    w_action_rate: float = 0.006
    w_action_mag: float = 0.0015
    w_foot_slip: float = 0.055
    w_energy: float = 0.0012

    # Event reward
    penalty_fall: float = -5.0

    # Reward kernels
    sigma_v: float = 10.0
    sigma_yaw: float = 4.0
    sigma_z: float = 12.0
    sigma_pose: float = 3.0
    sigma_vel: float = 0.08
    deadband_height: float = 0.04

    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 1000.0

    # ----------------------------- Debug -----------------------------
    strict_motion_joint_check: bool = True
    print_debug_info: bool = False

    @property
    def control_dt(self) -> float:
        return float(self.sim_dt * self.decimation)

    @property
    def stacked_obs_dim(self) -> int:
        return int(self.num_observations * self.frame_stack)

    def validate(self) -> None:
        assert len(self.all_joint_names) == 25, f"Expected 25 G1 joints, got {len(self.all_joint_names)}"
        assert len(self.sensor_joint_names) == 2, f"Expected 2 sensor joints, got {len(self.sensor_joint_names)}"
        assert len(self.controlled_joint_names) == self.num_actions, (
            f"controlled_joint_names length {len(self.controlled_joint_names)} != num_actions {self.num_actions}"
        )
        assert self.num_observations == 123, f"Task1 single obs dim should be 123, got {self.num_observations}"
        assert self.stacked_obs_dim == 615, f"Task1 stacked obs dim should be 615, got {self.stacked_obs_dim}"
        assert self.max_episode_length > 0
        assert self.control_dt > 0.0
