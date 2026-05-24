from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from g1_rl.common.paths import asset_path_from_env
from g1_rl.tasks.task2.task2_config import Task2Config


@dataclass
class Task3Config(Task2Config):
    """G1 Task3 whole-body locomotion config.

    Task3 is still the educational pure-RL baseline version:
        - no world file
        - no HoloSoma / OmniRetarget / BeyondMimic
        - same action_dim = 23
        - same obs_dim = 123
        - same training stack = 615
        - extra upper-body and arm-swing style terms
    """

    usd_path: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_USD_PATH",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd",
        )
    )

    motion_file: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_TASK3_MOTION_FILE",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_whole_body_walk.pt",
        )
    )

    # ----------------------------- Control -----------------------------
    ema_alpha_legs: float = 0.55
    ema_alpha_waist: float = 0.45
    ema_alpha_arms: float = 0.45

    leg_action_scale: float = 0.25
    waist_action_scale: float = 0.10
    shoulder_action_scale: float = 0.10
    elbow_action_scale: float = 0.08
    wrist_action_scale: float = 0.05

    max_joint_vel_abs: float = 55.0

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 600_000_000
    resample_command_steps: int = 200
    cmd_smoothing_factor: float = 0.08
    zero_command_prob: float = 0.08

    # Stage 0: keep Task2 low-speed stability, arms frozen
    cmd_vx_stage0: Tuple[float, float] = (0.00, 0.05)
    cmd_vy_stage0: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage0: Tuple[float, float] = (0.00, 0.00)

    # Stage 1: stable forward only
    cmd_vx_stage1: Tuple[float, float] = (0.04, 0.10)
    cmd_vy_stage1: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage1: Tuple[float, float] = (0.00, 0.00)

    # Stage 2: forward + tiny yaw
    cmd_vx_stage2: Tuple[float, float] = (0.08, 0.20)
    cmd_vy_stage2: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage2: Tuple[float, float] = (-0.08, 0.08)

    # Stage 3: mild lateral and turning
    cmd_vx_stage3: Tuple[float, float] = (-0.05, 0.35)
    cmd_vy_stage3: Tuple[float, float] = (-0.08, 0.08)
    cmd_wz_stage3: Tuple[float, float] = (-0.18, 0.18)

    # Stage 4: whole-body full curriculum, still conservative
    cmd_vx_stage4: Tuple[float, float] = (-0.10, 0.60)
    cmd_vy_stage4: Tuple[float, float] = (-0.15, 0.15)
    cmd_wz_stage4: Tuple[float, float] = (-0.30, 0.30)

    rsi_prob_stage0: float = 0.00
    rsi_prob_stage1: float = 0.00
    rsi_prob_stage2: float = 0.05
    rsi_prob_stage3: float = 0.15
    rsi_prob_stage4: float = 0.30

    reference_reset_scale_max: float = 0.35

    # Task2 may use tiny harness. Keep briefly, then remove.
    harness_start: float = 0.04
    harness_end: float = 0.0
    harness_body_id: int = 0

    # ----------------------------- Termination -----------------------------
    target_height: float = 0.75
    fall_height: float = 0.52
    jump_height: float = 1.05
    bad_orientation_xy: float = 0.85

    # ----------------------------- Contact -----------------------------
    contact_force_threshold: float = 8.0
    foot_clearance_target: float = 0.055
    gait_freq_hz: float = 1.45

    # ----------------------------- Reward weights -----------------------------
    # Command / gait
    w_cmd_lin: float = 0.08
    w_cmd_speed: float = 0.42
    w_cmd_yaw: float = 0.11
    w_yaw_drift: float = 0.04
    w_zero_vel: float = 0.010
    w_under_speed: float = 0.20
    w_double_contact: float = 0.08
    w_phase_contact: float = 0.105
    w_air_time: float = 0.085
    w_clearance: float = 0.075

    # Trunk / stability
    w_upright: float = 0.095
    w_height: float = 0.090
    w_base_ang_vel: float = 0.035
    w_base_acc: float = 0.0010
    w_com_support: float = 0.015
    w_z_vel: float = 0.020

    # Whole-body style
    w_ref_pose: float = 0.015
    w_ref_vel: float = 0.006
    w_arm_ref: float = 0.025
    w_arm_vel_ref: float = 0.008
    w_arm_leg_sync: float = 0.025
    w_arm_cross: float = 0.020
    w_default_pose: float = 0.014

    # Safety / efficiency
    w_alive: float = 0.001
    w_joint_limit: float = 0.05
    w_action_rate: float = 0.006
    w_action_mag: float = 0.0015
    w_foot_slip: float = 0.060
    w_energy: float = 0.0012

    penalty_fall: float = -5.0

    # Kernels
    sigma_cmd_lin: float = 28.0
    sigma_cmd_yaw: float = 8.0
    sigma_zero: float = 12.0
    sigma_z: float = 12.0
    sigma_pose: float = 3.0
    sigma_vel: float = 0.08
    sigma_arm: float = 4.0
    sigma_arm_vel: float = 0.08
    deadband_height: float = 0.04

    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 1000.0

    strict_motion_joint_check: bool = True
    print_debug_info: bool = False

    def validate(self) -> None:
        super().validate()
        assert self.num_actions == 23
        assert self.num_observations == 123
        assert self.stacked_obs_dim == 615
        assert self.resample_command_steps > 0
        assert 0.0 <= self.cmd_smoothing_factor <= 1.0
        assert 0.0 <= self.zero_command_prob <= 1.0
        assert self.leg_action_scale > 0.0
        assert self.shoulder_action_scale > 0.0
        assert self.elbow_action_scale > 0.0
        assert self.wrist_action_scale > 0.0
