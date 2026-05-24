from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

from g1_rl.common.paths import asset_path_from_env
from g1_rl.tasks.task2.task2_config import Task2Config


def _task4_motion_file() -> str:
    """Resolve Task4 motion file.

    Task4 is a Sim2Real robustness task. It does not require a new whole-body
    reference file. By default it reuses the Task2 omni walking reference.
    """
    if os.environ.get("G1_TASK4_MOTION_FILE"):
        return os.environ["G1_TASK4_MOTION_FILE"]

    if os.environ.get("G1_TASK2_MOTION_FILE"):
        return os.environ["G1_TASK2_MOTION_FILE"]

    return "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"


@dataclass
class Task4Config(Task2Config):
    """G1 Task4 Sim2Real robustness config.

    Task4 keeps the Task2 command locomotion structure but adds robustness
    mechanisms:
        - motor efficiency randomization
        - actuator lag scaling
        - action delay / observation delay
        - action deadzone / noise / quantization
        - payload force proxy
        - friction / slip-stress proxy
        - IMU / joint / height / foot noise
        - contact dropout / false positives
        - external pushes
        - privileged critic state

    This is still an educational pure-RL baseline.
    """

    usd_path: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_USD_PATH",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd",
        )
    )

    motion_file: str = field(default_factory=_task4_motion_file)

    # ----------------------------- Spaces -----------------------------
    num_actions: int = 23
    num_observations: int = 123
    num_privileged_obs: int = 162
    frame_stack: int = 5
    stacked_obs_dim: int = 615

    # ----------------------------- Control -----------------------------
    ema_alpha_legs: float = 0.55
    ema_alpha_waist: float = 0.45
    ema_alpha_arms: float = 0.45

    leg_action_scale: float = 0.25
    waist_action_scale: float = 0.10
    shoulder_action_scale: float = 0.08
    elbow_action_scale: float = 0.06
    wrist_action_scale: float = 0.04

    max_joint_vel_abs: float = 55.0

    # ----------------------------- Low-speed command curriculum -----------------------------
    curriculum_total_steps: int = 400_000_000
    resample_command_steps: int = 200
    cmd_smoothing_factor: float = 0.08
    zero_command_prob: float = 0.08

    # Task4 is robust low-speed locomotion, not high-speed running.
    cmd_vx_stage0: Tuple[float, float] = (0.00, 0.05)
    cmd_vy_stage0: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage0: Tuple[float, float] = (0.00, 0.00)

    cmd_vx_stage1: Tuple[float, float] = (0.03, 0.10)
    cmd_vy_stage1: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage1: Tuple[float, float] = (0.00, 0.00)

    cmd_vx_stage2: Tuple[float, float] = (0.04, 0.15)
    cmd_vy_stage2: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage2: Tuple[float, float] = (-0.06, 0.06)

    cmd_vx_stage3: Tuple[float, float] = (-0.04, 0.18)
    cmd_vy_stage3: Tuple[float, float] = (-0.06, 0.06)
    cmd_wz_stage3: Tuple[float, float] = (-0.12, 0.12)

    cmd_vx_stage4: Tuple[float, float] = (-0.08, 0.22)
    cmd_vy_stage4: Tuple[float, float] = (-0.10, 0.10)
    cmd_wz_stage4: Tuple[float, float] = (-0.20, 0.20)

    # ----------------------------- Sim2Real DR curriculum -----------------------------
    motor_efficiency_range: Tuple[float, float] = (0.70, 1.10)
    alpha_scale_range: Tuple[float, float] = (0.70, 1.15)

    action_deadzone_range: Tuple[float, float] = (0.00, 0.035)
    action_noise_std_max: float = 0.025
    action_quantization: float = 0.0025

    payload_mass_range: Tuple[float, float] = (0.0, 4.0)
    terrain_friction_range: Tuple[float, float] = (0.45, 1.40)

    action_delay_steps_max: int = 4
    obs_delay_steps_max: int = 2

    imu_noise_std_max: float = 0.035
    joint_pos_noise_std_max: float = 0.012
    joint_vel_noise_std_max: float = 0.08
    root_height_noise_std_max: float = 0.008
    foot_pos_noise_std_max: float = 0.006

    imu_bias_walk_std: float = 0.00035
    joint_bias_walk_std: float = 0.00015

    state_dropout_prob_max: float = 0.035
    contact_dropout_prob_max: float = 0.08
    contact_false_positive_prob_max: float = 0.015

    push_prob_per_step_max: float = 0.003
    push_force_range: Tuple[float, float] = (20.0, 90.0)
    push_duration_steps_range: Tuple[int, int] = (2, 8)

    slip_stress_force_max: float = 25.0

    # ----------------------------- Termination -----------------------------
    target_height: float = 0.75
    fall_height: float = 0.52
    jump_height: float = 1.05
    bad_orientation_xy: float = 0.85

    # ----------------------------- Contact / phase -----------------------------
    contact_force_threshold: float = 8.0
    foot_clearance_target: float = 0.055
    gait_freq_hz: float = 1.45
    contact_duty_ratio: float = 0.62

    # ----------------------------- Reward weights -----------------------------
    # Task / disturbance robustness
    w_cmd_lin: float = 0.08
    w_cmd_speed: float = 0.42
    w_cmd_yaw: float = 0.10
    w_zero_vel: float = 0.010
    w_under_speed: float = 0.20
    w_yaw_drift: float = 0.040
    w_double_contact: float = 0.08
    w_phase_contact: float = 0.105
    w_air_time: float = 0.085
    w_clearance: float = 0.075
    w_recovery: float = 0.080
    w_push_survival: float = 0.035

    # Trunk / safety
    w_upright: float = 0.095
    w_height: float = 0.090
    w_base_ang_vel: float = 0.035
    w_base_acc: float = 0.0010
    w_com_support: float = 0.015
    w_z_vel: float = 0.020
    w_over_speed: float = 0.12

    # Hardware / efficiency / smoothness
    w_default_pose: float = 0.014
    w_alive: float = 0.001
    w_joint_limit: float = 0.05
    w_action_rate: float = 0.006
    w_action_mag: float = 0.0015
    w_foot_slip: float = 0.060
    w_energy: float = 0.0012
    w_motor_temp: float = 0.002

    penalty_fall: float = -5.0

    # Kernels
    sigma_cmd_lin: float = 28.0
    sigma_cmd_yaw: float = 8.0
    sigma_zero: float = 12.0
    sigma_z: float = 12.0
    deadband_height: float = 0.04

    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 1000.0

    print_debug_info: bool = False

    def validate(self) -> None:
        super().validate()

        assert self.num_actions == 23
        assert self.num_observations == 123
        assert self.num_privileged_obs == 162
        assert self.stacked_obs_dim == 615

        assert self.curriculum_total_steps > 0
        assert self.resample_command_steps > 0
        assert 0.0 <= self.cmd_smoothing_factor <= 1.0
        assert 0.0 <= self.zero_command_prob <= 1.0

        assert self.action_delay_steps_max >= 0
        assert self.obs_delay_steps_max >= 0
        assert self.action_quantization >= 0.0
        assert self.action_noise_std_max >= 0.0

        assert self.payload_mass_range[0] <= self.payload_mass_range[1]
        assert self.terrain_friction_range[0] <= self.terrain_friction_range[1]
        assert self.push_force_range[0] <= self.push_force_range[1]
        assert self.push_duration_steps_range[0] <= self.push_duration_steps_range[1]

        assert self.fall_height < self.target_height < self.jump_height
