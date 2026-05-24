from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from g1_rl.common.paths import asset_path_from_env
from g1_rl.tasks.task1.task1_config import Task1Config


@dataclass
class Task2Config(Task1Config):
    """G1 Task2 omni-directional locomotion config.

    Task2 is still part of the pure-RL baseline project:
        - no world file
        - no imitation-learning policy
        - same 23-DoF action space as Task1
        - same single-frame obs dim = 123
        - training script later uses 5-frame stack = 615
    """

    usd_path: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_USD_PATH",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd",
        )
    )

    motion_file: str = field(
        default_factory=lambda: asset_path_from_env(
            "G1_TASK2_MOTION_FILE",
            "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt",
        )
    )

    # ----------------------------- Command curriculum -----------------------------
    curriculum_total_steps: int = 500_000_000
    resample_command_steps: int = 200
    cmd_smoothing_factor: float = 0.08
    zero_command_prob: float = 0.10

    # Stage 0: small forward only
    cmd_vx_stage0: Tuple[float, float] = (0.00, 0.03)
    cmd_vy_stage0: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage0: Tuple[float, float] = (0.00, 0.00)

    # Stage 1: forward / slight backward
    cmd_vx_stage1: Tuple[float, float] = (0.04, 0.22)
    cmd_vy_stage1: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage1: Tuple[float, float] = (0.00, 0.00)

    # Stage 2: forward/backward + yaw
    cmd_vx_stage2: Tuple[float, float] = (-0.08, 0.35)
    cmd_vy_stage2: Tuple[float, float] = (0.00, 0.00)
    cmd_wz_stage2: Tuple[float, float] = (-0.12, 0.12)

    # Stage 3: add lateral
    cmd_vx_stage3: Tuple[float, float] = (-0.25, 0.50)
    cmd_vy_stage3: Tuple[float, float] = (-0.12, 0.12)
    cmd_wz_stage3: Tuple[float, float] = (-0.25, 0.25)

    # Stage 4: full omni
    cmd_vx_stage4: Tuple[float, float] = (-0.35, 0.60)
    cmd_vy_stage4: Tuple[float, float] = (-0.25, 0.25)
    cmd_wz_stage4: Tuple[float, float] = (-0.40, 0.40)

    # ----------------------------- RSI / style curriculum -----------------------------
    rsi_prob_stage0: float = 0.00
    rsi_prob_stage1: float = 0.10
    rsi_prob_stage2: float = 0.25
    rsi_prob_stage3: float = 0.45
    rsi_prob_stage4: float = 0.65

    reference_reset_scale_max: float = 0.40

    # Task2 starts from a Task1 model that may still use small harness.
    # Keep tiny harness early, then remove it.
    harness_start: float = 0.08
    harness_end: float = 0.0

    # ----------------------------- Body / termination -----------------------------
    target_height: float = 0.75
    fall_height: float = 0.52
    jump_height: float = 1.05
    bad_orientation_xy: float = 0.85

    # ----------------------------- Reward weights -----------------------------
    # command-conditioned locomotion
    w_cmd_lin: float = 0.16
    w_cmd_yaw: float = 0.10
    w_zero_vel: float = 0.020
    w_phase_contact: float = 0.115
    w_air_time: float = 0.105
    w_clearance: float = 0.090
    w_cmd_speed: float = 0.22
    w_double_contact: float = 0.14
    w_under_speed: float = 0.12

    # trunk / stability
    w_upright: float = 0.115
    w_height: float = 0.105
    w_base_ang_vel: float = 0.040
    w_base_acc: float = 0.0010
    w_com_support: float = 0.020
    w_z_vel: float = 0.020

    # safety / efficiency / style
    w_default_pose: float = 0.018
    w_ref_pose: float = 0.025
    w_ref_vel: float = 0.010
    w_alive: float = 0.0015
    w_joint_limit: float = 0.05
    w_action_rate: float = 0.006
    w_action_mag: float = 0.0015
    w_foot_slip: float = 0.065
    w_energy: float = 0.0012

    # Event reward
    penalty_fall: float = -5.0

    # Reward kernels
    sigma_cmd_lin: float = 18.0
    sigma_cmd_yaw: float = 5.0
    sigma_zero: float = 10.0
    sigma_z: float = 12.0
    sigma_pose: float = 3.0
    sigma_vel: float = 0.08
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
