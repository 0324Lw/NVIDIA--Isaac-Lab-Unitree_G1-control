from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from g1_rl.tasks.task1.task1_env import G1Task1Env
from g1_rl.tasks.task2.task2_config import Task2Config


class G1OmniMotionManager:
    """Omni-directional reference motion manager for G1 Task2.

    Required file keys:
        pos, vel, cmd, num_frames, joint_names, phase,
        contact_ref, mode_id, mode_names
    """

    def __init__(
        self,
        motion_file: str,
        robot_joint_names: List[str],
        controlled_joint_ids: torch.Tensor,
        device: str,
        strict_check: bool = True,
        verbose: bool = False,
    ):
        self.motion_file = str(motion_file)
        self.device = str(device)
        self.robot_joint_names = list(robot_joint_names)
        self.controlled_joint_ids = controlled_joint_ids.to(device=self.device, dtype=torch.long)

        if not os.path.exists(self.motion_file):
            raise FileNotFoundError(
                f"[G1OmniMotionManager] Cannot find motion file: {self.motion_file}\n"
                "Please set G1_TASK2_MOTION_FILE or generate g1_omni_walk.pt first."
            )

        data = torch.load(self.motion_file, map_location=self.device)

        required = [
            "pos",
            "vel",
            "cmd",
            "num_frames",
            "joint_names",
            "phase",
            "contact_ref",
            "mode_id",
            "mode_names",
        ]

        missing = [k for k in required if k not in data]
        if missing:
            raise RuntimeError(
                f"[G1OmniMotionManager] motion file missing keys: {missing}. "
                f"Available keys: {list(data.keys())}"
            )

        self.pos_full = data["pos"].to(device=self.device, dtype=torch.float32)
        self.vel_full = data["vel"].to(device=self.device, dtype=torch.float32)
        self.cmd = data["cmd"].to(device=self.device, dtype=torch.float32)
        self.phase = data["phase"].to(device=self.device, dtype=torch.float32)
        self.contact_ref = data["contact_ref"].to(device=self.device, dtype=torch.float32)
        self.mode_id = data["mode_id"].to(device=self.device, dtype=torch.long)

        self.num_frames = int(data["num_frames"])
        self.joint_names = list(data.get("joint_names", []))
        self.mode_names = list(data.get("mode_names", []))
        self.fps = float(data.get("fps", 50.0))
        self.dt = float(data.get("dt", 1.0 / max(self.fps, 1e-6)))

        if self.pos_full.shape != self.vel_full.shape:
            raise RuntimeError(
                f"[G1OmniMotionManager] pos/vel shape mismatch: "
                f"{tuple(self.pos_full.shape)} vs {tuple(self.vel_full.shape)}"
            )

        if self.pos_full.shape != (self.num_frames, len(self.robot_joint_names)):
            raise RuntimeError(
                f"[G1OmniMotionManager] pos shape should be [T, {len(self.robot_joint_names)}], "
                f"got {tuple(self.pos_full.shape)}"
            )

        if self.cmd.shape != (self.num_frames, 3):
            raise RuntimeError(f"[G1OmniMotionManager] cmd shape should be [T, 3], got {tuple(self.cmd.shape)}")

        if self.phase.shape != (self.num_frames,):
            raise RuntimeError(f"[G1OmniMotionManager] phase shape should be [T], got {tuple(self.phase.shape)}")

        if self.contact_ref.shape != (self.num_frames, 2):
            raise RuntimeError(
                f"[G1OmniMotionManager] contact_ref shape should be [T, 2], "
                f"got {tuple(self.contact_ref.shape)}"
            )

        if self.mode_id.shape != (self.num_frames,):
            raise RuntimeError(f"[G1OmniMotionManager] mode_id shape should be [T], got {tuple(self.mode_id.shape)}")

        if strict_check:
            if self.joint_names != self.robot_joint_names:
                mismatch = [
                    (i, a, b)
                    for i, (a, b) in enumerate(zip(self.joint_names, self.robot_joint_names))
                    if a != b
                ]
                raise RuntimeError(
                    "[G1OmniMotionManager] motion joint_names do not match robot.joint_names. "
                    f"First mismatches: {mismatch[:8]}"
                )

        self.pos_ctrl = self.pos_full[:, self.controlled_joint_ids]
        self.vel_ctrl = self.vel_full[:, self.controlled_joint_ids]
        self.contact_ref = torch.clamp(self.contact_ref, 0.0, 1.0)

        self.num_modes = int(torch.max(self.mode_id).item()) + 1 if self.mode_id.numel() > 0 else 1

        if len(self.mode_names) == 0:
            self.mode_names = [f"mode_{i}" for i in range(self.num_modes)]

        if verbose:
            print("\n" + "=" * 100)
            print(" [G1OmniMotionManager] Omni reference motion loaded")
            print(f" file             : {self.motion_file}")
            print(f" num_frames       : {self.num_frames}")
            print(f" fps              : {self.fps}")
            print(f" full joint dim   : {self.pos_full.shape[1]}")
            print(f" controlled dim   : {self.pos_ctrl.shape[1]}")
            print(f" cmd shape        : {tuple(self.cmd.shape)}")
            print(f" contact_ref      : {tuple(self.contact_ref.shape)}")
            print(f" mode_names       : {self.mode_names}")
            print("=" * 100 + "\n")

    def _phase_to_frame(self, phase: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(phase, 1.0)
        ids = torch.clamp((phase * self.num_frames).long(), 0, self.num_frames - 1)
        return ids

    def sample_initial_state(self, env_ids: torch.Tensor, ref_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_ids = torch.randint(
            low=0,
            high=self.num_frames,
            size=(int(env_ids.numel()),),
            device=self.device,
        )
        q = self.pos_full[frame_ids] * float(ref_scale)
        qd = self.vel_full[frame_ids] * float(ref_scale)
        return q, qd

    def get_reference_by_phase(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_ids = self._phase_to_frame(phase)
        return (
            self.pos_ctrl[frame_ids],
            self.vel_ctrl[frame_ids],
            self.contact_ref[frame_ids],
        )

    def sample_reference_command(self, n: int) -> torch.Tensor:
        ids = torch.randint(0, self.num_frames, (int(n),), device=self.device)
        return self.cmd[ids]


class G1OmniEnv(G1Task1Env):
    """G1 Task2 omni-directional command tracking environment.

    This class reuses the already-tested Task1 G1 physical environment:
        - same USD asset
        - same 25 joints
        - same 23 controlled joints
        - same 123-D observation layout
        - same contact sensor setup
        - same sensor joint hard lock

    Task2 adds:
        - target_cmd [vx, vy, wz]
        - smoothed_cmd
        - command resampling
        - omni motion manager
        - command-conditioned rewards
    """

    def __init__(self, cfg: Task2Config):
        cfg.validate()
        super().__init__(cfg)

        self.cfg: Task2Config = cfg

        # Replace Task1 simple motion manager with Task2 omni motion manager.
        self.motion = G1OmniMotionManager(
            motion_file=str(cfg.motion_file),
            robot_joint_names=self.robot_joint_names,
            controlled_joint_ids=self.controlled_joint_ids_t,
            device=self.device,
            strict_check=bool(cfg.strict_motion_joint_check),
            verbose=bool(cfg.print_debug_info),
        )

        self.target_cmd = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.smoothed_cmd = torch.zeros_like(self.target_cmd)
        self.command_time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._resample_commands(all_ids)
        self.smoothed_cmd.copy_(self.target_cmd)

        # Reset once after replacing motion manager and command buffers.
        self.reset()

    # ------------------------------------------------------------------
    # Curriculum / commands
    # ------------------------------------------------------------------
    def _command_stage(self) -> int:
        k = self.curriculum_k()
        if k < 0.10:
            return 0
        if k < 0.30:
            return 1
        if k < 0.50:
            return 2
        if k < 0.75:
            return 3
        return 4

    def _rsi_probability(self) -> float:
        stage = self._command_stage()
        return [
            float(self.cfg.rsi_prob_stage0),
            float(self.cfg.rsi_prob_stage1),
            float(self.cfg.rsi_prob_stage2),
            float(self.cfg.rsi_prob_stage3),
            float(self.cfg.rsi_prob_stage4),
        ][stage]

    def _style_weight_scale(self) -> float:
        k = self.curriculum_k()
        if k < 0.20:
            return 0.0
        return self._smoothstep((k - 0.20) / 0.50)

    def _reference_reset_scale(self) -> float:
        k = self.curriculum_k()
        if k < 0.10:
            return 0.0
        return float(self.cfg.reference_reset_scale_max) * self._smoothstep((k - 0.10) / 0.60)

    def _harness_ratio(self) -> float:
        k = self.curriculum_k()
        if k < 0.30:
            return float(self.cfg.harness_start)
        if k < 0.70:
            s = self._smoothstep((k - 0.30) / 0.40)
            return float(self.cfg.harness_start) * (1.0 - s)
        return float(self.cfg.harness_end)

    def _curriculum_values(self):
        """Compatibility with Task1Env.step.

        Task1Env.step expects:
            target_vx, harness_ratio, stage

        Task2's real per-env command is stored in target_cmd / smoothed_cmd.
        """
        stage = self._command_stage()
        vx_range, _, _ = self._command_ranges()
        target_vx = 0.5 * (float(vx_range[0]) + float(vx_range[1]))
        return float(target_vx), float(self._harness_ratio()), int(stage)

    def _command_ranges(self):
        stage = self._command_stage()

        if stage == 0:
            return self.cfg.cmd_vx_stage0, self.cfg.cmd_vy_stage0, self.cfg.cmd_wz_stage0
        if stage == 1:
            return self.cfg.cmd_vx_stage1, self.cfg.cmd_vy_stage1, self.cfg.cmd_wz_stage1
        if stage == 2:
            return self.cfg.cmd_vx_stage2, self.cfg.cmd_vy_stage2, self.cfg.cmd_wz_stage2
        if stage == 3:
            return self.cfg.cmd_vx_stage3, self.cfg.cmd_vy_stage3, self.cfg.cmd_wz_stage3
        return self.cfg.cmd_vx_stage4, self.cfg.cmd_vy_stage4, self.cfg.cmd_wz_stage4

    def _sample_commands(self, n: int) -> torch.Tensor:
        vx_range, vy_range, wz_range = self._command_ranges()

        cmd = torch.zeros((int(n), 3), dtype=torch.float32, device=self.device)
        cmd[:, 0] = torch.empty(int(n), device=self.device).uniform_(float(vx_range[0]), float(vx_range[1]))
        cmd[:, 1] = torch.empty(int(n), device=self.device).uniform_(float(vy_range[0]), float(vy_range[1]))
        cmd[:, 2] = torch.empty(int(n), device=self.device).uniform_(float(wz_range[0]), float(wz_range[1]))

        zero = torch.rand(int(n), device=self.device) < float(self.cfg.zero_command_prob)
        cmd[zero] = 0.0

        return cmd

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return

        self.target_cmd[env_ids] = self._sample_commands(int(env_ids.numel()))
        self.command_time_left[env_ids] = float(self.cfg.resample_command_steps) * self.dt

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)

        if hasattr(self, "target_cmd"):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self._resample_commands(env_ids)
            self.smoothed_cmd[env_ids] = self.target_cmd[env_ids]

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        # Command resampling before physics step.
        if hasattr(self, "command_time_left"):
            self.command_time_left -= self.dt
            resample_ids = (self.command_time_left <= 0.0).nonzero(as_tuple=False).squeeze(-1)
            if resample_ids.numel() > 0:
                self._resample_commands(resample_ids)

            alpha = float(self.cfg.cmd_smoothing_factor)
            self.smoothed_cmd = (1.0 - alpha) * self.smoothed_cmd + alpha * self.target_cmd
            self.smoothed_cmd = torch.nan_to_num(self.smoothed_cmd, nan=0.0, posinf=0.0, neginf=0.0)

        return super().step(actions)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _task2_command_obs(self) -> torch.Tensor:
        if hasattr(self, "smoothed_cmd"):
            return self.smoothed_cmd

        # During parent __init__, Task1 reset may call _compute_obs before
        # Task2 buffers are created.
        return torch.stack(
            [
                self.current_target_vx,
                torch.zeros_like(self.current_target_vx),
                torch.zeros_like(self.current_target_vx),
            ],
            dim=-1,
        )

    def _compute_obs(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        q = self.robot.data.joint_pos[:, self.controlled_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.controlled_joint_ids_t]
        q_err = q - self.default_ctrl_pos

        contact, _ = self._get_feet_contact()

        root_pos = self.robot.data.root_pos_w
        root_height = (root_pos[:, 2] - self.env_origins[:, 2]).unsqueeze(-1)

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_rel_pos = (foot_pos - root_pos.unsqueeze(1)).reshape(self.num_envs, -1)

        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]
        foot_vel_xy_flat = foot_vel_xy.reshape(self.num_envs, -1)

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        self.base_acc_obs.copy_(base_acc)

        command = self._task2_command_obs()

        sin_phase = torch.sin(2.0 * math.pi * self.phase).unsqueeze(-1)
        cos_phase = torch.cos(2.0 * math.pi * self.phase).unsqueeze(-1)

        harness = self.current_harness_ratio.unsqueeze(-1)
        action_delta = self.last_action - self.prev_action

        obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                command,
                q_err,
                qd,
                self.last_action,
                action_delta,
                contact,
                foot_rel_pos,
                foot_vel_xy_flat,
                base_acc,
                sin_phase,
                cos_phase,
                harness,
                root_height,
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"[G1OmniEnv] Observation dim mismatch: got {obs.shape[-1]}, "
                f"expected {self.cfg.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------
    def _compute_rewards(self):
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        vx = base_lin_vel[:, 0]
        vy = base_lin_vel[:, 1]
        vz = base_lin_vel[:, 2]

        wx = base_ang_vel[:, 0]
        wy = base_ang_vel[:, 1]
        wz = base_ang_vel[:, 2]

        cmd = self._task2_command_obs()
        cmd_vx = cmd[:, 0]
        cmd_vy = cmd[:, 1]
        cmd_wz = cmd[:, 2]

        root_pos = self.robot.data.root_pos_w
        base_height = root_pos[:, 2] - self.env_origins[:, 2]

        contact, normal_force = self._get_feet_contact()
        contact_count = contact.sum(dim=-1)

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_z = foot_pos[:, :, 2] - self.env_origins[:, 2].unsqueeze(-1)
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]

        q = self.robot.data.joint_pos[:, self.controlled_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.controlled_joint_ids_t]
        q_err = q - self.default_ctrl_pos

        ref_pos, ref_vel, ref_contact = self.motion.get_reference_by_phase(self.phase)
        style_scale = float(self._style_weight_scale())

        lin_error = torch.square(vx - cmd_vx) + torch.square(vy - cmd_vy)
        yaw_error = torch.square(wz - cmd_wz)

        cmd_norm = torch.linalg.norm(cmd, dim=-1)
        moving = (cmd_norm > 0.04).float()
        standing = 1.0 - moving

        r_cmd_lin = torch.exp(-float(self.cfg.sigma_cmd_lin) * lin_error)
        r_cmd_yaw = torch.exp(-float(self.cfg.sigma_cmd_yaw) * yaw_error)

        r_zero_vel = torch.exp(
            -float(self.cfg.sigma_zero)
            * (
                torch.square(vx)
                + torch.square(vy)
                + 0.50 * torch.square(wz)
            )
        )

        actual_speed = torch.sqrt(torch.square(vx) + torch.square(vy) + 1e-6)
        target_speed = torch.sqrt(torch.square(cmd_vx) + torch.square(cmd_vy) + 1e-6)
        r_cmd_speed = torch.exp(-8.0 * torch.square(actual_speed - target_speed)) * moving

        p_under_speed = -torch.relu(target_speed - actual_speed) * moving

        double_contact_penalty = -moving * torch.clamp(contact_count - 1.20, min=0.0)

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += self.dt

        r_air_time = torch.sum(
            torch.clamp(self.feet_air_time - 0.10, min=0.0, max=0.45) * first_contact.float(),
            dim=-1,
        )
        r_air_time = r_air_time * moving

        self.feet_air_time = torch.where(contact > 0.5, torch.zeros_like(self.feet_air_time), self.feet_air_time)
        self.prev_foot_contact.copy_(contact)

        r_clearance = (
            (1.0 - contact)
            * torch.exp(-20.0 * torch.abs(foot_z - float(self.cfg.foot_clearance_target)))
        ).sum(dim=-1)
        r_clearance = r_clearance * moving

        r_phase_contact = 1.0 - torch.mean(torch.abs(contact - ref_contact), dim=-1)
        r_phase_contact = r_phase_contact * moving

        r_upright = (1.0 - projected_gravity[:, 2]) * 0.5

        h_err = torch.clamp(
            torch.abs(base_height - float(self.cfg.target_height)) - float(self.cfg.deadband_height),
            min=0.0,
        )
        r_height = torch.exp(-float(self.cfg.sigma_z) * torch.square(h_err))

        p_base_ang = -(torch.square(wx) + torch.square(wy))
        p_z_vel = -torch.square(vz)

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        p_base_acc = -torch.clamp(torch.sum(torch.square(base_acc), dim=-1), max=30.0)
        self.last_base_vel.copy_(base_lin_vel)

        r_com_support = torch.exp(-1.5 * torch.abs(vy)) * (contact_count > 0.5).float()

        p_default_pose = -torch.mean(torch.square(q_err), dim=-1)

        r_ref_pose = (
            torch.exp(-float(self.cfg.sigma_pose) * torch.mean(torch.square(q_err - ref_pos), dim=-1))
            * style_scale
        )

        r_ref_vel = (
            torch.exp(-float(self.cfg.sigma_vel) * torch.mean(torch.square(qd - ref_vel), dim=-1))
            * style_scale
        )

        r_alive = torch.ones_like(vx)

        lower_margin = q - self.ctrl_lower
        upper_margin = self.ctrl_upper - q
        p_joint_limit = -torch.mean(
            torch.square(torch.clamp(0.05 - lower_margin, min=0.0))
            + torch.square(torch.clamp(0.05 - upper_margin, min=0.0)),
            dim=-1,
        )

        p_action_rate = -torch.mean(torch.square(self.last_action - self.prev_action), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.last_action), dim=-1)

        p_slip = -torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * contact, dim=-1)

        tau_full = getattr(self.robot.data, "applied_torque", torch.zeros_like(self.robot.data.joint_vel))
        tau = tau_full[:, self.controlled_joint_ids_t]
        p_energy = -torch.mean(torch.abs(tau * qd), dim=-1)

        continuous_raw = (
            float(self.cfg.w_cmd_lin) * (moving * r_cmd_lin + standing * r_zero_vel)
            + float(self.cfg.w_cmd_yaw) * r_cmd_yaw
            + float(self.cfg.w_zero_vel) * r_zero_vel * standing
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_cmd_speed) * r_cmd_speed
            + float(self.cfg.w_double_contact) * double_contact_penalty
            + float(self.cfg.w_under_speed) * p_under_speed
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_base_ang_vel) * p_base_ang
            + float(self.cfg.w_base_acc) * p_base_acc
            + float(self.cfg.w_com_support) * r_com_support
            + float(self.cfg.w_z_vel) * p_z_vel
            + float(self.cfg.w_default_pose) * p_default_pose
            + float(self.cfg.w_ref_pose) * r_ref_pose
            + float(self.cfg.w_ref_vel) * r_ref_vel
            + float(self.cfg.w_alive) * r_alive
            + float(self.cfg.w_joint_limit) * p_joint_limit
            + float(self.cfg.w_action_rate) * p_action_rate
            + float(self.cfg.w_action_mag) * p_action_mag
            + float(self.cfg.w_foot_slip) * p_slip
            + float(self.cfg.w_energy) * p_energy
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        joint_vel_abs_max = torch.abs(self.robot.data.joint_vel).max(dim=-1)[0]
        roll_pitch_mag = torch.norm(projected_gravity[:, :2], dim=-1)

        is_fallen = (
            (base_height < float(self.cfg.fall_height))
            | (base_height > float(self.cfg.jump_height))
            | (roll_pitch_mag > float(self.cfg.bad_orientation_xy))
            | (~torch.isfinite(base_height))
            | (~torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
            | (joint_vel_abs_max > float(self.cfg.max_joint_vel_abs))
        )

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        event_fall = torch.where(
            is_fallen,
            torch.full_like(continuous, float(self.cfg.penalty_fall)),
            torch.zeros_like(continuous),
        )

        reward_raw = continuous + event_fall

        projected_return = self.episode_return + reward_raw
        no_event = event_fall.abs() < 1e-6

        reward = torch.where(
            (projected_return > float(self.cfg.episode_return_abs_limit)) & no_event,
            float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward_raw,
        )

        reward = torch.where(
            (projected_return < -float(self.cfg.episode_return_abs_limit)) & no_event,
            -float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward,
        )

        terminated = is_fallen
        truncated = timeout & (~terminated)
        done = terminated | truncated

        if done.any():
            done_f = done.float()
            self.total_done_episodes += done_f.sum()
            self.total_fall_episodes += terminated.float().sum()
            self.total_timeout_episodes += truncated.float().sum()

        total_done_safe = torch.clamp(self.total_done_episodes, min=1.0)
        stage = self._command_stage()

        info = {
            "reward_components": {
                "R_Cmd_Lin": self._mean_detached(float(self.cfg.w_cmd_lin) * (moving * r_cmd_lin + standing * r_zero_vel)),
                "R_Cmd_Yaw": self._mean_detached(float(self.cfg.w_cmd_yaw) * r_cmd_yaw),
                "R_Zero_Vel": self._mean_detached(float(self.cfg.w_zero_vel) * r_zero_vel * standing),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Cmd_Speed": self._mean_detached(float(self.cfg.w_cmd_speed) * r_cmd_speed),
                "P_Double_Contact": self._mean_detached(float(self.cfg.w_double_contact) * double_contact_penalty),
                "P_Under_Speed": self._mean_detached(float(self.cfg.w_under_speed) * p_under_speed),
                "R_Upright": self._mean_detached(float(self.cfg.w_upright) * r_upright),
                "R_Height": self._mean_detached(float(self.cfg.w_height) * r_height),
                "P_Base_Ang": self._mean_detached(float(self.cfg.w_base_ang_vel) * p_base_ang),
                "P_Base_Acc": self._mean_detached(float(self.cfg.w_base_acc) * p_base_acc),
                "P_Z_Vel": self._mean_detached(float(self.cfg.w_z_vel) * p_z_vel),
                "R_COM_Support": self._mean_detached(float(self.cfg.w_com_support) * r_com_support),
                "P_Default_Pose": self._mean_detached(float(self.cfg.w_default_pose) * p_default_pose),
                "R_Ref_Pose": self._mean_detached(float(self.cfg.w_ref_pose) * r_ref_pose),
                "R_Ref_Vel": self._mean_detached(float(self.cfg.w_ref_vel) * r_ref_vel),
                "R_Alive": self._mean_detached(float(self.cfg.w_alive) * r_alive),
                "P_Joint_Limit": self._mean_detached(float(self.cfg.w_joint_limit) * p_joint_limit),
                "P_Action_Rate": self._mean_detached(float(self.cfg.w_action_rate) * p_action_rate),
                "P_Action_Mag": self._mean_detached(float(self.cfg.w_action_mag) * p_action_mag),
                "P_Foot_Slip": self._mean_detached(float(self.cfg.w_foot_slip) * p_slip),
                "P_Energy": self._mean_detached(float(self.cfg.w_energy) * p_energy),
                "Continuous": self._mean_detached(continuous),
                "Event_Fall": self._mean_detached(event_fall),
                "Total": self._mean_detached(reward),
            },
            "events": {
                "Fall_Rate": self._mean_detached(terminated.float()),
                "Timeout_Rate": self._mean_detached(truncated.float()),
                "Done_Rate": self._mean_detached(done.float()),
                "Episode_Fall_Total_Rate": self.total_fall_episodes / total_done_safe,
                "Episode_Timeout_Total_Rate": self.total_timeout_episodes / total_done_safe,
            },
            "telemetry": {
                "Curriculum_K": self._float_tensor(self.curriculum_k()),
                "Command_Stage": self._float_tensor(float(stage)),
                "Cmd_Vx": self._mean_detached(cmd_vx),
                "Cmd_Vy": self._mean_detached(cmd_vy),
                "Cmd_Wz": self._mean_detached(cmd_wz),
                "Actual_Vx": self._mean_detached(vx),
                "Actual_Vy": self._mean_detached(vy),
                "Actual_Wz": self._mean_detached(wz),
                "Lin_Error": self._mean_detached(torch.sqrt(lin_error + 1e-6)),
                "Yaw_Error": self._mean_detached(torch.sqrt(yaw_error + 1e-6)),
                "Base_Height": self._mean_detached(base_height),
                "RollPitch_Mag": self._mean_detached(roll_pitch_mag),
                "Harness_Ratio": self._float_tensor(float(self._harness_ratio())),
                "Contact_Count": self._mean_detached(contact_count),
                "Left_Contact": self._mean_detached(contact[:, 0]),
                "Right_Contact": self._mean_detached(contact[:, 1]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Style_Scale": self._float_tensor(float(style_scale)),
                "RSI_Prob": self._float_tensor(float(self._rsi_probability())),
                "Foot_Slip_Raw": self._mean_detached(-p_slip),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Global_Steps": self._float_tensor(float(self.global_steps)),
            },
            "debug": {
                "Obs_Dim": self._float_tensor(float(self.cfg.num_observations)),
                "Action_Dim": self._float_tensor(float(self.num_actions)),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Base_Height_Min": base_height.detach().min(),
                "Base_Height_Max": base_height.detach().max(),
                "JointVel_Max": joint_vel_abs_max.detach().max(),
                "Motion_Num_Frames": self._float_tensor(float(self.motion.num_frames)),
            },
        }

        return reward, terminated, truncated, info


# Backward-compatible aliases
Task2Env = G1OmniEnv
G1Task2Env = G1OmniEnv
