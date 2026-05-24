from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

from g1_rl.tasks.task2.task2_env import G1OmniEnv
from g1_rl.tasks.task3.task3_config import Task3Config


class G1WholeBodyMotionManager:
    """Whole-body synthetic reference manager for G1 Task3.

    Required motion keys:
        pos, vel, cmd, num_frames, phase, contact_ref,
        mode_id, mode_names, joint_names, arm_swing_ref

    Important:
        This is only a joint-name-aligned synthetic whole-body reference.
        It is not AMP / AMASS / HoloSoma / OmniRetarget / BeyondMimic.
    """

    def __init__(
        self,
        motion_file: str,
        robot_joint_names: List[str],
        controlled_joint_ids: torch.Tensor,
        arm_joint_ids: torch.Tensor,
        device: str,
        strict_check: bool = True,
        verbose: bool = False,
    ):
        self.motion_file = str(motion_file)
        self.device = str(device)
        self.robot_joint_names = list(robot_joint_names)
        self.controlled_joint_ids = controlled_joint_ids.to(device=self.device, dtype=torch.long)
        self.arm_joint_ids = arm_joint_ids.to(device=self.device, dtype=torch.long)

        if not os.path.exists(self.motion_file):
            raise FileNotFoundError(
                f"[G1WholeBodyMotionManager] Cannot find motion file: {self.motion_file}\n"
                "Please generate Task3 whole-body motion file first, or set G1_TASK3_MOTION_FILE."
            )

        data = torch.load(self.motion_file, map_location=self.device)

        required = [
            "pos",
            "vel",
            "cmd",
            "num_frames",
            "phase",
            "contact_ref",
            "mode_id",
            "mode_names",
            "joint_names",
            "arm_swing_ref",
        ]

        missing = [k for k in required if k not in data]
        if missing:
            raise RuntimeError(
                f"[G1WholeBodyMotionManager] motion file missing keys: {missing}. "
                f"Available keys: {list(data.keys())}"
            )

        self.pos_full = data["pos"].to(device=self.device, dtype=torch.float32)
        self.vel_full = data["vel"].to(device=self.device, dtype=torch.float32)
        self.cmd = data["cmd"].to(device=self.device, dtype=torch.float32)
        self.phase = data["phase"].to(device=self.device, dtype=torch.float32)
        self.contact_ref = data["contact_ref"].to(device=self.device, dtype=torch.float32)
        self.mode_id = data["mode_id"].to(device=self.device, dtype=torch.long)
        self.arm_swing_ref = data["arm_swing_ref"].to(device=self.device, dtype=torch.float32)

        self.mode_names = list(data.get("mode_names", []))
        self.joint_names = list(data.get("joint_names", []))
        self.num_frames = int(data["num_frames"])
        self.fps = float(data.get("fps", 50.0))
        self.dt = float(data.get("dt", 1.0 / max(self.fps, 1e-6)))

        if self.pos_full.shape != self.vel_full.shape:
            raise RuntimeError(
                f"[G1WholeBodyMotionManager] pos/vel shape mismatch: "
                f"{tuple(self.pos_full.shape)} vs {tuple(self.vel_full.shape)}"
            )

        if self.pos_full.shape != (self.num_frames, len(self.robot_joint_names)):
            raise RuntimeError(
                f"[G1WholeBodyMotionManager] pos shape should be "
                f"[{self.num_frames}, {len(self.robot_joint_names)}], "
                f"got {tuple(self.pos_full.shape)}"
            )

        if self.cmd.shape != (self.num_frames, 3):
            raise RuntimeError(f"[G1WholeBodyMotionManager] cmd shape should be [T, 3], got {tuple(self.cmd.shape)}")

        if self.phase.shape != (self.num_frames,):
            raise RuntimeError(f"[G1WholeBodyMotionManager] phase shape should be [T], got {tuple(self.phase.shape)}")

        if self.contact_ref.shape != (self.num_frames, 2):
            raise RuntimeError(
                f"[G1WholeBodyMotionManager] contact_ref shape should be [T, 2], "
                f"got {tuple(self.contact_ref.shape)}"
            )

        if self.mode_id.shape != (self.num_frames,):
            raise RuntimeError(f"[G1WholeBodyMotionManager] mode_id shape should be [T], got {tuple(self.mode_id.shape)}")

        if strict_check:
            if self.joint_names != self.robot_joint_names:
                mismatch = [
                    (i, a, b)
                    for i, (a, b) in enumerate(zip(self.joint_names, self.robot_joint_names))
                    if a != b
                ]
                raise RuntimeError(
                    "[G1WholeBodyMotionManager] motion joint_names do not match robot.joint_names. "
                    f"First mismatches: {mismatch[:8]}"
                )

        self.pos_ctrl = self.pos_full[:, self.controlled_joint_ids]
        self.vel_ctrl = self.vel_full[:, self.controlled_joint_ids]
        self.pos_arm = self.pos_full[:, self.arm_joint_ids]
        self.vel_arm = self.vel_full[:, self.arm_joint_ids]

        if self.arm_swing_ref.ndim != 2:
            raise RuntimeError(
                f"[G1WholeBodyMotionManager] arm_swing_ref should be 2-D, "
                f"got {tuple(self.arm_swing_ref.shape)}"
            )

        if self.arm_swing_ref.shape[0] != self.num_frames:
            raise RuntimeError(
                f"[G1WholeBodyMotionManager] arm_swing_ref first dim should be num_frames, "
                f"got {tuple(self.arm_swing_ref.shape)}"
            )

        self.contact_ref = torch.clamp(self.contact_ref, 0.0, 1.0)

        self.num_modes = int(torch.max(self.mode_id).item()) + 1 if self.mode_id.numel() > 0 else 1

        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            self.frames_per_mode = int(metadata.get("frames_per_mode", max(1, self.num_frames // self.num_modes)))
        else:
            self.frames_per_mode = max(1, self.num_frames // self.num_modes)

        if len(self.mode_names) == 0:
            self.mode_names = [f"mode_{i}" for i in range(self.num_modes)]

        self.mode_cmds = torch.zeros((self.num_modes, 3), dtype=torch.float32, device=self.device)

        for i in range(self.num_modes):
            mask = self.mode_id == i
            if mask.any():
                self.mode_cmds[i] = self.cmd[mask].mean(dim=0)

        if verbose:
            print("\n" + "=" * 100)
            print(" [G1WholeBodyMotionManager] Whole-body reference loaded")
            print(f" file             : {self.motion_file}")
            print(f" motion type      : {data.get('motion_type', 'unknown')}")
            print(f" num_frames       : {self.num_frames}")
            print(f" num_modes        : {self.num_modes}")
            print(f" frames_per_mode  : {self.frames_per_mode}")
            print(f" full joint dim   : {self.pos_full.shape[1]}")
            print(f" controlled dim   : {self.pos_ctrl.shape[1]}")
            print(f" arm dim          : {self.pos_arm.shape[1]}")
            print(f" arm_swing_ref    : {tuple(self.arm_swing_ref.shape)}")
            print(f" cmd shape        : {tuple(self.cmd.shape)}")
            print(f" contact_ref      : {tuple(self.contact_ref.shape)}")
            print(f" mode_names       : {self.mode_names}")
            print("=" * 100 + "\n")

    def _phase_to_frame(self, phase: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(phase, 1.0)
        return torch.clamp(
            (phase * self.frames_per_mode).long(),
            0,
            self.frames_per_mode - 1,
        )

    def _cmd_to_mode(self, cmd: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.norm(cmd.unsqueeze(1) - self.mode_cmds.unsqueeze(0), dim=-1)
        return torch.argmin(dist, dim=1)

    def sample_initial_state(self, env_ids: torch.Tensor, ref_scale: float):
        """Return reset reference in the parent-env compatible format.

        Important:
            G1OmniEnv / Task2 reset logic expects exactly:
                q, qd = motion.sample_initial_state(...)

            Task3 still stores cmd / phase internally, but they must not be
            returned here, otherwise parent reset will raise:
                ValueError: too many values to unpack (expected 2)
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        frame_ids = torch.randint(0, self.num_frames, (int(env_ids.numel()),), device=self.device)

        q = self.pos_full[frame_ids] * float(ref_scale)
        qd = self.vel_full[frame_ids] * float(ref_scale)

        return q, qd

    def sample_initial_state_full(self, env_ids: torch.Tensor, ref_scale: float):
        """Optional Task3-only full reset reference.

        This method is not called by the parent reset logic. It is kept for
        future Task3-specific extensions if we want to explicitly initialize
        cmd / phase from the whole-body motion library.
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        frame_ids = torch.randint(0, self.num_frames, (int(env_ids.numel()),), device=self.device)

        q = self.pos_full[frame_ids] * float(ref_scale)
        qd = self.vel_full[frame_ids] * float(ref_scale)
        cmd = self.cmd[frame_ids]
        phase = self.phase[frame_ids]

        return q, qd, cmd, phase

    def get_reference_by_cmd_phase(self, cmd: torch.Tensor, phase: torch.Tensor):
        mode = self._cmd_to_mode(cmd)
        local_frame = self._phase_to_frame(phase)
        frame_ids = torch.clamp(mode * self.frames_per_mode + local_frame, 0, self.num_frames - 1)

        return (
            self.pos_ctrl[frame_ids],
            self.vel_ctrl[frame_ids],
            self.contact_ref[frame_ids],
            self.pos_arm[frame_ids],
            self.vel_arm[frame_ids],
            self.arm_swing_ref[frame_ids],
        )

    def get_reference_by_phase(self, phase: torch.Tensor):
        """Compatibility fallback for Task2-style callers."""
        frame_ids = self._phase_to_frame(phase)
        return (
            self.pos_ctrl[frame_ids],
            self.vel_ctrl[frame_ids],
            self.contact_ref[frame_ids],
        )


class G1WholeBodyEnv(G1OmniEnv):
    """G1 Task3 whole-body pure-RL environment.

    This environment reuses Task2's already-tested physical pipeline:
        USD / contact / reset / observation / command buffers

    Task3 adds:
        - whole-body motion manager
        - upper-body action gain curriculum
        - per-group action scale
        - arm reference reward
        - arm-leg synchronization reward
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task3Config):
        cfg.validate()

        self._task3_cfg_prepared = False

        super().__init__(cfg)

        self.cfg: Task3Config = cfg

        # Build joint / action groups again in a task-local robust way.
        self.leg_joint_ids_t = self._task3_joint_tensor(cfg.leg_joint_names)
        self.waist_joint_ids_t = self._task3_joint_tensor(cfg.waist_joint_names)
        self.arm_joint_ids_t = self._task3_joint_tensor(cfg.arm_joint_names)

        self.leg_action_ids_t = self._task3_action_tensor(cfg.leg_joint_names)
        self.waist_action_ids_t = self._task3_action_tensor(cfg.waist_joint_names)
        self.arm_action_ids_t = self._task3_action_tensor(cfg.arm_joint_names)

        # Replace Task2 omni reference manager with Task3 whole-body manager.
        self.motion = G1WholeBodyMotionManager(
            motion_file=str(cfg.motion_file),
            robot_joint_names=self.robot_joint_names,
            controlled_joint_ids=self.controlled_joint_ids_t,
            arm_joint_ids=self.arm_joint_ids_t,
            device=self.device,
            strict_check=bool(cfg.strict_motion_joint_check),
            verbose=bool(cfg.print_debug_info),
        )

        # Override action scale / EMA with whole-body grouped values.
        self.action_scale = self._make_task3_action_scale()
        self.ema_alpha_tensor = self._make_task3_ema_alpha_tensor()

        self.current_arm_gain = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._task3_cfg_prepared = True

        self.reset()

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------
    def _task3_joint_tensor(self, names: List[str]) -> torch.Tensor:
        missing = [name for name in names if name not in self.robot_joint_names]

        if missing:
            raise RuntimeError(f"[G1WholeBodyEnv] Missing joints: {missing}")

        ids = [self.robot_joint_names.index(name) for name in names]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _task3_action_tensor(self, names: List[str]) -> torch.Tensor:
        action_ids = []

        for name in names:
            if name in self.cfg.controlled_joint_names:
                action_ids.append(self.cfg.controlled_joint_names.index(name))

        return torch.tensor(action_ids, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def _command_stage(self) -> int:
        k = self.curriculum_k() if hasattr(self, "curriculum_k") else self._task3_curriculum_k()

        if k < 0.08:
            return 0
        if k < 0.25:
            return 1
        if k < 0.48:
            return 2
        if k < 0.72:
            return 3
        return 4

    def _task3_curriculum_k(self) -> float:
        return min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0))

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
        k = self.curriculum_k() if hasattr(self, "curriculum_k") else self._task3_curriculum_k()
        if k < 0.18:
            return 0.0
        return self._smoothstep((k - 0.18) / 0.55)

    def _arm_action_gain(self) -> float:
        k = self.curriculum_k() if hasattr(self, "curriculum_k") else self._task3_curriculum_k()

        if k < 0.18:
            return 0.0
        if k < 0.35:
            return 0.20 * self._smoothstep((k - 0.18) / 0.17)
        if k < 0.65:
            return 0.20 + 0.50 * self._smoothstep((k - 0.35) / 0.30)

        return 1.0

    def _reference_reset_scale(self) -> float:
        k = self.curriculum_k() if hasattr(self, "curriculum_k") else self._task3_curriculum_k()
        if k < 0.12:
            return 0.0
        return float(self.cfg.reference_reset_scale_max) * self._smoothstep((k - 0.12) / 0.55)

    def _harness_ratio(self) -> float:
        k = self.curriculum_k() if hasattr(self, "curriculum_k") else self._task3_curriculum_k()

        if k < 0.25:
            return float(self.cfg.harness_start)
        if k < 0.55:
            s = self._smoothstep((k - 0.25) / 0.30)
            return float(self.cfg.harness_start) * (1.0 - s)

        return float(self.cfg.harness_end)

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

    # ------------------------------------------------------------------
    # Action scale / EMA
    # ------------------------------------------------------------------
    def _make_task3_action_scale(self) -> torch.Tensor:
        scale = torch.full(
            (self.num_actions,),
            float(self.cfg.shoulder_action_scale),
            dtype=torch.float32,
            device=self.device,
        )

        for i, jid in enumerate(self.controlled_joint_ids):
            name = self.robot_joint_names[jid]

            if name in self.cfg.leg_joint_names:
                scale[i] = float(self.cfg.leg_action_scale)
            elif name in self.cfg.waist_joint_names:
                scale[i] = float(self.cfg.waist_action_scale)
            elif "elbow" in name:
                scale[i] = float(self.cfg.elbow_action_scale)
            elif "wrist" in name:
                scale[i] = float(self.cfg.wrist_action_scale)
            else:
                scale[i] = float(self.cfg.shoulder_action_scale)

        return scale

    def _make_task3_ema_alpha_tensor(self) -> torch.Tensor:
        alpha = torch.full(
            (self.num_envs, self.num_actions),
            float(self.cfg.ema_alpha_legs),
            dtype=torch.float32,
            device=self.device,
        )

        if self.waist_action_ids_t.numel() > 0:
            alpha[:, self.waist_action_ids_t] = float(self.cfg.ema_alpha_waist)

        if self.arm_action_ids_t.numel() > 0:
            alpha[:, self.arm_action_ids_t] = float(self.cfg.ema_alpha_arms)

        return alpha

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)

        if hasattr(self, "current_arm_gain"):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self.current_arm_gain[env_ids] = float(self._arm_action_gain())

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        # Safety guard before parent physics step.
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        arm_gain = float(self._arm_action_gain())

        if hasattr(self, "current_arm_gain"):
            self.current_arm_gain[:] = arm_gain

        action_mod = actions.clone()

        if hasattr(self, "arm_action_ids_t") and self.arm_action_ids_t.numel() > 0:
            action_mod[:, self.arm_action_ids_t] *= arm_gain

        if hasattr(self, "waist_action_ids_t") and self.waist_action_ids_t.numel() > 0:
            action_mod[:, self.waist_action_ids_t] *= max(0.25, arm_gain)

        return super().step(action_mod)


    # ------------------------------------------------------------------
    # Task3 observation
    # ------------------------------------------------------------------
    def _compute_obs(self) -> torch.Tensor:
        """Keep Task2 123-D observation layout.

        The arm action gain is not added to observation to keep:
            single obs dim = 123
            stacked obs dim = 615
            checkpoint compatibility with Task1 / Task2 architecture

        Whole-body information is injected through:
            - action gating
            - whole-body reward
            - reference manager
        """
        obs = super()._compute_obs()

        if obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"[G1WholeBodyEnv] Observation dim mismatch: got {obs.shape[-1]}, "
                f"expected {self.cfg.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    # ------------------------------------------------------------------
    # Task3 reward
    # ------------------------------------------------------------------
    def _arm_leg_sync_reward(
        self,
        q: torch.Tensor,
        ref_arm: torch.Tensor,
        arm_swing_ref: torch.Tensor,
        contact: torch.Tensor,
        moving: torch.Tensor,
        style_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute simple whole-body arm swing consistency.

        This is intentionally conservative. It is not imitation learning.
        It just encourages a reasonable opposite arm-leg rhythm when moving.

        Returns:
            r_arm_leg_sync
            r_arm_cross
        """
        if self.arm_action_ids_t.numel() == 0:
            z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            return z, z

        # q is controlled-joint tensor [N, 23].
        q_arm = q[:, self.arm_action_ids_t]

        # Shape fallback if generated arm_swing_ref dim differs.
        if arm_swing_ref.shape[-1] == q_arm.shape[-1]:
            ref = arm_swing_ref
        elif ref_arm.shape[-1] == q_arm.shape[-1]:
            ref = ref_arm
        else:
            min_dim = min(q_arm.shape[-1], arm_swing_ref.shape[-1])
            pad = torch.zeros_like(q_arm)
            pad[:, :min_dim] = arm_swing_ref[:, :min_dim]
            ref = pad

        r_arm_leg_sync = torch.exp(
            -float(self.cfg.sigma_arm) * torch.mean(torch.square(q_arm - ref), dim=-1)
        )

        # Cross rhythm proxy:
        # left arm should tend to move with right leg phase/contact,
        # right arm should tend to move with left leg phase/contact.
        #
        # We avoid relying on exact shoulder index names here; instead use the
        # first half / second half of arm action vector as left/right groups.
        half = max(1, q_arm.shape[-1] // 2)
        left_arm = q_arm[:, :half].mean(dim=-1)
        right_arm = q_arm[:, half:].mean(dim=-1) if q_arm.shape[-1] > half else q_arm[:, :half].mean(dim=-1)

        left_contact = contact[:, 0]
        right_contact = contact[:, 1]

        # Desired sign only acts as a weak phase-shaping hint.
        desired_left = 2.0 * right_contact - 1.0
        desired_right = 2.0 * left_contact - 1.0

        cross_error = torch.square(torch.tanh(left_arm) - 0.25 * desired_left) + torch.square(
            torch.tanh(right_arm) - 0.25 * desired_right
        )
        r_arm_cross = torch.exp(-2.0 * cross_error)

        moving_style = moving * float(style_scale)

        return r_arm_leg_sync * moving_style, r_arm_cross * moving_style

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

        root_pos = self.robot.data.root_pos_w
        base_height = root_pos[:, 2] - self.env_origins[:, 2]

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        self.base_acc_obs.copy_(base_acc)
        self.last_base_vel.copy_(base_lin_vel)

        contact, normal_force = self._get_feet_contact()
        contact_count = contact.sum(dim=-1)

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_z = foot_pos[:, :, 2] - self.env_origins[:, 2].unsqueeze(-1)
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]

        q = self.robot.data.joint_pos[:, self.controlled_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.controlled_joint_ids_t]
        q_err = q - self.default_ctrl_pos

        cmd = self.smoothed_cmd
        cmd_vx = cmd[:, 0]
        cmd_vy = cmd[:, 1]
        cmd_wz = cmd[:, 2]

        cmd_xy_norm = torch.linalg.norm(cmd[:, :2], dim=-1)
        cmd_yaw_abs = torch.abs(cmd[:, 2])

        moving = ((cmd_xy_norm > 0.035) | (cmd_yaw_abs > 0.08)).float()
        standing = 1.0 - moving

        p_yaw_drift = -standing * torch.abs(wz)

        lin_error = torch.square(vx - cmd_vx) + torch.square(vy - cmd_vy)
        yaw_error = torch.square(wz - cmd_wz)

        r_cmd_lin = torch.exp(-float(self.cfg.sigma_cmd_lin) * lin_error)
        r_cmd_yaw = torch.exp(-float(self.cfg.sigma_cmd_yaw) * yaw_error)

        actual_speed = torch.sqrt(torch.square(vx) + torch.square(vy) + 1e-6)
        target_speed = torch.sqrt(torch.square(cmd_vx) + torch.square(cmd_vy) + 1e-6)

        along_cmd = (
            vx * cmd_vx + vy * cmd_vy
        ) / torch.clamp(target_speed, min=0.05)

        r_cmd_speed = torch.exp(-8.0 * torch.square(actual_speed - target_speed)) * moving
        p_under_speed = -torch.relu(target_speed - actual_speed) * moving

        r_zero_vel = torch.exp(
            -float(self.cfg.sigma_zero)
            * (
                torch.square(vx)
                + torch.square(vy)
                + 0.50 * torch.square(wz)
            )
        )

        double_contact_penalty = -moving * torch.clamp(contact_count - 1.20, min=0.0)

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)

        self.feet_air_time += self.dt

        r_air_time = torch.sum(
            torch.clamp(self.feet_air_time - 0.10, min=0.0, max=0.45) * first_contact.float(),
            dim=-1,
        )
        r_air_time = r_air_time * moving

        self.feet_air_time = torch.where(
            contact > 0.5,
            torch.zeros_like(self.feet_air_time),
            self.feet_air_time,
        )
        self.prev_foot_contact.copy_(contact)

        r_clearance = (
            (1.0 - contact)
            * torch.exp(-20.0 * torch.abs(foot_z - float(self.cfg.foot_clearance_target)))
        ).sum(dim=-1)
        r_clearance = r_clearance * moving

        ref_pos, ref_vel, ref_contact, ref_arm, ref_arm_vel, arm_swing_ref = self.motion.get_reference_by_cmd_phase(
            cmd=cmd,
            phase=self.phase,
        )

        style_scale = float(self._style_weight_scale())
        arm_gain = float(self._arm_action_gain())

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

        p_base_acc = -torch.clamp(torch.sum(torch.square(base_acc), dim=-1), max=30.0)

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

        if self.arm_action_ids_t.numel() > 0:
            q_arm = q[:, self.arm_action_ids_t]
            qd_arm = qd[:, self.arm_action_ids_t]

            if ref_arm.shape[-1] == q_arm.shape[-1]:
                r_arm_ref = torch.exp(
                    -float(self.cfg.sigma_arm) * torch.mean(torch.square(q_arm - ref_arm), dim=-1)
                )
            else:
                min_dim = min(ref_arm.shape[-1], q_arm.shape[-1])
                r_arm_ref = torch.exp(
                    -float(self.cfg.sigma_arm)
                    * torch.mean(torch.square(q_arm[:, :min_dim] - ref_arm[:, :min_dim]), dim=-1)
                )

            if ref_arm_vel.shape[-1] == qd_arm.shape[-1]:
                r_arm_vel_ref = torch.exp(
                    -float(self.cfg.sigma_arm_vel) * torch.mean(torch.square(qd_arm - ref_arm_vel), dim=-1)
                )
            else:
                min_dim = min(ref_arm_vel.shape[-1], qd_arm.shape[-1])
                r_arm_vel_ref = torch.exp(
                    -float(self.cfg.sigma_arm_vel)
                    * torch.mean(torch.square(qd_arm[:, :min_dim] - ref_arm_vel[:, :min_dim]), dim=-1)
                )

            r_arm_ref = r_arm_ref * float(style_scale) * moving
            r_arm_vel_ref = r_arm_vel_ref * float(style_scale) * moving
        else:
            r_arm_ref = torch.zeros_like(vx)
            r_arm_vel_ref = torch.zeros_like(vx)

        r_arm_leg_sync, r_arm_cross = self._arm_leg_sync_reward(
            q=q,
            ref_arm=ref_arm,
            arm_swing_ref=arm_swing_ref,
            contact=contact,
            moving=moving,
            style_scale=style_scale,
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

        joint_vel_abs_max = torch.abs(self.robot.data.joint_vel).max(dim=-1)[0]
        roll_pitch_mag = torch.linalg.norm(projected_gravity[:, :2], dim=-1)

        is_fallen = (
            (base_height < float(self.cfg.fall_height))
            | (base_height > float(self.cfg.jump_height))
            | (roll_pitch_mag > float(self.cfg.bad_orientation_xy))
            | (~torch.isfinite(base_height))
            | (~torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
            | (joint_vel_abs_max > float(self.cfg.max_joint_vel_abs))
        )

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        continuous_raw = (
            float(self.cfg.w_cmd_lin) * (moving * r_cmd_lin + standing * r_zero_vel)
            + float(self.cfg.w_cmd_speed) * r_cmd_speed
            + float(self.cfg.w_cmd_yaw) * r_cmd_yaw
            + float(self.cfg.w_yaw_drift) * p_yaw_drift
            + float(self.cfg.w_zero_vel) * r_zero_vel * standing
            + float(self.cfg.w_under_speed) * p_under_speed
            + float(self.cfg.w_double_contact) * double_contact_penalty
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_base_ang_vel) * p_base_ang
            + float(self.cfg.w_base_acc) * p_base_acc
            + float(self.cfg.w_com_support) * r_com_support
            + float(self.cfg.w_z_vel) * p_z_vel
            + float(self.cfg.w_ref_pose) * r_ref_pose
            + float(self.cfg.w_ref_vel) * r_ref_vel
            + float(self.cfg.w_arm_ref) * r_arm_ref
            + float(self.cfg.w_arm_vel_ref) * r_arm_vel_ref
            + float(self.cfg.w_arm_leg_sync) * r_arm_leg_sync
            + float(self.cfg.w_arm_cross) * r_arm_cross
            + float(self.cfg.w_default_pose) * p_default_pose
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
        command_stage = int(self._command_stage())

        info = {
            "reward_components": {
                "R_Cmd_Lin": self._mean_detached(float(self.cfg.w_cmd_lin) * (moving * r_cmd_lin + standing * r_zero_vel)),
                "R_Cmd_Speed": self._mean_detached(float(self.cfg.w_cmd_speed) * r_cmd_speed),
                "R_Cmd_Yaw": self._mean_detached(float(self.cfg.w_cmd_yaw) * r_cmd_yaw),
                "P_Yaw_Drift": self._mean_detached(float(self.cfg.w_yaw_drift) * p_yaw_drift),
                "R_Zero_Vel": self._mean_detached(float(self.cfg.w_zero_vel) * r_zero_vel * standing),
                "P_Under_Speed": self._mean_detached(float(self.cfg.w_under_speed) * p_under_speed),
                "P_Double_Contact": self._mean_detached(float(self.cfg.w_double_contact) * double_contact_penalty),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Upright": self._mean_detached(float(self.cfg.w_upright) * r_upright),
                "R_Height": self._mean_detached(float(self.cfg.w_height) * r_height),
                "P_Base_Ang": self._mean_detached(float(self.cfg.w_base_ang_vel) * p_base_ang),
                "P_Base_Acc": self._mean_detached(float(self.cfg.w_base_acc) * p_base_acc),
                "R_COM_Support": self._mean_detached(float(self.cfg.w_com_support) * r_com_support),
                "P_Z_Vel": self._mean_detached(float(self.cfg.w_z_vel) * p_z_vel),
                "R_Ref_Pose": self._mean_detached(float(self.cfg.w_ref_pose) * r_ref_pose),
                "R_Ref_Vel": self._mean_detached(float(self.cfg.w_ref_vel) * r_ref_vel),
                "R_Arm_Ref": self._mean_detached(float(self.cfg.w_arm_ref) * r_arm_ref),
                "R_Arm_Vel_Ref": self._mean_detached(float(self.cfg.w_arm_vel_ref) * r_arm_vel_ref),
                "R_Arm_Leg_Sync": self._mean_detached(float(self.cfg.w_arm_leg_sync) * r_arm_leg_sync),
                "R_Arm_Cross": self._mean_detached(float(self.cfg.w_arm_cross) * r_arm_cross),
                "P_Default_Pose": self._mean_detached(float(self.cfg.w_default_pose) * p_default_pose),
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
                "Command_Stage": self._float_tensor(float(command_stage)),
                "Cmd_Vx": self._mean_detached(cmd_vx),
                "Cmd_Vy": self._mean_detached(cmd_vy),
                "Cmd_Wz": self._mean_detached(cmd_wz),
                "Actual_Vx": self._mean_detached(vx),
                "Actual_Vy": self._mean_detached(vy),
                "Actual_Wz": self._mean_detached(wz),
                "Actual_Along_Cmd": self._mean_detached(along_cmd),
                "Lin_Error": self._mean_detached(torch.sqrt(lin_error + 1e-6)),
                "Yaw_Error": self._mean_detached(torch.sqrt(yaw_error + 1e-6)),
                "Base_Height": self._mean_detached(base_height),
                "RollPitch_Mag": self._mean_detached(roll_pitch_mag),
                "Harness_Ratio": self._float_tensor(float(self._harness_ratio())),
                "Arm_Action_Gain": self._float_tensor(float(arm_gain)),
                "Style_Scale": self._float_tensor(float(style_scale)),
                "RSI_Prob": self._float_tensor(float(self._rsi_probability())),
                "Contact_Count": self._mean_detached(contact_count),
                "Left_Contact": self._mean_detached(contact[:, 0]),
                "Right_Contact": self._mean_detached(contact[:, 1]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Foot_Slip_Raw": self._mean_detached(-p_slip),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Global_Steps": self._float_tensor(float(self.global_steps)),
                "Arm_Ref_Raw": self._mean_detached(r_arm_ref),
                "Arm_Leg_Sync_Raw": self._mean_detached(r_arm_leg_sync),
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
                "Motion_Num_Modes": self._float_tensor(float(self.motion.num_modes)),
                "Motion_Arm_Ref_Dim": self._float_tensor(float(self.motion.arm_swing_ref.shape[-1])),
            },
        }

        return reward, terminated, truncated, info


# Backward-compatible aliases
Task3ConfigAlias = Task3Config
Task3Env = G1WholeBodyEnv
G1Task3Env = G1WholeBodyEnv
