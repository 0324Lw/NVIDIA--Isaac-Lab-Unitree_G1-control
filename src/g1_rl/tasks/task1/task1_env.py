# Copyright (c) 2026
# Unitree G1 Task1: assisted locomotion pure-RL baseline environment.
#
# Project positioning:
#   This is an educational pure-RL baseline for humanoid control.
#   It is kept as a learning / comparison project, not as the final professional
#   route for complex humanoid skills. High-quality humanoid motion usually
#   requires imitation learning, retargeted motion data, motion priors, and
#   sim-to-real engineering.
#
# Strict refactor notes:
#   1. This file defines IsaacLab environment only.
#   2. It does not start AppLauncher.
#   3. AppLauncher must be launched before importing this file in test/train scripts.
#   4. Policy controls 23 joints.
#   5. Sensor joints xl330_joint and d455_joint are fixed.
#   6. Single-frame observation dim = 123.
#   7. Training code will later apply 5-frame stacking: 123 * 5 = 615.

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*set_external_force_and_torque.*")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

try:
    import warp as wp
except Exception:
    wp = None

from g1_rl.tasks.task1.task1_config import Task1Config


def make_g1_task1_scene_cfg(cfg: Task1Config):
    """Create IsaacLab scene config for G1 Task1."""

    @configclass
    class G1Task1SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(cfg.usd_path),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(cfg.target_height)),
                joint_pos={".*": 0.0},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                    stiffness=150.0,
                    damping=5.0,
                ),
                "upper_body": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "waist_.*",
                        ".*_shoulder_.*",
                        ".*_elbow_.*",
                        ".*_wrist_.*",
                    ],
                    stiffness=40.0,
                    damping=2.0,
                ),
                "sensors": ImplicitActuatorCfg(
                    joint_names_expr=["xl330_joint", "d455_joint"],
                    stiffness=10_000.0,
                    damping=1_000.0,
                ),
            },
        )

        # Capture ankle pitch / ankle roll links, then select true foot links by name.
        contact_forces: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_.*",
            update_period=0.0,
            history_length=3,
            track_air_time=False,
            debug_vis=False,
        )

        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

    return G1Task1SceneCfg(num_envs=int(cfg.num_envs), env_spacing=float(cfg.env_spacing))


class G1MotionManager:
    """Reference motion loader for G1 Task1.

    Expected motion file keys:
        pos:         [T, num_robot_joints]
        vel:         [T, num_robot_joints]
        num_frames:  int
        joint_names: optional list[str]
        fps:         optional float
        dt:          optional float
        phase:       optional [T]
        contact_ref: optional [T, 2]
    """

    def __init__(
        self,
        motion_file: str,
        robot_joint_names: List[str],
        controlled_joint_ids: torch.Tensor,
        device: str,
        strict_check: bool = True,
        verbose: bool = True,
    ):
        self.motion_file = str(motion_file)
        self.device = str(device)
        self.robot_joint_names = list(robot_joint_names)
        self.controlled_joint_ids = controlled_joint_ids.to(device=self.device, dtype=torch.long)

        if not os.path.exists(self.motion_file):
            raise FileNotFoundError(
                f"[G1MotionManager] Cannot find motion file: {self.motion_file}\n"
                "Please set environment variable G1_TASK1_MOTION_FILE or put g1_walk.pt at the default path."
            )

        data = torch.load(self.motion_file, map_location=self.device)

        if "pos" not in data or "vel" not in data:
            raise RuntimeError(
                f"[G1MotionManager] motion file must contain keys 'pos' and 'vel'. "
                f"Available keys: {list(data.keys())}"
            )

        self.pos_full = data["pos"].to(device=self.device, dtype=torch.float32)
        self.vel_full = data["vel"].to(device=self.device, dtype=torch.float32)

        self.num_frames = int(data.get("num_frames", self.pos_full.shape[0]))
        self.joint_names = list(data.get("joint_names", []))
        self.fps = float(data.get("fps", 50.0))
        self.dt = float(data.get("dt", 1.0 / max(self.fps, 1e-6)))

        self.phase = data.get("phase", torch.linspace(0.0, 1.0, self.num_frames))
        self.phase = self.phase.to(device=self.device, dtype=torch.float32)

        self.contact_ref = data.get("contact_ref", torch.ones((self.num_frames, 2)))
        self.contact_ref = self.contact_ref.to(device=self.device, dtype=torch.float32)

        if self.pos_full.ndim != 2:
            raise RuntimeError(f"[G1MotionManager] pos must be [T, J], got {tuple(self.pos_full.shape)}")

        if self.vel_full.shape != self.pos_full.shape:
            raise RuntimeError(
                f"[G1MotionManager] vel shape mismatch: vel={tuple(self.vel_full.shape)}, "
                f"pos={tuple(self.pos_full.shape)}"
            )

        if self.pos_full.shape[0] != self.num_frames:
            raise RuntimeError(
                f"[G1MotionManager] num_frames mismatch: data={self.num_frames}, "
                f"pos.shape[0]={self.pos_full.shape[0]}"
            )

        if self.pos_full.shape[1] != len(self.robot_joint_names):
            raise RuntimeError(
                f"[G1MotionManager] motion joint dim mismatch: "
                f"motion={self.pos_full.shape[1]}, robot={len(self.robot_joint_names)}"
            )

        if strict_check:
            if len(self.joint_names) != len(self.robot_joint_names):
                raise RuntimeError(
                    f"[G1MotionManager] motion['joint_names'] length mismatch. "
                    f"Expected {len(self.robot_joint_names)}, got {len(self.joint_names)}"
                )

            if self.joint_names != self.robot_joint_names:
                mismatch = [
                    (i, a, b)
                    for i, (a, b) in enumerate(zip(self.joint_names, self.robot_joint_names))
                    if a != b
                ]
                raise RuntimeError(
                    "[G1MotionManager] motion joint_names do not match robot.joint_names. "
                    f"First mismatches: {mismatch[:8]}"
                )

        self.pos_ctrl = self.pos_full[:, self.controlled_joint_ids]
        self.vel_ctrl = self.vel_full[:, self.controlled_joint_ids]

        if self.contact_ref.ndim != 2 or self.contact_ref.shape[-1] != 2:
            raise RuntimeError(
                f"[G1MotionManager] contact_ref should be [T, 2], got {tuple(self.contact_ref.shape)}"
            )

        if self.contact_ref.shape[0] != self.num_frames:
            if self.contact_ref.shape[0] > self.num_frames:
                self.contact_ref = self.contact_ref[: self.num_frames]
            else:
                pad = self.contact_ref[-1:].repeat(self.num_frames - self.contact_ref.shape[0], 1)
                self.contact_ref = torch.cat([self.contact_ref, pad], dim=0)

        self.contact_ref = torch.clamp(self.contact_ref, 0.0, 1.0)

        if verbose:
            print("\n" + "=" * 88)
            print(" [G1MotionManager] Reference motion loaded")
            print(f" file             : {self.motion_file}")
            print(f" motion type      : {data.get('motion_type', 'unknown')}")
            print(f" num_frames       : {self.num_frames}")
            print(f" fps              : {self.fps}")
            print(f" full joint dim   : {self.pos_full.shape[1]}")
            print(f" controlled dim   : {self.pos_ctrl.shape[1]}")
            print(f" contact_ref shape: {tuple(self.contact_ref.shape)}")
            print("=" * 88 + "\n")

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


class G1Task1Env(gym.Env):
    """Unitree G1 assisted locomotion pure-RL environment.

    Observation dim = 123:
        base_lin_vel             3
        base_ang_vel             3
        projected_gravity        3
        command                  3
        controlled q error      23
        controlled qd           23
        last action             23
        action delta            23
        foot contact             2
        foot relative position   6
        foot xy velocity         4
        base acceleration        3
        sin phase                1
        cos phase                1
        harness ratio            1
        root height              1
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task1Config):
        super().__init__()

        cfg.validate()

        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.control_dt)

        if not os.path.exists(str(cfg.usd_path)):
            raise FileNotFoundError(
                f"[G1Task1Env] Cannot find G1 USD: {cfg.usd_path}\n"
                "Please set environment variable G1_USD_PATH or update Task1Config.usd_path."
            )

        sim_cfg = sim_utils.SimulationCfg(
            dt=float(cfg.sim_dt),
            device=str(cfg.device),
            physx=sim_utils.PhysxCfg(
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
            ),
        )

        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = InteractiveScene(make_g1_task1_scene_cfg(cfg))
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact: ContactSensor = self.scene["contact_forces"]

        self.env_origins = self._get_env_origins()

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)

        self.default_joint_pos = self.robot.data.default_joint_pos.detach().clone()
        self.default_joint_vel = torch.zeros_like(self.default_joint_pos)

        self.controlled_joint_ids = self._joint_ids(cfg.controlled_joint_names)
        self.leg_joint_ids = self._joint_ids(cfg.leg_joint_names)
        self.waist_joint_ids = self._joint_ids(cfg.waist_joint_names)
        self.arm_joint_ids = self._joint_ids(cfg.arm_joint_names)
        self.sensor_joint_ids = self._joint_ids(cfg.sensor_joint_names)

        self.controlled_joint_ids_t = torch.as_tensor(self.controlled_joint_ids, dtype=torch.long, device=self.device)
        self.leg_joint_ids_t = torch.as_tensor(self.leg_joint_ids, dtype=torch.long, device=self.device)
        self.waist_joint_ids_t = torch.as_tensor(self.waist_joint_ids, dtype=torch.long, device=self.device)
        self.arm_joint_ids_t = torch.as_tensor(self.arm_joint_ids, dtype=torch.long, device=self.device)
        self.sensor_joint_ids_t = torch.as_tensor(self.sensor_joint_ids, dtype=torch.long, device=self.device)

        self.foot_body_ids = self._body_ids(cfg.foot_body_names)
        self.foot_body_ids_t = torch.as_tensor(self.foot_body_ids, dtype=torch.long, device=self.device)

        self.contact_foot_ids = self._contact_ids(cfg.foot_body_names)
        self.contact_foot_ids_t = torch.as_tensor(self.contact_foot_ids, dtype=torch.long, device=self.device)

        self.num_actions = int(len(self.controlled_joint_ids))
        self.cfg.num_actions = self.num_actions

        if self.num_actions != 23:
            raise RuntimeError(f"[G1Task1Env] Expected 23 controlled joints, got {self.num_actions}")

        self.action_scale = self._make_action_scale()
        self.default_ctrl_pos = self.default_joint_pos[:, self.controlled_joint_ids_t].detach().clone()

        lower, upper = self._get_joint_limits()
        self.joint_lower_all = lower
        self.joint_upper_all = upper
        self.ctrl_lower = lower[:, self.controlled_joint_ids_t]
        self.ctrl_upper = upper[:, self.controlled_joint_ids_t]

        try:
            self.robot_mass = float(self.robot.root_physx_view.get_masses()[0].sum().item())
        except Exception:
            self.robot_mass = 32.8

        self.motion = G1MotionManager(
            motion_file=str(cfg.motion_file),
            robot_joint_names=self.robot_joint_names,
            controlled_joint_ids=self.controlled_joint_ids_t,
            device=self.device,
            strict_check=bool(cfg.strict_motion_joint_check),
            verbose=bool(cfg.print_debug_info),
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_observations),),
            dtype=np.float32,
        )

        self.state_space = self.observation_space

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # Buffers
        n, a = self.num_envs, self.num_actions

        self.global_steps = 0
        self.episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.last_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.joint_position_targets = self.default_joint_pos.detach().clone()

        self.last_base_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_acc_obs = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.phase = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.current_target_vx = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.current_harness_ratio = torch.full(
            (n,),
            float(cfg.harness_start),
            dtype=torch.float32,
            device=self.device,
        )

        self.feet_air_time = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.prev_foot_contact = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_fall_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        self.reset()

        if bool(self.cfg.print_debug_info):
            self._print_debug_info()

    # ------------------------------------------------------------------
    # Name helpers / limits / debug
    # ------------------------------------------------------------------
    def _get_env_origins(self) -> torch.Tensor:
        if hasattr(self.scene, "env_origins"):
            return self.scene.env_origins.to(self.device)

        try:
            return self.scene._default_env_origins.to(self.device)
        except Exception:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _get_joint_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.robot.data, "soft_joint_pos_limits"):
            limits = self.robot.data.soft_joint_pos_limits
        elif hasattr(self.robot.data, "joint_pos_limits"):
            limits = self.robot.data.joint_pos_limits
        else:
            raise RuntimeError("[G1Task1Env] Cannot find joint position limits.")

        limits = limits.detach().clone().to(self.device)

        if limits.shape[0] == 1:
            limits = limits.repeat(self.num_envs, 1, 1)

        if limits.shape[0] != self.num_envs:
            limits = limits[:1].repeat(self.num_envs, 1, 1)

        lower = limits[:, :, 0]
        upper = limits[:, :, 1]

        return lower, upper

    def _joint_ids(self, names: Iterable[str]) -> List[int]:
        ids: List[int] = []
        missing: List[str] = []

        for target in list(names):
            if target in self.robot_joint_names:
                ids.append(self.robot_joint_names.index(target))
            else:
                matches = [i for i, name in enumerate(self.robot_joint_names) if target in name]
                if matches:
                    ids.append(matches[0])
                else:
                    missing.append(target)

        if missing:
            raise RuntimeError(
                f"[G1Task1Env] Missing joints: {missing}\n"
                f"Available joints: {self.robot_joint_names}"
            )

        return ids

    def _body_ids(self, names: Iterable[str]) -> List[int]:
        ids: List[int] = []
        missing: List[str] = []

        for target in list(names):
            if target in self.robot_body_names:
                ids.append(self.robot_body_names.index(target))
            else:
                matches = [i for i, name in enumerate(self.robot_body_names) if target in name]
                if matches:
                    ids.append(matches[0])
                else:
                    missing.append(target)

        if missing:
            raise RuntimeError(
                f"[G1Task1Env] Missing body links: {missing}\n"
                f"Available body names: {self.robot_body_names}"
            )

        return ids

    def _contact_ids(self, names: Iterable[str]) -> List[int]:
        contact_names = list(self.contact.body_names)

        ids: List[int] = []
        missing: List[str] = []

        for target in list(names):
            if target in contact_names:
                ids.append(contact_names.index(target))
            else:
                matches = [i for i, name in enumerate(contact_names) if target in name]
                if matches:
                    ids.append(matches[0])
                else:
                    missing.append(target)

        if missing:
            raise RuntimeError(
                f"[G1Task1Env] Missing contact links: {missing}\n"
                f"Available contact sensor body names: {contact_names}"
            )

        return ids

    def _print_debug_info(self) -> None:
        print("\n" + "=" * 100)
        print(" [G1Task1Env] Initialized")
        print("=" * 100)
        print(f" num_envs             : {self.cfg.num_envs}")
        print(f" device               : {self.device}")
        print(f" num_joints           : {self.robot.num_joints}")
        print(f" num_actions          : {self.num_actions}")
        print(f" num_observations     : {self.cfg.num_observations}")
        print(f" robot_mass           : {self.robot_mass:.3f} kg")
        print(f" usd_path             : {self.cfg.usd_path}")
        print(f" motion_file          : {self.cfg.motion_file}")
        print(f" controlled_joint_ids : {self.controlled_joint_ids}")
        print(f" sensor_joint_ids     : {self.sensor_joint_ids}")
        print(f" foot_body_ids        : {self.foot_body_ids}")
        print(f" contact_foot_ids     : {self.contact_foot_ids}")
        print("-" * 100)
        print(" controlled joints:")
        for i, jid in enumerate(self.controlled_joint_ids):
            print(f"   action {i:02d} -> joint {jid:02d}: {self.robot_joint_names[jid]}")
        print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # Curriculum / action scale / harness
    # ------------------------------------------------------------------
    @staticmethod
    def _smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def curriculum_k(self) -> float:
        return min(1.0, max(0.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0)))

    def _curriculum_values(self) -> Tuple[float, float, int]:
        """Return target_vx, harness_ratio, stage."""

        k = self.curriculum_k()

        if k < 0.15:
            s = self._smoothstep(k / 0.15)
            return 0.0, 0.80 - 0.15 * s, 0

        if k < 0.35:
            s = self._smoothstep((k - 0.15) / 0.20)
            return 0.10 * s, 0.65 - 0.20 * s, 1

        if k < 0.65:
            s = self._smoothstep((k - 0.35) / 0.30)
            return 0.10 + 0.25 * s, 0.45 - 0.30 * s, 2

        if k < 0.90:
            s = self._smoothstep((k - 0.65) / 0.25)
            return 0.35 + (float(self.cfg.target_vx_final) - 0.35) * s, 0.15 * (1.0 - s), 3

        return float(self.cfg.target_vx_final), 0.0, 4

    def _reference_reset_scale(self) -> float:
        k = self.curriculum_k()

        if k < 0.15:
            return 0.0

        if k < 0.65:
            return float(self.cfg.reference_reset_scale_max) * self._smoothstep((k - 0.15) / 0.50)

        return float(self.cfg.reference_reset_scale_max)

    def _style_weight_scale(self) -> float:
        k = self.curriculum_k()

        if k < 0.35:
            return 0.0

        return self._smoothstep((k - 0.35) / 0.45)

    def _make_action_scale(self) -> torch.Tensor:
        scale = torch.full((self.num_actions,), float(self.cfg.arm_action_scale), device=self.device)

        leg_set = set(self.cfg.leg_joint_names)
        waist_set = set(self.cfg.waist_joint_names)

        for i, jid in enumerate(self.controlled_joint_ids):
            name = self.robot_joint_names[jid]

            if name in leg_set:
                scale[i] = float(self.cfg.leg_action_scale)
            elif name in waist_set:
                scale[i] = float(self.cfg.waist_action_scale)
            elif "wrist" in name:
                scale[i] = float(self.cfg.wrist_action_scale)
            else:
                scale[i] = float(self.cfg.arm_action_scale)

        return scale

    def _apply_harness_force(self, harness_ratio: float) -> None:
        ratio = float(harness_ratio)
        n = self.num_envs

        forces = torch.zeros((n, 1, 3), dtype=torch.float32, device=self.device)
        torques = torch.zeros_like(forces)

        forces[:, 0, 2] = ratio * float(self.robot_mass) * 9.81

        # Prefer permanent_wrench_composer if available. Fall back to older API.
        if wp is not None and hasattr(self.robot, "permanent_wrench_composer"):
            try:
                env_ids_wp = wp.from_torch(
                    torch.arange(n, dtype=torch.int32, device=self.device),
                    dtype=wp.int32,
                )
                body_ids_wp = wp.array(
                    [int(self.cfg.harness_body_id)],
                    dtype=wp.int32,
                    device=self.device,
                )
                self.robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=wp.from_torch(forces, dtype=wp.vec3f),
                    torques=wp.from_torch(torques, dtype=wp.vec3f),
                    body_ids=body_ids_wp,
                    env_ids=env_ids_wp,
                    is_global=True,
                )
                return
            except Exception:
                pass

        try:
            self.robot.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                body_ids=[int(self.cfg.harness_body_id)],
                is_global=True,
            )
        except TypeError:
            try:
                self.robot.set_external_force_and_torque(
                    forces,
                    torques,
                    body_ids=[int(self.cfg.harness_body_id)],
                )
            except Exception:
                pass
        except Exception:
            pass

    def _lock_sensor_joints(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Hard-lock sensor joints to default position.

        G1 USD contains xl330_joint and d455_joint as physical joints.
        They are not policy-controlled joints, but physics can still move them
        slightly through articulation coupling. For this pure-RL baseline we keep
        them fixed so the action space remains exactly 23 DoF.
        """
        if len(self.sensor_joint_ids) == 0:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return

        q = self.robot.data.joint_pos[env_ids].detach().clone()
        qd = self.robot.data.joint_vel[env_ids].detach().clone()

        q[:, self.sensor_joint_ids_t] = self.default_joint_pos[env_ids][:, self.sensor_joint_ids_t]
        qd[:, self.sensor_joint_ids_t] = 0.0

        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        self._reset_idx(env_ids)

        return self._compute_obs(), {}

    @torch.no_grad()
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return

        num_reset = int(env_ids.numel())

        ref_scale = self._reference_reset_scale()
        ref_pos, ref_vel = self.motion.sample_initial_state(env_ids, ref_scale=ref_scale)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = self.env_origins[env_ids, 0:2]
        root_state[:, 2] = float(self.cfg.target_height)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        root_state[:, 7:13] = 0.0

        q0 = self.default_joint_pos[env_ids].clone()
        qd0 = torch.zeros_like(q0)

        # Reference motion is stored in full 25-joint robot order.
        q0 = q0 + ref_pos
        qd0 = qd0 + ref_vel

        # Sensor joints are always fixed.
        if len(self.sensor_joint_ids) > 0:
            q0[:, self.sensor_joint_ids_t] = self.default_joint_pos[env_ids][:, self.sensor_joint_ids_t]
            qd0[:, self.sensor_joint_ids_t] = 0.0

        lower = self.joint_lower_all[env_ids]
        upper = self.joint_upper_all[env_ids]
        q0 = torch.clamp(q0, lower, upper)

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)

        self.robot.reset(env_ids)
        try:
            self.contact.reset(env_ids)
        except Exception:
            pass

        self.scene.update(dt=0.0)
        self._lock_sensor_joints(env_ids)
        self.scene.update(dt=0.0)

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0

        self.last_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.joint_position_targets[env_ids] = self.default_joint_pos[env_ids]

        self.last_base_vel[env_ids] = self.robot.data.root_lin_vel_b[env_ids]
        self.base_acc_obs[env_ids] = 0.0

        self.phase[env_ids] = torch.rand(num_reset, dtype=torch.float32, device=self.device)

        self.feet_air_time[env_ids] = 0.0
        self.prev_foot_contact[env_ids] = 0.0

        target_vx, harness_ratio, _ = self._curriculum_values()
        self.current_target_vx[env_ids] = float(target_vx)
        self.current_harness_ratio[env_ids] = float(harness_ratio)

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        actions = actions.to(device=self.device, dtype=torch.float32)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)

        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise RuntimeError(
                f"[G1Task1Env] action shape mismatch: got {tuple(actions.shape)}, "
                f"expected {(self.num_envs, self.num_actions)}"
            )

        # Safety guard:
        # Model-test checkpoints, broken normalizers, or partially trained policies may
        # occasionally produce NaN/Inf actions. torch.clamp does not remove NaN.
        # Passing NaN joint targets into PhysX can make the first simulation step hang.
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        self.prev_action.copy_(self.last_action)

        filtered_action = (
            float(self.cfg.ema_alpha) * actions
            + (1.0 - float(self.cfg.ema_alpha)) * self.last_action
        )
        self.last_action.copy_(torch.clamp(filtered_action, -1.0, 1.0))

        target_vx, harness_ratio, _ = self._curriculum_values()
        self.current_target_vx[:] = float(target_vx)
        self.current_harness_ratio[:] = float(harness_ratio)

        target_ctrl = self.default_ctrl_pos + self.last_action * self.action_scale
        target_ctrl = torch.nan_to_num(target_ctrl, nan=0.0, posinf=0.0, neginf=0.0)
        target_ctrl = torch.clamp(target_ctrl, self.ctrl_lower, self.ctrl_upper)

        full_target = self.default_joint_pos.clone()
        full_target[:, self.controlled_joint_ids_t] = target_ctrl

        if len(self.sensor_joint_ids) > 0:
            full_target[:, self.sensor_joint_ids_t] = self.default_joint_pos[:, self.sensor_joint_ids_t]

        self.joint_position_targets.copy_(full_target)

        self._apply_harness_force(harness_ratio)

        self.robot.set_joint_position_target(self.joint_position_targets)
        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self._lock_sensor_joints()
        self.scene.update(dt=0.0)

        self.global_steps += self.num_envs
        self.episode_steps += 1

        self.phase = torch.remainder(
            self.phase + self.dt * float(self.cfg.gait_freq_hz),
            1.0,
        )

        rewards, terminated, truncated, info = self._compute_rewards()
        self.episode_return += rewards

        obs_before_reset = self._compute_obs()

        done = terminated | truncated
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)

        obs = obs_before_reset
        if reset_ids.numel() > 0:
            info["terminal_observation"] = obs_before_reset[reset_ids].clone()
            self._reset_idx(reset_ids)
            reset_obs = self._compute_obs()
            obs[reset_ids] = reset_obs[reset_ids]

        return obs, rewards, terminated, truncated, info

    def close(self) -> None:
        try:
            forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
            torques = torch.zeros_like(forces)
            self.robot.set_external_force_and_torque(
                forces,
                torques,
                body_ids=[int(self.cfg.harness_body_id)],
            )
        except Exception:
            pass

        try:
            self.sim.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_feet_contact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.contact.data

        if hasattr(data, "net_forces_w_history") and data.net_forces_w_history is not None:
            forces = data.net_forces_w_history[:, :, self.contact_foot_ids_t, :]
            normal_force = torch.max(forces[..., 2], dim=1)[0]
        else:
            forces = data.net_forces_w[:, self.contact_foot_ids_t, :]
            normal_force = forces[..., 2]

        contact = (normal_force > float(self.cfg.contact_force_threshold)).float()
        return contact, normal_force

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

        command = torch.stack(
            [
                self.current_target_vx,
                torch.full_like(self.current_target_vx, float(self.cfg.target_vy)),
                torch.full_like(self.current_target_vx, float(self.cfg.target_yaw_rate)),
            ],
            dim=-1,
        )

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
                f"[G1Task1Env] Observation dim mismatch: "
                f"got {obs.shape[-1]}, expected {self.cfg.num_observations}"
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
    @staticmethod
    def _mean_detached(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().mean()

    def _float_tensor(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

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

        # ----------------------------- Locomotion / gait -----------------------------
        target_vx = self.current_target_vx

        move_gate = (torch.abs(target_vx) > 0.03).float()
        stand_gate = (target_vx < 0.05).float()
        support = (contact_count > 0.5).float()

        double_contact_penalty = -move_gate * torch.clamp(contact_count - 1.20, min=0.0)

        r_stand_still = torch.exp(
            -10.0
            * (
                torch.square(vx)
                + torch.square(vy)
                + 0.5 * torch.square(vz)
            )
        )

        r_walk_vx = torch.exp(-float(self.cfg.sigma_v) * torch.square(vx - target_vx))
        r_vx = stand_gate * r_stand_still + move_gate * r_walk_vx

        r_yaw = torch.exp(-float(self.cfg.sigma_yaw) * torch.square(wz - float(self.cfg.target_yaw_rate)))

        p_cmd_lat = -(
            torch.abs(vy - float(self.cfg.target_vy))
            + 0.50 * torch.abs(vz)
            + stand_gate * 0.50 * torch.abs(vx)
        )

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += self.dt

        r_air_time = torch.sum(
            torch.clamp(self.feet_air_time - 0.10, min=0.0, max=0.45) * first_contact.float(),
            dim=-1,
        )
        r_air_time = r_air_time * support * move_gate

        self.feet_air_time = torch.where(contact > 0.5, torch.zeros_like(self.feet_air_time), self.feet_air_time)
        self.prev_foot_contact.copy_(contact)

        r_clearance = (
            (1.0 - contact)
            * torch.exp(-20.0 * torch.abs(foot_z - float(self.cfg.foot_clearance_target)))
        ).sum(dim=-1)
        r_clearance = r_clearance * move_gate

        r_phase_contact = 1.0 - torch.mean(torch.abs(contact - ref_contact), dim=-1)
        r_phase_contact = r_phase_contact * move_gate

        # ----------------------------- Stability -----------------------------
        r_upright = (1.0 - projected_gravity[:, 2]) * 0.5

        h_err = torch.clamp(
            torch.abs(base_height - float(self.cfg.target_height)) - float(self.cfg.deadband_height),
            min=0.0,
        )
        r_height = torch.exp(-float(self.cfg.sigma_z) * torch.square(h_err))

        p_base_ang = -(torch.square(wx) + torch.square(wy))

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        p_base_acc = -torch.clamp(torch.sum(torch.square(base_acc), dim=-1), max=30.0)
        self.last_base_vel.copy_(base_lin_vel)

        r_com_support = torch.exp(-1.5 * torch.abs(vy)) * support

        # ----------------------------- Safety / efficiency / style -----------------------------
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

        p_slip = -torch.sum(
            torch.sum(torch.square(foot_vel_xy), dim=-1) * contact,
            dim=-1,
        )

        tau_full = getattr(self.robot.data, "applied_torque", torch.zeros_like(self.robot.data.joint_vel))
        tau = tau_full[:, self.controlled_joint_ids_t]

        p_energy = -torch.mean(torch.abs(tau * qd), dim=-1)

        continuous_raw = (
            float(self.cfg.w_vx) * r_vx
            + float(self.cfg.w_double_contact) * double_contact_penalty
            + float(self.cfg.w_yaw) * r_yaw
            + float(self.cfg.w_cmd_lat) * p_cmd_lat
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_base_ang_vel) * p_base_ang
            + float(self.cfg.w_base_acc) * p_base_acc
            + float(self.cfg.w_com_support) * r_com_support
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

        _, harness_ratio, stage = self._curriculum_values()
        total_done_safe = torch.clamp(self.total_done_episodes, min=1.0)

        info = {
            "reward_components": {
                "R_Vx": self._mean_detached(float(self.cfg.w_vx) * r_vx),
                "P_Double_Contact": self._mean_detached(float(self.cfg.w_double_contact) * double_contact_penalty),
                "R_Yaw": self._mean_detached(float(self.cfg.w_yaw) * r_yaw),
                "P_Cmd_Lat": self._mean_detached(float(self.cfg.w_cmd_lat) * p_cmd_lat),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Upright": self._mean_detached(float(self.cfg.w_upright) * r_upright),
                "R_Height": self._mean_detached(float(self.cfg.w_height) * r_height),
                "P_Base_Ang": self._mean_detached(float(self.cfg.w_base_ang_vel) * p_base_ang),
                "P_Base_Acc": self._mean_detached(float(self.cfg.w_base_acc) * p_base_acc),
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
                "Event": self._mean_detached(event_fall),
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
                "Curriculum_Stage": self._float_tensor(float(stage)),
                "Target_Vx": self._mean_detached(self.current_target_vx),
                "Actual_Vx": self._mean_detached(vx),
                "Actual_Vy": self._mean_detached(vy),
                "Actual_Wz": self._mean_detached(wz),
                "Base_Height": self._mean_detached(base_height),
                "RollPitch_Mag": self._mean_detached(roll_pitch_mag),
                "Harness_Ratio": self._float_tensor(float(harness_ratio)),
                "Contact_Count": self._mean_detached(contact_count),
                "Left_Contact": self._mean_detached(contact[:, 0]),
                "Right_Contact": self._mean_detached(contact[:, 1]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Style_Scale": self._float_tensor(float(style_scale)),
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


# Backward-compatible aliases for old scripts.
G1HarnessEnv = G1Task1Env
G1AssistedLocomotionEnv = G1Task1Env
UnitreeG1Task1Env = G1Task1Env
