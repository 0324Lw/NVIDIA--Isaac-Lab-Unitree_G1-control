from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from g1_rl.tasks.task4.task4_config import Task4Config


def _make_scene_cfg(cfg: Task4Config):
    @configclass
    class G1Task4SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = 2.0

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

        contact_forces: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_.*",
            update_period=0.0,
            history_length=3,
            track_air_time=False,
            debug_vis=False,
        )

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

    return G1Task4SceneCfg


class G1Sim2RealEnv(gym.Env):
    """Unitree G1 Task4: standalone Sim2Real robustness environment.

    This task is an educational pure-RL robustness baseline. It does not use
    HoloSoma, OmniRetarget, BeyondMimic, AMP, or motion imitation.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task4Config):
        super().__init__()

        cfg.validate()
        self.cfg = cfg
        self.device = str(cfg.device)
        self.dt = float(cfg.control_dt)

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

        SceneCfg = _make_scene_cfg(cfg)
        self.scene = InteractiveScene(SceneCfg(num_envs=int(cfg.num_envs)))

        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact: ContactSensor = self.scene["contact_forces"]

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)

        self.controlled_joint_ids = self._joint_ids(cfg.controlled_joint_names)
        self.leg_joint_ids = self._joint_ids(cfg.leg_joint_names)
        self.waist_joint_ids = self._joint_ids(cfg.waist_joint_names)
        self.arm_joint_ids = self._joint_ids(cfg.arm_joint_names)
        self.sensor_joint_ids = self._joint_ids(cfg.sensor_joint_names)

        self.controlled_joint_ids_t = torch.tensor(self.controlled_joint_ids, dtype=torch.long, device=self.device)
        self.leg_joint_ids_t = torch.tensor(self.leg_joint_ids, dtype=torch.long, device=self.device)
        self.waist_joint_ids_t = torch.tensor(self.waist_joint_ids, dtype=torch.long, device=self.device)
        self.arm_joint_ids_t = torch.tensor(self.arm_joint_ids, dtype=torch.long, device=self.device)
        self.sensor_joint_ids_t = torch.tensor(self.sensor_joint_ids, dtype=torch.long, device=self.device)

        self.leg_action_ids_t = self._action_ids(cfg.leg_joint_names)
        self.waist_action_ids_t = self._action_ids(cfg.waist_joint_names)
        self.arm_action_ids_t = self._action_ids(cfg.arm_joint_names)

        self.foot_body_ids = self._body_ids(cfg.foot_body_names)
        self.foot_body_ids_t = torch.tensor(self.foot_body_ids, dtype=torch.long, device=self.device)

        self.contact_foot_ids = self._contact_ids(cfg.foot_body_names)
        self.contact_foot_ids_t = torch.tensor(self.contact_foot_ids, dtype=torch.long, device=self.device)

        self.num_envs = int(cfg.num_envs)
        self.num_actions = len(self.controlled_joint_ids)
        self.cfg.num_actions = self.num_actions

        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = torch.zeros_like(self.default_joint_pos)

        self.default_ctrl_pos = self.default_joint_pos[:, self.controlled_joint_ids_t]

        self.joint_limits = self.robot.data.joint_pos_limits.clone()
        self.ctrl_lower = self.joint_limits[:, self.controlled_joint_ids_t, 0]
        self.ctrl_upper = self.joint_limits[:, self.controlled_joint_ids_t, 1]

        self.base_action_scale = self._make_action_scale()
        self.base_ema_alpha = self._make_ema_alpha_tensor()

        self.action_scale = self.base_action_scale.clone()
        self.ema_alpha_tensor = self.base_ema_alpha.clone()

        try:
            self.robot_mass = float(self.robot.root_physx_view.get_masses()[0].sum().item())
        except Exception:
            self.robot_mass = 32.8

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_observations),),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_privileged_obs),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(self.num_actions),),
            dtype=np.float32,
        )

        n, a = self.num_envs, self.num_actions

        self.global_steps = 0
        self.episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.raw_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.phase = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.target_cmd = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.smoothed_cmd = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.last_base_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_acc_obs = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.feet_air_time = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.prev_foot_contact = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_fall_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        self.current_dr_scale = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.dr_motor_eff = torch.ones((n, a), dtype=torch.float32, device=self.device)
        self.dr_alpha_scale = torch.ones((n, 1), dtype=torch.float32, device=self.device)
        self.dr_action_deadzone = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_action_noise_std = torch.zeros((n, 1), dtype=torch.float32, device=self.device)

        self.dr_payload_mass = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.dr_friction = torch.ones(n, dtype=torch.float32, device=self.device)

        self.dr_imu_noise_std = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_q_noise_std = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_qd_noise_std = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_h_noise_std = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_foot_noise_std = torch.zeros((n, 1), dtype=torch.float32, device=self.device)

        self.dr_state_dropout = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_contact_dropout = torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        self.dr_contact_false_positive = torch.zeros((n, 1), dtype=torch.float32, device=self.device)

        self.imu_bias = torch.zeros((n, 6), dtype=torch.float32, device=self.device)
        self.joint_bias = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.action_delay_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.obs_delay_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        self.action_delay_buffer = torch.zeros(
            (n, int(cfg.action_delay_steps_max) + 1, a),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_delay_buffer = torch.zeros(
            (n, int(cfg.obs_delay_steps_max) + 1, int(cfg.num_observations)),
            dtype=torch.float32,
            device=self.device,
        )

        self.push_timer = torch.zeros(n, dtype=torch.long, device=self.device)
        self.push_force = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.is_pushed_flag = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.external_force = torch.zeros((n, 1, 3), dtype=torch.float32, device=self.device)
        self.external_torque = torch.zeros_like(self.external_force)

        if cfg.print_debug_info:
            self._print_debug_info()

        self.reset()

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------
    def _joint_ids(self, names: List[str]) -> List[int]:
        missing = [name for name in names if name not in self.robot_joint_names]
        if missing:
            raise RuntimeError(f"[G1Sim2RealEnv] Missing joints: {missing}")
        return [self.robot_joint_names.index(name) for name in names]

    def _action_ids(self, joint_names: List[str]) -> torch.Tensor:
        ids = [
            self.cfg.controlled_joint_names.index(name)
            for name in joint_names
            if name in self.cfg.controlled_joint_names
        ]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _body_ids(self, names: List[str]) -> List[int]:
        missing = [name for name in names if name not in self.robot_body_names]
        if missing:
            raise RuntimeError(f"[G1Sim2RealEnv] Missing body links: {missing}")
        return [self.robot_body_names.index(name) for name in names]

    def _contact_ids(self, names: List[str]) -> List[int]:
        contact_names = list(self.contact.body_names)
        missing = [name for name in names if name not in contact_names]
        if missing:
            raise RuntimeError(
                f"[G1Sim2RealEnv] Missing contact links: {missing}. "
                f"Available contact sensor body names: {contact_names}"
            )
        return [contact_names.index(name) for name in names]

    def _print_debug_info(self):
        print("\n" + "=" * 100)
        print("[G1Sim2RealEnv] Task4 Sim2Real Robustness Environment Initialized")
        print(f"num_envs             : {self.cfg.num_envs}")
        print(f"num_joints           : {self.robot.num_joints}")
        print(f"num_actions          : {self.num_actions}")
        print(f"num_observations     : {self.cfg.num_observations}")
        print(f"num_privileged_obs   : {self.cfg.num_privileged_obs}")
        print(f"robot_mass           : {self.robot_mass:.3f} kg")
        print(f"controlled_joint_ids : {self.controlled_joint_ids}")
        print(f"sensor_joint_ids     : {self.sensor_joint_ids}")
        print(f"foot_body_ids        : {self.foot_body_ids}")
        print(f"contact_foot_ids     : {self.contact_foot_ids}")
        print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _env_origins(self) -> torch.Tensor:
        if hasattr(self, "env_origins"):
            return self.env_origins
        return self.scene.env_origins

    def _mean_detached(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(float(x), dtype=torch.float32, device=self.device)
        return x.detach().float().mean()

    def _float_tensor(self, x: float) -> torch.Tensor:
        return torch.tensor(float(x), dtype=torch.float32, device=self.device)

    def _smoothstep(self, x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def curriculum_k(self) -> float:
        return min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0))

    def _command_stage(self) -> int:
        k = self.curriculum_k()
        if k < 0.08:
            return 0
        if k < 0.25:
            return 1
        if k < 0.50:
            return 2
        if k < 0.75:
            return 3
        return 4

    def _dr_scale(self) -> float:
        k = self.curriculum_k()
        if k < 0.06:
            return 0.0
        if k < 0.25:
            return 0.35 * self._smoothstep((k - 0.06) / 0.19)
        if k < 0.65:
            return 0.35 + 0.40 * self._smoothstep((k - 0.25) / 0.40)
        return 0.75 + 0.25 * self._smoothstep((k - 0.65) / 0.35)

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
        cmd[:, 0] = torch.empty(int(n), device=self.device).uniform_(*vx_range)
        cmd[:, 1] = torch.empty(int(n), device=self.device).uniform_(*vy_range)
        cmd[:, 2] = torch.empty(int(n), device=self.device).uniform_(*wz_range)
        zero = torch.rand(int(n), device=self.device) < float(self.cfg.zero_command_prob)
        cmd[zero] = 0.0
        return cmd

    @torch.no_grad()
    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return
        new_cmd = self._sample_commands(int(env_ids.numel()))
        self.target_cmd[env_ids] = new_cmd
        self.smoothed_cmd[env_ids] = new_cmd.clone()

    def _sample_range_around_one(self, env_ids: torch.Tensor, low: float, high: float, scale: float) -> torch.Tensor:
        lo = 1.0 + float(scale) * (float(low) - 1.0)
        hi = 1.0 + float(scale) * (float(high) - 1.0)
        return torch.empty((len(env_ids),), dtype=torch.float32, device=self.device).uniform_(lo, hi)

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_domain_randomization(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())
        s = float(self._dr_scale())

        self.current_dr_scale[env_ids] = s

        eff_low, eff_high = self.cfg.motor_efficiency_range
        eff_lo = 1.0 + s * (float(eff_low) - 1.0)
        eff_hi = 1.0 + s * (float(eff_high) - 1.0)
        self.dr_motor_eff[env_ids] = torch.empty((n, self.num_actions), device=self.device).uniform_(eff_lo, eff_hi)

        alpha_low, alpha_high = self.cfg.alpha_scale_range
        self.dr_alpha_scale[env_ids, 0] = self._sample_range_around_one(env_ids, alpha_low, alpha_high, s)

        dz_max = float(self.cfg.action_deadzone_range[1]) * s
        self.dr_action_deadzone[env_ids, 0] = torch.empty(n, device=self.device).uniform_(0.0, dz_max)
        self.dr_action_noise_std[env_ids, 0] = float(self.cfg.action_noise_std_max) * s

        payload_max = float(self.cfg.payload_mass_range[1]) * s
        self.dr_payload_mass[env_ids] = torch.empty(n, device=self.device).uniform_(0.0, payload_max)

        fr_low, fr_high = self.cfg.terrain_friction_range
        fr_lo = 1.0 + s * (float(fr_low) - 1.0)
        fr_hi = 1.0 + s * (float(fr_high) - 1.0)
        self.dr_friction[env_ids] = torch.empty(n, device=self.device).uniform_(fr_lo, fr_hi)

        self.dr_imu_noise_std[env_ids, 0] = float(self.cfg.imu_noise_std_max) * s
        self.dr_q_noise_std[env_ids, 0] = float(self.cfg.joint_pos_noise_std_max) * s
        self.dr_qd_noise_std[env_ids, 0] = float(self.cfg.joint_vel_noise_std_max) * s
        self.dr_h_noise_std[env_ids, 0] = float(self.cfg.root_height_noise_std_max) * s
        self.dr_foot_noise_std[env_ids, 0] = float(self.cfg.foot_pos_noise_std_max) * s

        self.dr_state_dropout[env_ids, 0] = float(self.cfg.state_dropout_prob_max) * s
        self.dr_contact_dropout[env_ids, 0] = float(self.cfg.contact_dropout_prob_max) * s
        self.dr_contact_false_positive[env_ids, 0] = float(self.cfg.contact_false_positive_prob_max) * s

        max_action_delay = int(round(int(self.cfg.action_delay_steps_max) * s))
        max_obs_delay = int(round(int(self.cfg.obs_delay_steps_max) * s))

        if max_action_delay > 0:
            self.action_delay_steps[env_ids] = torch.randint(0, max_action_delay + 1, (n,), device=self.device)
        else:
            self.action_delay_steps[env_ids] = 0

        if max_obs_delay > 0:
            self.obs_delay_steps[env_ids] = torch.randint(0, max_obs_delay + 1, (n,), device=self.device)
        else:
            self.obs_delay_steps[env_ids] = 0

        self.imu_bias[env_ids] = 0.0
        self.joint_bias[env_ids] = 0.0
        self.push_timer[env_ids] = 0
        self.push_force[env_ids] = 0.0
        self.is_pushed_flag[env_ids] = False
        self.action_delay_buffer[env_ids] = 0.0
        self.obs_delay_buffer[env_ids] = 0.0

    def _make_action_scale(self) -> torch.Tensor:
        scale = torch.full((self.num_actions,), float(self.cfg.shoulder_action_scale), device=self.device)
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

    def _make_ema_alpha_tensor(self) -> torch.Tensor:
        alpha = torch.full((self.cfg.num_envs, self.num_actions), float(self.cfg.ema_alpha_legs), device=self.device)
        if self.waist_action_ids_t.numel() > 0:
            alpha[:, self.waist_action_ids_t] = float(self.cfg.ema_alpha_waist)
        if self.arm_action_ids_t.numel() > 0:
            alpha[:, self.arm_action_ids_t] = float(self.cfg.ema_alpha_arms)
        return alpha


    @torch.no_grad()
    def _freeze_sensor_joints(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Hard-freeze non-control sensor joints.

        G1 has two sensor joints:
            - xl330_joint
            - d455_joint

        They are not part of the 23-DoF policy action. During robustness
        stress tests with action noise, payload proxy and external force,
        these joints can drift slightly if only position targets are used.
        For Task4, they must remain fixed, so we explicitly write them back
        to default position and zero velocity.
        """
        if len(getattr(self, "sensor_joint_ids", [])) == 0:
            return

        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return

        q = self.robot.data.joint_pos[env_ids].clone()
        qd = self.robot.data.joint_vel[env_ids].clone()

        q[:, self.sensor_joint_ids_t] = self.default_joint_pos[env_ids][:, self.sensor_joint_ids_t]
        qd[:, self.sensor_joint_ids_t] = 0.0

        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)

        # Keep IsaacLab data tensors consistent for immediate post-step tests.
        try:
            self.scene.update(dt=0.0)
        except Exception:
            pass


    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, device=self.device)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())

        if n == 0:
            raw_obs = self._compute_policy_obs(noisy=True)
            return self._get_delayed_obs(raw_obs, update_buffer=False), {}

        self._sample_domain_randomization(env_ids)

        origins = self._env_origins()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = origins[env_ids, 0:2]
        root_state[:, 2] = origins[env_ids, 2] + float(self.cfg.target_height)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        root_state[:, 7:13] = 0.0

        q0 = self.default_joint_pos[env_ids].clone()
        qd0 = torch.zeros_like(q0)

        if len(self.sensor_joint_ids) > 0:
            q0[:, self.sensor_joint_ids_t] = self.default_joint_pos[env_ids][:, self.sensor_joint_ids_t]
            qd0[:, self.sensor_joint_ids_t] = 0.0

        lower = self.joint_limits[env_ids, :, 0]
        upper = self.joint_limits[env_ids, :, 1]
        q0 = torch.clamp(q0, lower, upper)

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)
        self.robot.reset(env_ids)
        self._freeze_sensor_joints(env_ids)

        self.scene.update(dt=0.0)

        self._resample_commands(env_ids)

        self.raw_action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.action_delay_buffer[env_ids] = 0.0
        self.obs_delay_buffer[env_ids] = 0.0

        self.last_base_vel[env_ids] = self.robot.data.root_lin_vel_b[env_ids]
        self.base_acc_obs[env_ids] = 0.0

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.phase[env_ids] = torch.rand(n, device=self.device)

        self.feet_air_time[env_ids] = 0.0
        self.prev_foot_contact[env_ids] = 0.0

        raw_obs = self._compute_policy_obs(noisy=True)
        self.obs_delay_buffer[env_ids] = raw_obs[env_ids].unsqueeze(1).repeat(
            1,
            int(self.cfg.obs_delay_steps_max) + 1,
            1,
        )

        return self._get_delayed_obs(raw_obs, update_buffer=False), {}

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        self.raw_action = actions.clone()

        resample = (
            (self.episode_steps % int(self.cfg.resample_command_steps) == 0)
            & (self.episode_steps > 0)
        )
        resample_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self.target_cmd[resample_ids] = self._sample_commands(int(resample_ids.numel()))

        self.smoothed_cmd = (
            float(self.cfg.cmd_smoothing_factor) * self.target_cmd
            + (1.0 - float(self.cfg.cmd_smoothing_factor)) * self.smoothed_cmd
        )

        self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=1)
        self.action_delay_buffer[:, 0, :] = actions

        env_arange = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        delayed = self.action_delay_buffer[env_arange, self.action_delay_steps, :]

        deadzone = self.dr_action_deadzone
        delayed = torch.sign(delayed) * torch.clamp(
            (torch.abs(delayed) - deadzone) / torch.clamp(1.0 - deadzone, min=1e-6),
            0.0,
            1.0,
        )

        if float(self.cfg.action_noise_std_max) > 0.0:
            delayed = delayed + torch.randn_like(delayed) * self.dr_action_noise_std

        if float(self.cfg.action_quantization) > 0.0:
            q = float(self.cfg.action_quantization)
            delayed = torch.round(delayed / q) * q

        delayed = torch.clamp(delayed, -1.0, 1.0)

        alpha = torch.clamp(self.base_ema_alpha * self.dr_alpha_scale, 0.20, 0.85)
        self.prev_action = self.last_action.clone()
        filtered = alpha * delayed + (1.0 - alpha) * self.last_action

        applied_action = torch.clamp(filtered * self.dr_motor_eff, -1.0, 1.0)
        self.last_action = applied_action.clone()

        target_ctrl = self.default_ctrl_pos + applied_action * self.base_action_scale
        target_ctrl = torch.clamp(target_ctrl, self.ctrl_lower, self.ctrl_upper)

        full_target = self.default_joint_pos.clone()
        full_target[:, self.controlled_joint_ids_t] = target_ctrl
        full_target[:, self.sensor_joint_ids_t] = self.default_joint_pos[:, self.sensor_joint_ids_t]

        self._update_pushes_and_forces()

        self.robot.set_joint_position_target(full_target)
        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(dt=float(self.cfg.sim_dt))

        self._freeze_sensor_joints()

        self.global_steps += int(self.cfg.num_envs)
        self.episode_steps += 1

        cmd_speed = torch.clamp(
            torch.linalg.norm(self.smoothed_cmd[:, :2], dim=-1)
            + 0.5 * torch.abs(self.smoothed_cmd[:, 2]),
            0.5,
            1.6,
        )
        self.phase = torch.remainder(
            self.phase + float(self.dt) * float(self.cfg.gait_freq_hz) * cmd_speed,
            1.0,
        )

        rewards, terminated, truncated, info = self._compute_rewards()
        self.episode_return += rewards

        raw_obs = self._compute_policy_obs(noisy=True)
        obs = self._get_delayed_obs(raw_obs, update_buffer=True)

        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids)

        return obs, rewards, terminated, truncated, info

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Sim2Real mechanics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_pushes_and_forces(self) -> None:
        dr = self.current_dr_scale
        cmd_speed = torch.linalg.norm(self.smoothed_cmd[:, :2], dim=-1)
        cmd_yaw = torch.abs(self.smoothed_cmd[:, 2])
        moving = (cmd_speed > 0.035) | (cmd_yaw > 0.08)

        active = self.push_timer > 0
        self.is_pushed_flag = active.clone()
        self.push_timer[active] -= 1

        can_start = self.push_timer <= 0
        phase_sensitive = (self.phase < 0.08) | ((self.phase > 0.45) & (self.phase < 0.58))
        prob = float(self.cfg.push_prob_per_step_max) * dr

        start_push = (
            can_start
            & phase_sensitive
            & moving
            & (torch.rand(self.cfg.num_envs, device=self.device) < prob)
        )

        if start_push.any():
            ids = start_push.nonzero(as_tuple=False).squeeze(-1)
            mag = torch.empty(len(ids), device=self.device).uniform_(
                float(self.cfg.push_force_range[0]),
                float(self.cfg.push_force_range[1]),
            )
            angle = torch.empty(len(ids), device=self.device).uniform_(-math.pi, math.pi)

            self.push_force[ids, 0] = mag * torch.cos(angle)
            self.push_force[ids, 1] = mag * torch.sin(angle)
            self.push_force[ids, 2] = 0.0

            low, high = self.cfg.push_duration_steps_range
            self.push_timer[ids] = torch.randint(int(low), int(high) + 1, (len(ids),), device=self.device)
            self.is_pushed_flag[ids] = True

        self.external_force.zero_()
        self.external_torque.zero_()

        self.external_force[:, 0, 2] += -self.dr_payload_mass * 9.81

        active_push = self.push_timer > 0
        self.external_force[active_push, 0, :] += self.push_force[active_push]

        low_friction = torch.clamp(0.85 - self.dr_friction, min=0.0)
        slip_stress = (
            torch.randn((self.cfg.num_envs, 2), device=self.device)
            * float(self.cfg.slip_stress_force_max)
            * low_friction.unsqueeze(-1)
            * dr.unsqueeze(-1)
            * moving.float().unsqueeze(-1)
        )
        self.external_force[:, 0, 0:2] += slip_stress

        try:
            self.robot.set_external_force_and_torque(
                forces=self.external_force,
                torques=self.external_torque,
                body_ids=[0],
                is_global=True,
            )
        except Exception:
            pass

    def _apply_dropout(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if p.max().item() <= 0.0:
            return x
        while p.ndim < x.ndim:
            p = p.unsqueeze(-1)
        mask = (torch.rand_like(x) > p).float()
        return x * mask

    def _get_feet_contact(self, noisy: bool = False):
        data = self.contact.data

        if hasattr(data, "net_forces_w_history") and data.net_forces_w_history is not None:
            forces = data.net_forces_w_history[:, :, self.contact_foot_ids_t, :]
            normal_force = torch.max(forces[..., 2], dim=1)[0]
        else:
            forces = data.net_forces_w[:, self.contact_foot_ids_t, :]
            normal_force = forces[..., 2]

        contact = (normal_force > float(self.cfg.contact_force_threshold)).float()

        if noisy:
            drop = torch.rand_like(contact) < self.dr_contact_dropout
            fp = torch.rand_like(contact) < self.dr_contact_false_positive
            contact = torch.where(drop, torch.zeros_like(contact), contact)
            contact = torch.where(fp, torch.ones_like(contact), contact)

        return contact, normal_force

    def _build_contact_ref(self) -> torch.Tensor:
        left_phase = self.phase
        right_phase = torch.remainder(self.phase + 0.5, 1.0)
        left = (left_phase < float(self.cfg.contact_duty_ratio)).float()
        right = (right_phase < float(self.cfg.contact_duty_ratio)).float()
        return torch.stack([left, right], dim=-1)

    def _get_delayed_obs(self, raw_obs: torch.Tensor, update_buffer: bool) -> torch.Tensor:
        if update_buffer:
            self.obs_delay_buffer = torch.roll(self.obs_delay_buffer, shifts=1, dims=1)
            self.obs_delay_buffer[:, 0, :] = raw_obs

        env_arange = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        return self.obs_delay_buffer[env_arange, self.obs_delay_steps, :]

    # ------------------------------------------------------------------
    # Observation / privileged state
    # ------------------------------------------------------------------
    def _compute_policy_obs(self, noisy: bool = True) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b.clone()
        base_ang_vel = self.robot.data.root_ang_vel_b.clone()
        projected_gravity = self.robot.data.projected_gravity_b.clone()

        q = self.robot.data.joint_pos[:, self.controlled_joint_ids_t].clone()
        qd = self.robot.data.joint_vel[:, self.controlled_joint_ids_t].clone()

        root_pos = self.robot.data.root_pos_w.clone()
        origins = self._env_origins()
        root_height = (root_pos[:, 2] - origins[:, 2]).unsqueeze(-1)

        contact, _ = self._get_feet_contact(noisy=noisy)

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :].clone()
        foot_rel_pos = (foot_pos - root_pos.unsqueeze(1)).reshape(self.cfg.num_envs, -1)

        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2].clone()
        foot_vel_xy_flat = foot_vel_xy.reshape(self.cfg.num_envs, -1)

        if noisy:
            scale = self.current_dr_scale.unsqueeze(-1)

            self.imu_bias += torch.randn_like(self.imu_bias) * float(self.cfg.imu_bias_walk_std) * scale
            self.joint_bias += torch.randn_like(self.joint_bias) * float(self.cfg.joint_bias_walk_std) * scale

            base_lin_vel += self.imu_bias[:, 0:3]
            base_ang_vel += self.imu_bias[:, 3:6]

            base_lin_vel += torch.randn_like(base_lin_vel) * self.dr_imu_noise_std
            base_ang_vel += torch.randn_like(base_ang_vel) * self.dr_imu_noise_std

            q += torch.randn_like(q) * self.dr_q_noise_std
            qd += torch.randn_like(qd) * self.dr_qd_noise_std
            q += self.joint_bias

            root_height += torch.randn_like(root_height) * self.dr_h_noise_std
            foot_rel_pos += torch.randn_like(foot_rel_pos) * self.dr_foot_noise_std
            foot_vel_xy_flat += torch.randn_like(foot_vel_xy_flat) * self.dr_foot_noise_std

        q_err = q - self.default_ctrl_pos
        action_delta = self.last_action - self.prev_action

        sin_phase = torch.sin(2.0 * math.pi * self.phase).unsqueeze(-1)
        cos_phase = torch.cos(2.0 * math.pi * self.phase).unsqueeze(-1)
        dr_scale = self.current_dr_scale.unsqueeze(-1)

        obs = torch.cat(
            [
                base_lin_vel,          # 3
                base_ang_vel,          # 3
                projected_gravity,     # 3
                self.smoothed_cmd,     # 3
                q_err,                 # 23
                qd,                    # 23
                self.last_action,      # 23
                action_delta,          # 23
                contact,               # 2
                foot_rel_pos,          # 6
                foot_vel_xy_flat,      # 4
                self.base_acc_obs,     # 3
                sin_phase,             # 1
                cos_phase,             # 1
                dr_scale,              # 1
                root_height,           # 1
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"[G1Sim2RealEnv] actor obs dim mismatch: "
                f"got {obs.shape[-1]}, expected {self.cfg.num_observations}"
            )

        if noisy:
            obs = self._apply_dropout(obs, self.dr_state_dropout)

        return torch.nan_to_num(
            torch.clamp(obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def _compute_obs(self) -> torch.Tensor:
        return self._compute_policy_obs(noisy=True)

    def _compute_privileged_obs(self) -> torch.Tensor:
        clean_obs = self._compute_policy_obs(noisy=False)

        action_delay_norm = self.action_delay_steps.float().unsqueeze(-1) / max(
            float(self.cfg.action_delay_steps_max),
            1.0,
        )
        obs_delay_norm = self.obs_delay_steps.float().unsqueeze(-1) / max(
            float(self.cfg.obs_delay_steps_max),
            1.0,
        )
        privileged_extra = torch.cat(
            [
                self.current_dr_scale.unsqueeze(-1),     # 1
                self.dr_motor_eff,                       # 23
                self.dr_alpha_scale,                     # 1
                self.dr_action_deadzone,                 # 1
                self.dr_action_noise_std,                # 1
                self.dr_payload_mass.unsqueeze(-1),      # 1
                self.dr_friction.unsqueeze(-1),          # 1
                self.dr_imu_noise_std,                   # 1
                self.dr_q_noise_std,                     # 1
                self.dr_qd_noise_std,                    # 1
                self.dr_h_noise_std,                     # 1
                self.dr_foot_noise_std,                  # 1
                self.dr_state_dropout,                   # 1
                self.dr_contact_dropout,                 # 1
                self.dr_contact_false_positive,          # 1
                action_delay_norm,                       # 1
                obs_delay_norm,                          # 1
            ],
            dim=-1,
        )

        expected_extra = int(self.cfg.num_privileged_obs - self.cfg.num_observations)
        if privileged_extra.shape[-1] != expected_extra:
            raise RuntimeError(
                f"[G1Sim2RealEnv] privileged extra dim mismatch: "
                f"got {privileged_extra.shape[-1]}, expected {expected_extra}"
            )

        state = torch.cat([clean_obs, privileged_extra], dim=-1)

        if state.shape[-1] != int(self.cfg.num_privileged_obs):
            raise RuntimeError(
                f"[G1Sim2RealEnv] privileged obs dim mismatch: "
                f"got {state.shape[-1]}, expected {self.cfg.num_privileged_obs}"
            )

        return torch.nan_to_num(
            torch.clamp(state, -20.0, 20.0),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

    def _compute_states(self) -> torch.Tensor:
        return self._compute_privileged_obs()

    def get_privileged_observations(self) -> torch.Tensor:
        return self._compute_privileged_obs()

    # ------------------------------------------------------------------
    # Rewards / termination
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

        origins = self._env_origins()
        root_pos = self.robot.data.root_pos_w
        base_height = root_pos[:, 2] - origins[:, 2]

        base_acc = (base_lin_vel - self.last_base_vel) / max(float(self.dt), 1e-6)
        self.base_acc_obs.copy_(base_acc)
        self.last_base_vel.copy_(base_lin_vel)

        contact, normal_force = self._get_feet_contact(noisy=False)
        contact_count = contact.sum(dim=-1)

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_z = foot_pos[:, :, 2] - origins[:, 2].unsqueeze(-1)
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

        lin_error = torch.square(vx - cmd_vx) + torch.square(vy - cmd_vy)
        yaw_error = torch.square(wz - cmd_wz)

        r_cmd_lin = torch.exp(-float(self.cfg.sigma_cmd_lin) * lin_error)
        r_cmd_yaw = torch.exp(-float(self.cfg.sigma_cmd_yaw) * yaw_error)

        actual_speed = torch.sqrt(torch.square(vx) + torch.square(vy) + 1e-6)
        target_speed = torch.sqrt(torch.square(cmd_vx) + torch.square(cmd_vy) + 1e-6)

        r_cmd_speed = torch.exp(-8.0 * torch.square(actual_speed - target_speed)) * moving
        p_under_speed = -torch.relu(target_speed - actual_speed) * moving
        p_over_speed = -torch.relu(actual_speed - target_speed - 0.12) * moving

        r_zero_vel = torch.exp(
            -float(self.cfg.sigma_zero)
            * (torch.square(vx) + torch.square(vy) + 0.50 * torch.square(wz))
        )
        p_yaw_drift = -standing * torch.abs(wz)

        double_contact_penalty = -moving * torch.clamp(contact_count - 1.20, min=0.0)

        ref_contact = self._build_contact_ref()
        r_phase_contact = 1.0 - torch.mean(torch.abs(contact - ref_contact), dim=-1)
        r_phase_contact = r_phase_contact * moving

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += float(self.dt)

        r_air_time = torch.sum(
            torch.clamp(self.feet_air_time - 0.10, min=0.0, max=0.45) * first_contact.float(),
            dim=-1,
        ) * moving

        self.feet_air_time = torch.where(
            contact > 0.5,
            torch.zeros_like(self.feet_air_time),
            self.feet_air_time,
        )
        self.prev_foot_contact.copy_(contact)

        r_clearance = (
            (1.0 - contact)
            * torch.exp(-20.0 * torch.abs(foot_z - float(self.cfg.foot_clearance_target)))
        ).sum(dim=-1) * moving

        roll_pitch_mag = torch.linalg.norm(projected_gravity[:, :2], dim=-1)
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

        r_recovery = torch.exp(-3.0 * roll_pitch_mag) * (
            1.0 - torch.clamp(contact_count / 4.0, 0.0, 1.0) * 0.25
        )

        push_active = self.is_pushed_flag.float()
        r_push_survival = (
            push_active
            * torch.exp(-2.5 * roll_pitch_mag)
            * torch.exp(-2.0 * torch.abs(base_height - float(self.cfg.target_height)))
        )

        motor_stress = torch.mean(
            torch.abs(self.last_action)
            * torch.clamp(1.0 / torch.clamp(self.dr_motor_eff, min=0.20), 0.0, 5.0),
            dim=-1,
        )
        p_motor_temp = -motor_stress * self.current_dr_scale

        r_alive = torch.ones_like(vx)
        joint_vel_abs_max = torch.abs(self.robot.data.joint_vel).max(dim=-1)[0]

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
            + float(self.cfg.w_zero_vel) * r_zero_vel * standing
            + float(self.cfg.w_under_speed) * p_under_speed
            + float(self.cfg.w_yaw_drift) * p_yaw_drift
            + float(self.cfg.w_double_contact) * double_contact_penalty
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_recovery) * r_recovery
            + float(self.cfg.w_push_survival) * r_push_survival
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_base_ang_vel) * p_base_ang
            + float(self.cfg.w_base_acc) * p_base_acc
            + float(self.cfg.w_com_support) * r_com_support
            + float(self.cfg.w_z_vel) * p_z_vel
            + float(self.cfg.w_over_speed) * p_over_speed
            + float(self.cfg.w_default_pose) * p_default_pose
            + float(self.cfg.w_alive) * r_alive
            + float(self.cfg.w_joint_limit) * p_joint_limit
            + float(self.cfg.w_action_rate) * p_action_rate
            + float(self.cfg.w_action_mag) * p_action_mag
            + float(self.cfg.w_foot_slip) * p_slip
            + float(self.cfg.w_energy) * p_energy
            + float(self.cfg.w_motor_temp) * p_motor_temp
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
            self.total_done_episodes += done.float().sum()
            self.total_fall_episodes += terminated.float().sum()
            self.total_timeout_episodes += truncated.float().sum()

        total_done_safe = torch.clamp(self.total_done_episodes, min=1.0)
        command_stage = int(self._command_stage())
        dr_scale = float(self._dr_scale())

        info = {
            "reward_components": {
                "R_Cmd_Lin": self._mean_detached(float(self.cfg.w_cmd_lin) * (moving * r_cmd_lin + standing * r_zero_vel)),
                "R_Cmd_Speed": self._mean_detached(float(self.cfg.w_cmd_speed) * r_cmd_speed),
                "R_Cmd_Yaw": self._mean_detached(float(self.cfg.w_cmd_yaw) * r_cmd_yaw),
                "R_Zero_Vel": self._mean_detached(float(self.cfg.w_zero_vel) * r_zero_vel * standing),
                "P_Under_Speed": self._mean_detached(float(self.cfg.w_under_speed) * p_under_speed),
                "P_Yaw_Drift": self._mean_detached(float(self.cfg.w_yaw_drift) * p_yaw_drift),
                "P_Double_Contact": self._mean_detached(float(self.cfg.w_double_contact) * double_contact_penalty),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Recovery": self._mean_detached(float(self.cfg.w_recovery) * r_recovery),
                "R_Push_Survival": self._mean_detached(float(self.cfg.w_push_survival) * r_push_survival),
                "R_Upright": self._mean_detached(float(self.cfg.w_upright) * r_upright),
                "R_Height": self._mean_detached(float(self.cfg.w_height) * r_height),
                "P_Base_Ang": self._mean_detached(float(self.cfg.w_base_ang_vel) * p_base_ang),
                "P_Base_Acc": self._mean_detached(float(self.cfg.w_base_acc) * p_base_acc),
                "R_COM_Support": self._mean_detached(float(self.cfg.w_com_support) * r_com_support),
                "P_Z_Vel": self._mean_detached(float(self.cfg.w_z_vel) * p_z_vel),
                "P_Over_Speed": self._mean_detached(float(self.cfg.w_over_speed) * p_over_speed),
                "P_Default_Pose": self._mean_detached(float(self.cfg.w_default_pose) * p_default_pose),
                "R_Alive": self._mean_detached(float(self.cfg.w_alive) * r_alive),
                "P_Joint_Limit": self._mean_detached(float(self.cfg.w_joint_limit) * p_joint_limit),
                "P_Action_Rate": self._mean_detached(float(self.cfg.w_action_rate) * p_action_rate),
                "P_Action_Mag": self._mean_detached(float(self.cfg.w_action_mag) * p_action_mag),
                "P_Foot_Slip": self._mean_detached(float(self.cfg.w_foot_slip) * p_slip),
                "P_Energy": self._mean_detached(float(self.cfg.w_energy) * p_energy),
                "P_Motor_Temp": self._mean_detached(float(self.cfg.w_motor_temp) * p_motor_temp),
                "Continuous": self._mean_detached(continuous),
                "Event_Fall": self._mean_detached(event_fall),
                "Total": self._mean_detached(reward),
            },
            "events": {
                "Fall_Rate": self._mean_detached(terminated.float()),
                "Timeout_Rate": self._mean_detached(truncated.float()),
                "Done_Rate": self._mean_detached(done.float()),
                "Push_Active_Rate": self._mean_detached(push_active),
                "Episode_Fall_Total_Rate": self.total_fall_episodes / total_done_safe,
                "Episode_Timeout_Total_Rate": self.total_timeout_episodes / total_done_safe,
            },
            "telemetry": {
                "Curriculum_K": self._float_tensor(self.curriculum_k()),
                "Command_Stage": self._float_tensor(float(command_stage)),
                "DR_Scale": self._float_tensor(float(dr_scale)),
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
                "Contact_Count": self._mean_detached(contact_count),
                "Left_Contact": self._mean_detached(contact[:, 0]),
                "Right_Contact": self._mean_detached(contact[:, 1]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Foot_Slip_Raw": self._mean_detached(-p_slip),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Global_Steps": self._float_tensor(float(self.global_steps)),
                "Motor_Eff_Mean": self._mean_detached(self.dr_motor_eff),
                "Alpha_Scale_Mean": self._mean_detached(self.dr_alpha_scale),
                "Action_Deadzone": self._mean_detached(self.dr_action_deadzone),
                "Action_Noise_Std": self._mean_detached(self.dr_action_noise_std),
                "Payload_Mass": self._mean_detached(self.dr_payload_mass),
                "Friction_Proxy": self._mean_detached(self.dr_friction),
                "IMU_Noise": self._mean_detached(self.dr_imu_noise_std),
                "Joint_Pos_Noise": self._mean_detached(self.dr_q_noise_std),
                "Joint_Vel_Noise": self._mean_detached(self.dr_qd_noise_std),
                "State_Dropout": self._mean_detached(self.dr_state_dropout),
                "Contact_Dropout": self._mean_detached(self.dr_contact_dropout),
                "Contact_False_Positive": self._mean_detached(self.dr_contact_false_positive),
                "Action_Delay": self._mean_detached(self.action_delay_steps.float()),
                "Obs_Delay": self._mean_detached(self.obs_delay_steps.float()),
            },
            "debug": {
                "Obs_Dim": self._float_tensor(float(self.cfg.num_observations)),
                "Privileged_Obs_Dim": self._float_tensor(float(self.cfg.num_privileged_obs)),
                "Action_Dim": self._float_tensor(float(self.num_actions)),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Base_Height_Min": base_height.detach().min(),
                "Base_Height_Max": base_height.detach().max(),
                "JointVel_Max": joint_vel_abs_max.detach().max(),
                "External_Force_Norm": torch.linalg.norm(self.external_force.reshape(self.cfg.num_envs, -1), dim=-1).detach().mean(),
            },
        }

        return reward, terminated, truncated, info


Task4ConfigAlias = Task4Config
Task4Env = G1Sim2RealEnv
G1Task4Env = G1Sim2RealEnv
G1Sim2RealRobustEnv = G1Sim2RealEnv
