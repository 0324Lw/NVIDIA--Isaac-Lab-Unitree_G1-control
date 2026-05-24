from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Unitree G1 Task4 Sim2Real RMA PPO from scratch")

parser.add_argument("--total-env-steps", type=int, default=700_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)

parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK4_MOTION_FILE", ""))

parser.add_argument("--resume", type=str, default="", help="Optional full Task4 checkpoint to resume")

parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--min-lr", type=float, default=2e-5)
parser.add_argument("--max-lr", type=float, default=1e-4)
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--entropy-coef", type=float, default=0.0010)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=0.5)

parser.add_argument("--latent-dim", type=int, default=32)
parser.add_argument("--teacher-peak-ratio", type=float, default=0.20)
parser.add_argument("--teacher-bootstrap-env-steps", type=int, default=120_000_000)
parser.add_argument("--teacher-peak-env-steps", type=int, default=300_000_000)
parser.add_argument("--teacher-decay-end-env-steps", type=int, default=550_000_000)
parser.add_argument("--distill-coef", type=float, default=0.03)
parser.add_argument("--teacher-action-coef", type=float, default=0.005)

parser.add_argument("--target-kl", type=float, default=0.015)
parser.add_argument("--hard-kl-stop", type=float, default=0.08)
parser.add_argument("--init-log-std", type=float, default=-1.6)
parser.add_argument("--max-log-std", type=float, default=-0.8)
parser.add_argument("--min-log-std", type=float, default=-4.0)

parser.add_argument("--log-root", type=str, default=str(PROJECT_ROOT / "logs" / "task4"))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

from g1_rl.tasks.task4.task4_config import Task4Config
from g1_rl.tasks.task4.task4_env import G1Sim2RealEnv


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (list, tuple)):
            return float(np.mean(x)) if len(x) else None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flat_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in (d or {}).items():
        name = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat_dict(v, name))
        else:
            val = to_float(v)
            if val is not None:
                out[name] = val
    return out


def write_scalars(writer: SummaryWriter, data: Dict[str, Any], step: int, prefix: str) -> None:
    for k, v in (data or {}).items():
        val = to_float(v)
        if val is not None:
            try:
                writer.add_scalar(f"{prefix}/{k}".replace("//", "/"), val, step)
            except Exception:
                pass


def make_table(title: str, data: Dict[str, Any], width: int = 112) -> str:
    lines = ["-" * width, f"| {title:<{width - 4}} |", "-" * width]
    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)
    for k in sorted(data.keys()):
        v = data[k]
        ks = (k[:68] + "...") if len(k) > 71 else k
        if isinstance(v, float):
            vs = f"{v:.6e}" if abs(v) > 1e4 or 0 < abs(v) < 1e-3 else f"{v:.6f}"
        else:
            vs = str(v)
        vs = (vs[:36] + "...") if len(vs) > 39 else vs
        lines.append(f"| {ks:<71} | {vs:>{width - 78}} |")
    lines.append("-" * width)
    return "\n".join(lines)


class RunningMeanStd:
    def __init__(self, shape, device, eps: float = 1e-4, clip: float = 10.0):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(eps, dtype=torch.float32, device=device)
        self.clip = float(clip)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=x.device)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)

    def state_dict(self):
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count.detach().cpu(),
            "clip": self.clip,
        }

    def load_state_dict(self, state):
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count.copy_(state["count"].to(self.count.device))
        self.clip = float(state.get("clip", self.clip))


def configure_from_scratch_env(env_cfg: Task4Config) -> None:
    env_cfg.curriculum_total_steps = 700_000_000
    env_cfg.max_episode_length = 1200
    env_cfg.zero_command_prob = 0.12

    env_cfg.cmd_vx_stage0 = (0.00, 0.03)
    env_cfg.cmd_vy_stage0 = (0.00, 0.00)
    env_cfg.cmd_wz_stage0 = (0.00, 0.00)

    env_cfg.cmd_vx_stage1 = (0.02, 0.06)
    env_cfg.cmd_vy_stage1 = (0.00, 0.00)
    env_cfg.cmd_wz_stage1 = (0.00, 0.00)

    env_cfg.cmd_vx_stage2 = (0.03, 0.10)
    env_cfg.cmd_vy_stage2 = (0.00, 0.00)
    env_cfg.cmd_wz_stage2 = (-0.04, 0.04)

    env_cfg.cmd_vx_stage3 = (-0.03, 0.14)
    env_cfg.cmd_vy_stage3 = (-0.04, 0.04)
    env_cfg.cmd_wz_stage3 = (-0.08, 0.08)

    env_cfg.cmd_vx_stage4 = (-0.06, 0.20)
    env_cfg.cmd_vy_stage4 = (-0.08, 0.08)
    env_cfg.cmd_wz_stage4 = (-0.15, 0.15)

    env_cfg.w_cmd_lin = 0.045
    env_cfg.w_cmd_speed = 0.48
    env_cfg.w_cmd_yaw = 0.085
    env_cfg.w_zero_vel = 0.006
    env_cfg.w_under_speed = 0.24
    env_cfg.w_yaw_drift = 0.045

    env_cfg.w_double_contact = 0.09
    env_cfg.w_phase_contact = 0.105
    env_cfg.w_air_time = 0.085
    env_cfg.w_clearance = 0.075

    env_cfg.w_upright = 0.100
    env_cfg.w_height = 0.100
    env_cfg.w_base_ang_vel = 0.040
    env_cfg.w_base_acc = 0.0010
    env_cfg.w_com_support = 0.018
    env_cfg.w_z_vel = 0.022

    env_cfg.w_default_pose = 0.016
    env_cfg.w_alive = 0.0010
    env_cfg.w_joint_limit = 0.05
    env_cfg.w_action_rate = 0.006
    env_cfg.w_action_mag = 0.0015
    env_cfg.w_foot_slip = 0.065
    env_cfg.w_energy = 0.0012
    env_cfg.w_motor_temp = 0.002

    env_cfg.sigma_cmd_lin = 36.0
    env_cfg.sigma_cmd_yaw = 8.0
    env_cfg.sigma_zero = 12.0


def install_slow_dr_curriculum(env: G1Sim2RealEnv) -> None:
    import types

    def slow_dr_scale(self):
        k = self.curriculum_k()
        if k < 0.15:
            return 0.0
        if k < 0.35:
            return 0.25 * self._smoothstep((k - 0.15) / 0.20)
        if k < 0.70:
            return 0.25 + 0.50 * self._smoothstep((k - 0.35) / 0.35)
        return 0.75 + 0.25 * self._smoothstep((k - 0.70) / 0.30)

    env._dr_scale = types.MethodType(slow_dr_scale, env)


class G1RMAFrameStackWrapper(gym.Env):
    def __init__(self, env: G1Sim2RealEnv, log_dir: str, n_stack: int = 5):
        super().__init__()
        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = env.cfg.num_envs
        self.device = env.device
        self.single_obs_dim = env.observation_space.shape[0]
        self.stacked_obs_dim = self.single_obs_dim * self.n_stack
        self.priv_dim = env.state_space.shape[0]

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.stacked_obs_dim,), dtype=np.float32)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.priv_dim,), dtype=np.float32)
        self.action_space = env.action_space

        self.obs_stack = torch.zeros((self.num_envs, self.stacked_obs_dim), dtype=torch.float32, device=self.device)
        self.writer = SummaryWriter(log_dir)

        self.last_info = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0
        self.global_env_steps = 0

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)
        for i in range(self.n_stack):
            self.obs_stack[:, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs
        priv = self.env.get_privileged_observations()
        self.last_info = info or {}
        return self.obs_stack.clone(), priv.clone(), info

    def step(self, actions: torch.Tensor):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.obs_stack[:, :-self.single_obs_dim] = self.obs_stack[:, self.single_obs_dim :].clone()
        self.obs_stack[:, -self.single_obs_dim :] = obs

        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                self.obs_stack[ids, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs[ids]

        priv = self.env.get_privileged_observations()

        self.global_env_steps += self.num_envs
        self.last_info = info or {}
        self.last_reward_mean = to_float(reward) or 0.0
        self.last_done_count = int(done.sum().detach().cpu().item())

        write_scalars(self.writer, info.get("reward_components", {}), self.global_env_steps, "rewards")
        write_scalars(self.writer, info.get("events", {}), self.global_env_steps, "events")
        write_scalars(self.writer, info.get("telemetry", {}), self.global_env_steps, "telemetry")
        write_scalars(self.writer, info.get("debug", {}), self.global_env_steps, "debug")
        self.writer.add_scalar("rollout/reward_mean_raw", self.last_reward_mean, self.global_env_steps)
        self.writer.add_scalar("rollout/done_count", self.last_done_count, self.global_env_steps)

        return self.obs_stack.clone(), priv.clone(), reward, terminated, truncated, info

    def close(self):
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass


class RMATeacherStudentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        priv_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        init_log_std: float = -1.6,
        min_log_std: float = -4.0,
        max_log_std: float = -0.8,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.priv_dim = int(priv_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.teacher_encoder = nn.Sequential(
            nn.Linear(priv_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        self.student_adapter = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + priv_dim + latent_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.apply(self._orthogonal_init)

        with torch.no_grad():
            self.actor[0].weight[:, obs_dim:] *= 0.0
            self.log_std.fill_(float(init_log_std))

    @staticmethod
    def _orthogonal_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def encode(self, obs, priv):
        z_teacher = torch.tanh(self.teacher_encoder(priv))
        z_student = torch.tanh(self.student_adapter(obs))
        return z_student, z_teacher

    def actor_mean_from_z(self, obs, z):
        return self.actor(torch.cat([obs, z], dim=-1))

    def value(self, obs, priv, z_teacher):
        return self.critic(torch.cat([obs, priv, z_teacher], dim=-1)).squeeze(-1)

    def distribution(self, obs, priv, teacher_ratio: float):
        z_student, z_teacher = self.encode(obs, priv)
        ratio = float(max(0.0, min(1.0, teacher_ratio)))
        z_mix = (1.0 - ratio) * z_student + ratio * z_teacher
        mean = self.actor_mean_from_z(obs, z_mix)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value(obs, priv, z_teacher)

        with torch.no_grad():
            mean_student = self.actor_mean_from_z(obs, z_student)
            mean_teacher = self.actor_mean_from_z(obs, z_teacher)

        aux = {
            "z_student": z_student,
            "z_teacher": z_teacher,
            "mean_student": mean_student,
            "mean_teacher": mean_teacher,
            "mean": mean,
            "std": std,
        }
        return dist, value, aux

    @torch.no_grad()
    def act(self, obs, priv, teacher_ratio: float):
        dist, value, aux = self.distribution(obs, priv, teacher_ratio)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return action, raw_action, log_prob, value, aux

    def evaluate_raw_actions(self, obs, priv, raw_actions, teacher_ratio: float):
        dist, value, aux = self.distribution(obs, priv, teacher_ratio)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value, aux

    @torch.no_grad()
    def act_student_only(self, obs):
        z = torch.tanh(self.student_adapter(obs))
        return torch.tanh(self.actor_mean_from_z(obs, z))


class RolloutStorage:
    def __init__(self, rollouts, num_envs, obs_dim, priv_dim, action_dim, device):
        self.T = int(rollouts)
        self.N = int(num_envs)
        self.device = device
        self.obs = torch.zeros((self.T, self.N, obs_dim), dtype=torch.float32, device=device)
        self.priv = torch.zeros((self.T, self.N, priv_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.T, self.N, action_dim), dtype=torch.float32, device=device)
        self.raw_actions = torch.zeros((self.T, self.N, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.returns = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)

    def flatten(self):
        return {
            "obs": self.obs.reshape(self.T * self.N, -1),
            "priv": self.priv.reshape(self.T * self.N, -1),
            "actions": self.actions.reshape(self.T * self.N, -1),
            "raw_actions": self.raw_actions.reshape(self.T * self.N, -1),
            "log_probs": self.log_probs.reshape(self.T * self.N),
            "values": self.values.reshape(self.T * self.N),
            "returns": self.returns.reshape(self.T * self.N),
            "advantages": self.advantages.reshape(self.T * self.N),
        }


def teacher_ratio_schedule(env_steps: int, args) -> float:
    b = int(args.teacher_bootstrap_env_steps)
    p = int(args.teacher_peak_env_steps)
    e = int(args.teacher_decay_end_env_steps)
    peak = float(args.teacher_peak_ratio)

    if env_steps < b:
        return 0.0
    if env_steps < p:
        x = (env_steps - b) / max(p - b, 1)
        x = x * x * (3.0 - 2.0 * x)
        return peak * x
    if env_steps < e:
        x = (env_steps - p) / max(e - p, 1)
        x = x * x * (3.0 - 2.0 * x)
        return peak * (1.0 - x)
    return 0.0


def adaptive_lr_by_kl(optimizer, approx_kl: float, target_kl: float, min_lr: float, max_lr: float) -> float:
    lr = optimizer.param_groups[0]["lr"]
    new_lr = lr

    if approx_kl > target_kl * 1.5:
        new_lr = max(lr / 1.5, min_lr)
    elif approx_kl < target_kl / 1.5:
        new_lr = min(lr * 1.1, max_lr)

    if abs(new_lr - lr) > 1e-12:
        for group in optimizer.param_groups:
            group["lr"] = new_lr

    return float(new_lr)


@torch.no_grad()
def compute_gae(storage: RolloutStorage, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
    last_gae = torch.zeros(storage.N, dtype=torch.float32, device=storage.device)
    for t in reversed(range(storage.T)):
        if t == storage.T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - storage.dones[t]
        else:
            next_value = storage.values[t + 1]
            next_non_terminal = 1.0 - storage.dones[t]
        delta = storage.rewards[t] + gamma * next_value * next_non_terminal - storage.values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        storage.advantages[t] = last_gae
    storage.returns = storage.advantages + storage.values


def ppo_update(model, optimizer, storage: RolloutStorage, args, teacher_ratio: float, writer: SummaryWriter, env_steps: int):
    data = storage.flatten()
    obs = data["obs"]
    priv = data["priv"]
    raw_actions = data["raw_actions"]
    old_log_probs = data["log_probs"]
    old_values = data["values"]
    returns = data["returns"]
    advantages = data["advantages"]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.shape[0]
    mini_batch_size = max(1, batch_size // int(args.mini_batches))
    indices = torch.arange(batch_size, device=obs.device)

    stats = {
        "loss_policy": [],
        "loss_value": [],
        "loss_entropy": [],
        "loss_distill": [],
        "loss_teacher_action": [],
        "approx_kl": [],
        "clip_fraction": [],
        "z_mse": [],
    }

    early_stop = False
    stop_reason = "none"

    for _ in range(int(args.epochs)):
        perm = indices[torch.randperm(batch_size, device=obs.device)]

        for start in range(0, batch_size, mini_batch_size):
            mb = perm[start : start + mini_batch_size]

            mb_obs = obs[mb]
            mb_priv = priv[mb]
            mb_raw_actions = raw_actions[mb]
            mb_old_log_probs = old_log_probs[mb]
            mb_old_values = old_values[mb]
            mb_returns = returns[mb]
            mb_adv = advantages[mb]

            new_log_probs, entropy, values, aux = model.evaluate_raw_actions(
                mb_obs, mb_priv, mb_raw_actions, teacher_ratio=teacher_ratio
            )

            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > float(args.clip_range)).float().mean()

            if torch.isfinite(approx_kl) and approx_kl.item() > float(args.hard_kl_stop):
                stats["approx_kl"].append(approx_kl.detach())
                stats["clip_fraction"].append(clip_fraction.detach())
                early_stop = True
                stop_reason = f"hard_kl_stop>{args.hard_kl_stop}"
                break

            policy_loss = -torch.min(
                ratio * mb_adv,
                torch.clamp(ratio, 1.0 - float(args.clip_range), 1.0 + float(args.clip_range)) * mb_adv,
            ).mean()

            values_clipped = mb_old_values + torch.clamp(
                values - mb_old_values,
                -float(args.value_clip),
                float(args.value_clip),
            )
            value_loss = torch.max(
                torch.square(values - mb_returns),
                torch.square(values_clipped - mb_returns),
            ).mean()

            entropy_loss = -entropy.mean()

            z_student = aux["z_student"]
            z_teacher = aux["z_teacher"].detach()
            distill_loss = F.mse_loss(z_student, z_teacher)

            mean_student = model.actor_mean_from_z(mb_obs, z_student)
            mean_teacher = model.actor_mean_from_z(mb_obs, z_teacher).detach()
            teacher_action_loss = F.mse_loss(mean_student, mean_teacher)

            total_loss = (
                policy_loss
                + float(args.value_coef) * value_loss
                + float(args.entropy_coef) * entropy_loss
                + float(args.distill_coef) * distill_loss
                + float(args.teacher_action_coef) * teacher_action_loss
            )

            if not torch.isfinite(total_loss):
                early_stop = True
                stop_reason = "non_finite_loss"
                break

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            stats["loss_policy"].append(policy_loss.detach())
            stats["loss_value"].append(value_loss.detach())
            stats["loss_entropy"].append(entropy_loss.detach())
            stats["loss_distill"].append(distill_loss.detach())
            stats["loss_teacher_action"].append(teacher_action_loss.detach())
            stats["approx_kl"].append(approx_kl.detach())
            stats["clip_fraction"].append(clip_fraction.detach())
            stats["z_mse"].append(distill_loss.detach())

        if early_stop:
            break

    out = {k: (torch.stack(v).mean().item() if v else 0.0) for k, v in stats.items()}
    out["learning_rate"] = adaptive_lr_by_kl(
        optimizer,
        float(out["approx_kl"]),
        float(args.target_kl),
        float(args.min_lr),
        float(args.max_lr),
    )
    out["teacher_ratio"] = float(teacher_ratio)
    out["early_stop"] = float(early_stop)
    out["stop_reason_code"] = 1.0 if stop_reason != "none" else 0.0

    for k, v in out.items():
        writer.add_scalar(f"ppo/{k}", v, env_steps)

    return out, stop_reason


def save_checkpoint(path, model, optimizer, obs_norm, priv_norm, env_cfg, args, env_steps: int, extra=None):
    os.makedirs(path, exist_ok=True)

    full_path = os.path.join(path, "g1_task4_rma_full_checkpoint.pt")
    deploy_path = os.path.join(path, "g1_task4_student_deploy.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "priv_norm": priv_norm.state_dict(),
            "env_steps": int(env_steps),
            "args": vars(args),
            "metadata": {
                "stage": "g1_task4_sim2real_from_scratch_rma",
                "single_obs_dim": int(env_cfg.num_observations),
                "stacked_obs_dim": int(env_cfg.num_observations * 5),
                "privileged_obs_dim": int(env_cfg.num_privileged_obs),
                "num_actions": int(env_cfg.num_actions),
                "frame_stack": 5,
                "from_scratch": True,
                "student_deploy_only": True,
                "note": "Task4 independent RMA PPO. No inheritance from other task envs.",
            },
            "extra": extra or {},
        },
        full_path,
    )

    torch.save(
        {
            "student_adapter": model.student_adapter.state_dict(),
            "actor": model.actor.state_dict(),
            "log_std": model.log_std.detach().cpu(),
            "obs_norm": obs_norm.state_dict(),
            "metadata": {
                "deploy": "student_only",
                "obs_dim": int(env_cfg.num_observations * 5),
                "single_obs_dim": int(env_cfg.num_observations),
                "action_dim": int(env_cfg.num_actions),
                "latent_dim": int(args.latent_dim),
                "note": "obs_stack -> obs_norm -> student_adapter -> actor([obs_stack, z_student]) -> tanh action",
            },
        },
        deploy_path,
    )


def resolve_checkpoint(path: str) -> str:
    if not path:
        return ""
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return str(p)
    if p.is_dir():
        for name in [
            "g1_task4_rma_full_checkpoint.pt",
            "final_checkpoint/g1_task4_rma_full_checkpoint.pt",
        ]:
            cand = p / name
            if cand.exists():
                return str(cand)
    return str(p)


def try_resume(path, model, optimizer, obs_norm, priv_norm, device) -> int:
    ckpt_path = resolve_checkpoint(path)
    if not ckpt_path:
        return 0
    if not os.path.exists(ckpt_path):
        print(f"[WARN] resume checkpoint 不存在: {ckpt_path}")
        return 0

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    obs_norm.load_state_dict(ckpt["obs_norm"])
    priv_norm.load_state_dict(ckpt["priv_norm"])

    env_steps = int(ckpt.get("env_steps", 0))
    print(f" ✅ 已恢复 Task4 checkpoint: {ckpt_path}, env_steps={env_steps:,}")
    return env_steps


def make_log_dir() -> str:
    name = args_cli.run_name.strip()
    if not name:
        name = f"g1_task4_from_scratch_rma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = os.path.abspath(os.path.join(args_cli.log_root, name))
    os.makedirs(path, exist_ok=True)
    return path


def main():
    set_seed(int(args_cli.seed))

    log_dir = make_log_dir()
    writer = SummaryWriter(log_dir)

    print("\n" + "=" * 112)
    print("🚀 G1 Task4: From-Scratch Low-Speed Sim2Real RMA PPO")
    print("=" * 112)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_dir      = {log_dir}")
    print("[INFO] Task4 是独立 RMA PPO 训练，不继承 Task1/2/3 环境。")

    env_cfg = Task4Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)

    if args_cli.usd_path:
        env_cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        env_cfg.motion_file = str(args_cli.motion_file)

    configure_from_scratch_env(env_cfg)

    base_env = G1Sim2RealEnv(env_cfg)
    install_slow_dr_curriculum(base_env)

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * base_env.cfg.curriculum_total_steps)
        print(f"[INFO] start_k={args_cli.start_k:.4f}, global_steps={base_env.global_steps:,}")

    env = G1RMAFrameStackWrapper(base_env, log_dir=log_dir, n_stack=5)

    device = base_env.device
    num_envs = env.num_envs
    obs_dim = env.stacked_obs_dim
    priv_dim = env.priv_dim
    action_dim = env.action_space.shape[0]

    model = RMATeacherStudentActorCritic(
        obs_dim=obs_dim,
        priv_dim=priv_dim,
        action_dim=action_dim,
        latent_dim=int(args_cli.latent_dim),
        init_log_std=float(args_cli.init_log_std),
        min_log_std=float(args_cli.min_log_std),
        max_log_std=float(args_cli.max_log_std),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args_cli.lr))
    obs_norm = RunningMeanStd(shape=(obs_dim,), device=device, clip=10.0)
    priv_norm = RunningMeanStd(shape=(priv_dim,), device=device, clip=10.0)

    env_steps = try_resume(args_cli.resume, model, optimizer, obs_norm, priv_norm, device)

    if env_steps > 0:
        base_env.global_steps = env_steps

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / num_envs)
    update_env_steps = int(args_cli.rollouts) * num_envs
    total_updates = math.ceil(total_vector_steps / int(args_cli.rollouts))

    storage = RolloutStorage(
        rollouts=int(args_cli.rollouts),
        num_envs=num_envs,
        obs_dim=obs_dim,
        priv_dim=priv_dim,
        action_dim=action_dim,
        device=device,
    )

    print("\n[INFO] Task4 RMA 从零训练配置")
    print(f"  - num_envs              : {num_envs:,}")
    print(f"  - total_env_steps       : {total_env_steps:,}")
    print(f"  - total_updates         : {total_updates:,}")
    print(f"  - update_env_steps      : {update_env_steps:,}")
    print(f"  - policy_obs_dim        : {obs_dim}")
    print(f"  - privileged_obs_dim    : {priv_dim}")
    print(f"  - action_dim            : {action_dim}")
    print(f"  - lr/min/max            : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - init_log_std          : {args_cli.init_log_std}")
    print(f"  - teacher_bootstrap     : {args_cli.teacher_bootstrap_env_steps:,}")
    print(f"  - teacher_peak_ratio    : {args_cli.teacher_peak_ratio}")
    print(f"  - DR clean warmup       : first 15% curriculum")
    print(f"  - tensorboard           : tensorboard --logdir={args_cli.log_root}")
    print("\n🔥 [Task4 From-Scratch RMA PPO 已点火]")
    print("👉 不加载 Task2/Task3，避免继承归一化和动作接口问题。")
    print("👉 前 15% 课程无 DR，先学习 clean low-speed locomotion。")
    print("👉 Teacher 在 120M 步前不干预 actor，只后台学习 privileged latent。")
    print("👉 最终保存 full checkpoint 和 student-only 部署模型。\n")

    obs, priv, _ = env.reset()
    obs_norm.update(obs)
    priv_norm.update(priv)

    last_save_env_steps = env_steps
    update_id = 0
    start_time = time.time()

    try:
        with tqdm(
            total=total_env_steps,
            initial=env_steps,
            desc="G1 Task4 From-Scratch RMA PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            while env_steps < total_env_steps:
                model.eval()
                teacher_ratio = teacher_ratio_schedule(env_steps, args_cli)

                for t in range(int(args_cli.rollouts)):
                    obs_norm.update(obs)
                    priv_norm.update(priv)

                    obs_n = obs_norm.normalize(obs)
                    priv_n = priv_norm.normalize(priv)

                    with torch.no_grad():
                        action, raw_action, log_prob, value, _ = model.act(
                            obs_n, priv_n, teacher_ratio=teacher_ratio
                        )

                    next_obs, next_priv, reward, terminated, truncated, info = env.step(action)
                    done = terminated | truncated

                    storage.obs[t].copy_(obs_n)
                    storage.priv[t].copy_(priv_n)
                    storage.actions[t].copy_(action)
                    storage.raw_actions[t].copy_(raw_action)
                    storage.log_probs[t].copy_(log_prob)
                    storage.values[t].copy_(value)
                    storage.rewards[t].copy_(reward)
                    storage.dones[t].copy_(done.float())

                    obs, priv = next_obs, next_priv
                    env_steps += num_envs
                    pbar.update(num_envs)

                    if env_steps >= total_env_steps:
                        break

                with torch.no_grad():
                    obs_norm.update(obs)
                    priv_norm.update(priv)
                    obs_n = obs_norm.normalize(obs)
                    priv_n = priv_norm.normalize(priv)
                    _, last_value, _ = model.distribution(obs_n, priv_n, teacher_ratio=teacher_ratio)

                compute_gae(storage, last_value, float(args_cli.gamma), float(args_cli.gae_lambda))

                model.train()
                stats, stop_reason = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    storage=storage,
                    args=args_cli,
                    teacher_ratio=teacher_ratio,
                    writer=writer,
                    env_steps=env_steps,
                )

                update_id += 1
                elapsed = time.time() - start_time
                fps = env_steps / max(elapsed, 1e-6)

                flat = flat_dict(env.last_info)
                pbar.set_postfix(
                    {
                        "steps": f"{env_steps:,}",
                        "fps": f"{fps:,.0f}",
                        "rew": f"{env.last_reward_mean:+.3f}",
                        "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
                        "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
                        "dr": f"{flat.get('telemetry/DR_Scale', 0.0):.2f}",
                        "cmd": f"{flat.get('telemetry/Cmd_Vx', 0.0):+.2f}",
                        "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f}",
                        "kl": f"{stats.get('approx_kl', 0.0):.4f}",
                    }
                )

                writer.add_scalar("train/env_steps", env_steps, env_steps)
                writer.add_scalar("train/fps", fps, env_steps)
                writer.add_scalar("rma/teacher_ratio", teacher_ratio, env_steps)
                writer.add_scalar("rma/log_std_mean", model.log_std.detach().mean().item(), env_steps)
                writer.add_scalar(
                    "rma/std_mean",
                    torch.exp(torch.clamp(model.log_std, args_cli.min_log_std, args_cli.max_log_std)).mean().item(),
                    env_steps,
                )

                if update_id % max(int(args_cli.summary_interval), 1) == 0:
                    table_progress = {
                        "update": float(update_id),
                        "env_steps": float(env_steps),
                        "total_env_steps": float(total_env_steps),
                        "progress_percent": 100.0 * env_steps / max(total_env_steps, 1),
                        "fps": float(fps),
                        "learning_rate": float(stats.get("learning_rate", args_cli.lr)),
                        "teacher_ratio": float(teacher_ratio),
                        "stop_reason": stop_reason,
                    }

                    pbar.write(
                        "\n".join(
                            [
                                "\n" + "=" * 112,
                                f"📊 [G1 Task4 RMA PPO 更新 {update_id}] "
                                f"总步数: {env_steps:,} / {total_env_steps:,} | "
                                f"FPS: {fps:,.0f} | LR: {stats.get('learning_rate', args_cli.lr):.3e} | "
                                f"TeacherRatio: {teacher_ratio:.3f} | Stop: {stop_reason}",
                                "=" * 112,
                                make_table("time / progress", table_progress),
                                make_table("env info: reward + events + telemetry + debug", flat),
                                make_table("ppo / rma update info", stats),
                                "=" * 112 + "\n",
                            ]
                        )
                    )

                if env_steps - last_save_env_steps >= int(args_cli.save_freq_env_steps):
                    last_save_env_steps = env_steps
                    save_dir = os.path.join(log_dir, f"checkpoint_{env_steps}")
                    save_checkpoint(
                        save_dir,
                        model,
                        optimizer,
                        obs_norm,
                        priv_norm,
                        env_cfg,
                        args_cli,
                        env_steps,
                        extra={"last_stats": stats, "last_info": env.last_info, "stop_reason": stop_reason},
                    )
                    pbar.write(f"\n💾 [Task4 RMA 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到 Ctrl+C，正在保存最终模型...")
    except Exception:
        print("\n[ERROR] Task4 RMA 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, "final_checkpoint")
        try:
            save_checkpoint(
                final_dir,
                model,
                optimizer,
                obs_norm,
                priv_norm,
                env_cfg,
                args_cli,
                env_steps,
                extra={"final": True, "last_info": env.last_info},
            )
            print(f"✅ Task4 RMA 模型和 student-only 部署模型已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终模型失败: {type(exc).__name__}: {exc}")

        try:
            env.close()
        except Exception:
            pass
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ G1 Task4 From-Scratch RMA PPO 训练管线安全退出")


if __name__ == "__main__":
    main()
