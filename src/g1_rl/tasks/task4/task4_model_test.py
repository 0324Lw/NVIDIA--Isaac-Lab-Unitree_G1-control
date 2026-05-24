from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Unitree G1 Task4 RMA student policy")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK4_MOTION_FILE", ""))
parser.add_argument("--visualize", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(args_cli.visualize)
simulation_app = AppLauncher(args_cli).app

from g1_rl.tasks.task4.task4_config import Task4Config
from g1_rl.tasks.task4.task4_env import G1Sim2RealEnv


def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
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


def summarize(records: List[Dict[str, float]]):
    if not records:
        return {}
    keys = sorted({k for row in records for k in row.keys()})
    out = {}
    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def print_table(summary):
    print("\n" + "=" * 170)
    print("G1 Task4 RMA Student Model Test Summary")
    print("=" * 170)
    print(f"{'metric':<82} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12}")
    print("-" * 170)
    for key in sorted(summary):
        row = summary[key]
        print(
            f"{key:<82} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )
    print("=" * 170 + "\n")


class RunningMeanStd:
    def __init__(self, shape, device, eps: float = 1e-4, clip: float = 10.0):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(eps, dtype=torch.float32, device=device)
        self.clip = float(clip)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)

    def load_state_dict(self, state):
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count.copy_(state["count"].to(self.count.device))
        self.clip = float(state.get("clip", self.clip))


class RMAStudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
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

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.student_adapter(obs))
        mean = self.actor(torch.cat([obs, z], dim=-1))
        return torch.tanh(mean)


class FullRMAStudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
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
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.student_adapter(obs))
        mean = self.actor(torch.cat([obs, z], dim=-1))
        return torch.tanh(mean)


class EvalFrameStack:
    def __init__(self, env: G1Sim2RealEnv, n_stack: int = 5):
        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = env.cfg.num_envs
        self.device = env.device
        self.single_obs_dim = env.observation_space.shape[0]
        self.stacked_obs_dim = self.single_obs_dim * self.n_stack
        self.obs_stack = torch.zeros((self.num_envs, self.stacked_obs_dim), dtype=torch.float32, device=self.device)
        self.last_info = {}

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        for i in range(self.n_stack):
            self.obs_stack[:, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs
        self.last_info = info or {}
        return self.obs_stack.clone(), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.obs_stack[:, :-self.single_obs_dim] = self.obs_stack[:, self.single_obs_dim :].clone()
        self.obs_stack[:, -self.single_obs_dim :] = obs
        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                self.obs_stack[ids, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs[ids]
        self.last_info = info or {}
        return self.obs_stack.clone(), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def configure_eval_env(env_cfg: Task4Config) -> None:
    env_cfg.curriculum_total_steps = 700_000_000
    env_cfg.max_episode_length = 1200

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


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for name in [
            "g1_task4_student_deploy.pt",
            "g1_task4_rma_full_checkpoint.pt",
            "final_checkpoint/g1_task4_student_deploy.pt",
            "final_checkpoint/g1_task4_rma_full_checkpoint.pt",
        ]:
            cand = p / name
            if cand.exists():
                return cand
    return p


def load_policy(ckpt_path: Path, device: str, obs_dim: int, priv_dim: int, action_dim: int):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if "student_adapter" in ckpt and "actor" in ckpt:
        meta = ckpt.get("metadata", {})
        latent_dim = int(meta.get("latent_dim", 32))
        policy = RMAStudentPolicy(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim).to(device)
        policy.student_adapter.load_state_dict(ckpt["student_adapter"])
        policy.actor.load_state_dict(ckpt["actor"])
        obs_norm = RunningMeanStd(shape=(obs_dim,), device=device, clip=10.0)
        obs_norm.load_state_dict(ckpt["obs_norm"])
        policy.eval()
        return policy, obs_norm, "student_deploy"

    if "model" in ckpt:
        args = ckpt.get("args", {})
        latent_dim = int(args.get("latent_dim", 32))
        policy = FullRMAStudentPolicy(
            obs_dim=obs_dim,
            priv_dim=priv_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
        ).to(device)
        policy.load_state_dict(ckpt["model"], strict=False)
        obs_norm = RunningMeanStd(shape=(obs_dim,), device=device, clip=10.0)
        obs_norm.load_state_dict(ckpt["obs_norm"])
        policy.eval()
        return policy, obs_norm, "full_checkpoint_student_branch"

    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")


def force_eval_curriculum(env: G1Sim2RealEnv, start_k: float, label: str) -> None:
    k = max(0.0, min(1.0, float(start_k)))
    env.global_steps = int(k * env.cfg.curriculum_total_steps)
    ids = torch.arange(env.cfg.num_envs, dtype=torch.long, device=env.device)
    env._resample_commands(ids)
    print(
        f"[CURRICULUM][{label}] forced start_k={k:.4f}, "
        f"global_steps={env.global_steps:,}",
        flush=True,
    )


def main():
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)

    if args_cli.usd_path:
        cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        cfg.motion_file = str(args_cli.motion_file)

    configure_eval_env(cfg)

    base_env = G1Sim2RealEnv(cfg)
    install_slow_dr_curriculum(base_env)

    force_eval_curriculum(base_env, args_cli.start_k, "after_env_creation")

    env = EvalFrameStack(base_env, n_stack=5)
    obs, _ = env.reset(seed=args_cli.seed)

    force_eval_curriculum(base_env, args_cli.start_k, "after_rollout_reset")

    ckpt = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")

    obs_dim = env.stacked_obs_dim
    priv_dim = base_env.state_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    policy, obs_norm, mode = load_policy(
        ckpt_path=ckpt,
        device=base_env.device,
        obs_dim=obs_dim,
        priv_dim=priv_dim,
        action_dim=action_dim,
    )

    print("\n" + "=" * 150)
    print("Unitree G1 Task4 RMA student model test started")
    print("=" * 150)
    print(f"checkpoint : {ckpt}")
    print(f"load mode  : {mode}")
    print(f"num_envs   : {base_env.num_envs}")
    print(f"steps      : {args_cli.steps}")
    print(f"start_k    : {args_cli.start_k}")
    print(f"device     : {base_env.device}")
    print(f"usd_path   : {cfg.usd_path}")
    print(f"motion     : {cfg.motion_file}")
    print("note       : Task4 independent Sim2Real RMA baseline evaluation")
    print("=" * 150 + "\n")

    records = []
    total_terminated = 0
    total_truncated = 0
    start = time.time()

    try:
        with tqdm(total=int(args_cli.steps), desc="G1 Task4 RMA Model Test", dynamic_ncols=True, mininterval=0.5) as pbar:
            for step in range(int(args_cli.steps)):
                if step < 3:
                    print(f"[DEBUG][eval step {step}] before student policy", flush=True)

                with torch.no_grad():
                    obs_n = obs_norm.normalize(obs)
                    actions = policy.act(obs_n)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after student policy", flush=True)
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True)

                obs, rewards, terminated, truncated, info = env.step(actions)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after env.step", flush=True)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    flat = flat_dict(info)
                    row = {
                        "reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "terminated_rate": float(terminated.float().mean().cpu().item()),
                        "truncated_rate": float(truncated.float().mean().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    pbar.set_postfix(
                        {
                            "rew": f"{row['reward_mean']:+.3f}",
                            "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
                            "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
                            "dr": f"{flat.get('telemetry/DR_Scale', 0.0):.2f}",
                            "cmd": f"{flat.get('telemetry/Cmd_Vx', 0.0):+.2f}",
                            "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f}",
                            "h": f"{flat.get('telemetry/Base_Height', 0.0):.3f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(base_env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ G1 Task4 RMA model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_table(summarize(records))

        print("G1 Task4 model test checklist:")
        print("1. 本脚本只使用 student branch，不需要 teacher privileged obs。")
        print("2. 默认 start_k=1.0，测试最终 DR / command 阶段。")
        print("3. 如果 smoke checkpoint 表现差是正常的，重点先看能否稳定推理、无 NaN/Inf。")
        print("4. 正式效果重点看 Fall_Rate、Base_Height、Cmd/Actual、DR_Scale、Push_Active_Rate。")

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
