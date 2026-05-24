from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from isaaclab.app import AppLauncher
from g1_rl.common.paths import default_log_root


parser = argparse.ArgumentParser(
    description="Train Unitree G1 Task1 assisted locomotion pure-RL baseline with skrl PPO"
)

# Runtime
parser.add_argument("--total-env-steps", type=int, default=300_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)

# Assets
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK1_MOTION_FILE", ""))

# Resume
parser.add_argument("--resume", type=str, default="", help="Optional new skrl checkpoint file or checkpoint directory")

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", "--epochs", dest="learning_epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--min-lr", type=float, default=2e-5)
parser.add_argument("--max-lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--kl-threshold", "--target-kl", dest="kl_threshold", type=float, default=0.015)
parser.add_argument("--entropy-coef", type=float, default=0.002)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--ratio-clip", "--clip-range", dest="ratio_clip", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--init-log-std", type=float, default=-1.35)
parser.add_argument("--min-log-std", type=float, default=-5.0)
parser.add_argument("--max-log-std", type=float, default=0.20)

# Logging
parser.add_argument("--log-root", type=str, default=os.environ.get("RT_G1_TASK1_LOG_ROOT", default_log_root("task1")))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=10)
parser.add_argument("--tb-log-interval-steps", type=int, default=50)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

# AppLauncher owns --headless / --device / --experience.
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

try:
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except ImportError:
    from skrl.resources.schedulers.torch import KLAdaptiveRL as KLAdaptiveLR

from g1_rl.common.g1_skrl_models import G1Critic, G1Actor
from g1_rl.common.info_utils import (
    current_lr,
    flat_dict,
    make_table,
    save_normalizers,
    to_float,
    tracking_mean,
    write_scalars,
)
from g1_rl.tasks.task1.task1_config import Task1Config
from g1_rl.tasks.task1.task1_env import G1Task1Env


class G1Task1FrameStackWrapper(gym.Env):
    """5-frame stack wrapper for G1 Task1 skrl PPO.

    Raw env obs:
        123

    Policy / critic obs:
        123 * 5 = 615

    This remains a pure-RL baseline. The wrapper does not introduce imitation
    learning or privileged critic inputs.
    """

    def __init__(
        self,
        env: G1Task1Env,
        log_dir: str,
        n_stack: int = 5,
        tb_log_interval_steps: int = 50,
    ):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device
        self.tb_log_interval_steps = int(tb_log_interval_steps)

        self.single_dim = int(env.observation_space.shape[0])
        self.stacked_dim = self.single_dim * self.n_stack

        if self.single_dim != 123:
            raise RuntimeError(f"G1 Task1 single obs dim should be 123, got {self.single_dim}")
        if self.stacked_dim != 615:
            raise RuntimeError(f"G1 Task1 stacked obs dim should be 615, got {self.stacked_dim}")

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_dim,),
            dtype=np.float32,
        )

        self.state_space = self.observation_space

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.observation_space,
                "critic": self.state_space,
            }
        )

        self.action_space = env.action_space
        self.single_action_space = env.action_space

        self.obs_stack = torch.zeros(
            (self.num_envs, self.stacked_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.writer = SummaryWriter(log_dir) if self.tb_log_interval_steps != 0 else None

        self.global_env_steps = 0
        self.local_step_count = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def _pack(self):
        obs = torch.nan_to_num(
            torch.clamp(self.obs_stack, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        return {
            "policy": obs.clone(),
            "critic": obs.clone(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.obs_stack[:, i * self.single_dim : (i + 1) * self.single_dim] = obs

        self.last_info = info or {}
        return self._pack(), self.last_info

    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs_stack[:, :-self.single_dim] = self.obs_stack[:, self.single_dim :].clone()
        self.obs_stack[:, -self.single_dim :] = obs

        done = terminated | truncated

        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                self.obs_stack[
                    ids,
                    i * self.single_dim : (i + 1) * self.single_dim,
                ] = obs[ids]

        self.global_env_steps += self.num_envs
        self.local_step_count += 1

        self.last_info = info or {}
        self.last_reward_mean = to_float(reward) or 0.0
        self.last_done_count = int(done.sum().detach().cpu().item())

        if (
            self.writer is not None
            and self.tb_log_interval_steps > 0
            and self.local_step_count % self.tb_log_interval_steps == 0
        ):
            write_scalars(self.writer, self.last_info.get("reward_components", {}), self.global_env_steps, "rewards")
            write_scalars(self.writer, self.last_info.get("events", {}), self.global_env_steps, "events")
            write_scalars(self.writer, self.last_info.get("telemetry", {}), self.global_env_steps, "telemetry")
            write_scalars(self.writer, self.last_info.get("debug", {}), self.global_env_steps, "debug")
            self.writer.add_scalar("rollout/reward_mean_raw", self.last_reward_mean, self.global_env_steps)
            self.writer.add_scalar("rollout/done_count", self.last_done_count, self.global_env_steps)

        return self._pack(), reward, terminated, truncated, self.last_info

    def close(self):
        try:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass

        try:
            self.env.close()
        except Exception:
            pass


def make_log_dir() -> str:
    run_name = args_cli.run_name.strip()
    if not run_name:
        run_name = f"g1_task1_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = os.path.abspath(os.path.join(args_cli.log_root, run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def task1_progress_postfix(env_steps: int, start_time: float, reward_mean: float, done_count: int, info: Dict[str, Any]):
    flat = flat_dict(info)
    fps = env_steps / max(time.time() - start_time, 1e-6)

    return {
        "steps": f"{env_steps:,}",
        "fps": f"{fps:,.0f}",
        "rew": f"{reward_mean:+.3f}",
        "done": int(done_count),
        "stage": f"{flat.get('telemetry/Curriculum_Stage', 0.0):.0f}",
        "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f}/{flat.get('telemetry/Target_Vx', 0.0):+.2f}",
        "h": f"{flat.get('telemetry/Base_Height', 0.0):.2f}",
        "har": f"{flat.get('telemetry/Harness_Ratio', 0.0):.2f}",
        "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
    }


def print_update(pbar, update_id, env_steps, total_steps, elapsed, num_envs, rollouts, info, ppo, lr):
    stat = {
        "update": float(update_id),
        "total_env_steps": float(env_steps),
        "target_env_steps": float(total_steps),
        "progress_percent": 100.0 * env_steps / max(total_steps, 1),
        "num_envs": float(num_envs),
        "rollouts_per_update": float(rollouts),
        "fps_env_steps": env_steps / max(elapsed, 1e-6),
        "learning_rate": lr,
    }

    pbar.write(
        "\n".join(
            [
                "\n" + "=" * 124,
                f"📊 [G1 Task1 pure-RL skrl PPO 更新 {update_id}] 总步数: {env_steps:,} / {total_steps:,} | "
                f"环境 FPS: {stat['fps_env_steps']:,.0f} | LR: {lr:.3e}",
                "=" * 124,
                make_table("time / progress", stat),
                make_table("env info: reward_components + events + telemetry + debug", flat_dict(info)),
                make_table("ppo update info", ppo),
                "=" * 124 + "\n",
            ]
        )
    )


def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def _set_if_supported(cfg: dict, requested: dict) -> None:
    skipped = []

    for key, value in requested.items():
        if key in cfg:
            cfg[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"[WARN] 当前 skrl.PPO_CFG 不支持这些字段，已跳过: {skipped}")


def build_skrl_cfg(env, log_dir):
    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": int(args_cli.rollouts),
        "learning_epochs": int(args_cli.learning_epochs),
        "mini_batches": int(args_cli.mini_batches),
        "discount_factor": float(args_cli.gamma),
        "gae_lambda": float(args_cli.gae_lambda),
        "learning_rate": float(args_cli.lr),
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {
            "kl_threshold": float(args_cli.kl_threshold),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
        },
        "observation_preprocessor": RunningStandardScaler,
        "observation_preprocessor_kwargs": {
            "size": env.observation_space,
            "device": env.device,
        },
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {
            "size": env.state_space,
            "device": env.device,
        },
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {
            "size": 1,
            "device": env.device,
        },
        "grad_norm_clip": float(args_cli.grad_clip),
        "ratio_clip": float(args_cli.ratio_clip),
        "value_clip": float(args_cli.value_clip),
        "entropy_loss_scale": float(args_cli.entropy_coef),
        "value_loss_scale": float(args_cli.value_coef),
    }

    _set_if_supported(cfg, requested)

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": log_dir,
            "experiment_name": "g1_task1_skrl",
            "write_interval": int(args_cli.skrl_write_interval),
            "checkpoint_interval": int(args_cli.skrl_checkpoint_interval),
            "store_separately": True,
            "wandb": False,
        }
    )

    return cfg


def _resolve_checkpoint_file(path: str, default_name: str = "g1_task1_model.pt") -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / default_name,
            p / "agent.pt",
            p / "checkpoint.pt",
            p / "best_agent.pt",
            p / "final_checkpoint" / default_name,
        ]

        for cand in candidates:
            if cand.exists():
                return str(cand)

    return str(p)


def save_train_metadata(path, env_steps, num_envs, base_env, stacked_env):
    torch.save(
        {
            "stage": "unitree_g1_task1_assisted_locomotion_pure_rl_baseline",
            "algorithm": "skrl_ppo",
            "project_positioning": "educational pure-RL baseline, not a professional imitation-learning humanoid pipeline",
            "global_env_steps": int(env_steps),
            "num_envs": int(num_envs),
            "single_obs_dim": int(base_env.cfg.num_observations),
            "stacked_obs_dim": int(stacked_env.observation_space.shape[0]),
            "num_actions": int(stacked_env.action_space.shape[0]),
            "frame_stack": int(stacked_env.n_stack),
            "controlled_joint_names": list(base_env.cfg.controlled_joint_names),
            "sensor_joint_names": list(base_env.cfg.sensor_joint_names),
            "foot_body_names": list(base_env.cfg.foot_body_names),
            "usd_path": str(base_env.cfg.usd_path),
            "motion_file": str(base_env.cfg.motion_file),
        },
        os.path.join(path, "train_metadata.pt"),
    )


def save_agent_checkpoint(agent, save_dir: str, env_steps: int, num_envs: int, base_env, stacked_env):
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, "g1_task1_model.pt"))
    save_normalizers(agent, save_dir)
    save_train_metadata(save_dir, env_steps, num_envs, base_env, stacked_env)


def main():
    set_seed(int(args_cli.seed))

    log_dir = make_log_dir()

    print("\n" + "=" * 124)
    print("🚀 Unitree G1 Task1: Assisted Locomotion Pure-RL Baseline skrl PPO 训练启动")
    print("=" * 124)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_dir      = {log_dir}")
    print(f"[INFO] device       = {args_cli.device}")
    print(f"[INFO] TF32 enabled = {getattr(torch.backends.cuda.matmul, 'allow_tf32', False)}")
    print("[INFO] 定位说明：这是 G1 人形机器人纯 RL 学习版 baseline，不等同于 HoloSoma / BeyondMimic 专业路线。")

    env_cfg = Task1Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.print_debug_info = False

    if args_cli.usd_path:
        env_cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        env_cfg.motion_file = str(args_cli.motion_file)

    base_env = G1Task1Env(env_cfg)

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * base_env.cfg.curriculum_total_steps)
        print(
            f"[INFO] 已设置初始课程进度 start_k={args_cli.start_k:.4f}, "
            f"global_steps={base_env.global_steps:,}"
        )

    stacked_env = G1Task1FrameStackWrapper(
        base_env,
        log_dir=log_dir,
        n_stack=5,
        tb_log_interval_steps=int(args_cli.tb_log_interval_steps),
    )

    env = wrap_env(stacked_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", stacked_env.num_envs)

    print("\n[DEBUG] G1 Task1 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    if int(env.observation_space.shape[0]) != 615:
        raise RuntimeError(f"G1 Task1 policy input dim should be 615, got {env.observation_space.shape[0]}")
    if int(env.state_space.shape[0]) != 615:
        raise RuntimeError(f"G1 Task1 critic input dim should be 615, got {env.state_space.shape[0]}")
    if int(env.action_space.shape[0]) != 23:
        raise RuntimeError(f"G1 Task1 action dim should be 23, got {env.action_space.shape[0]}")

    models = {
        "policy": G1Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
        ),
        "value": G1Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / num_envs)
    save_freq_env_steps = int(args_cli.save_freq_env_steps)

    cfg = build_skrl_cfg(env, log_dir)
    update_env_steps = int(cfg["rollouts"]) * int(num_envs)

    print("\n[INFO] G1 Task1 训练配置")
    print(f"  - num_envs             : {num_envs:,}")
    print(f"  - total_env_steps      : {total_env_steps:,}")
    print(f"  - total_vector_steps   : {total_vector_steps:,}")
    print(f"  - rollouts             : {cfg['rollouts']}")
    print(f"  - learning_epochs      : {cfg['learning_epochs']}")
    print(f"  - mini_batches         : {cfg['mini_batches']}")
    print(f"  - update_env_steps     : {update_env_steps:,}")
    print(f"  - save_freq_env_steps  : {save_freq_env_steps:,}")
    print(f"  - single_obs_dim       : {base_env.cfg.num_observations}")
    print(f"  - stacked_obs_dim      : {env.observation_space.shape[0]}")
    print(f"  - action dim           : {env.action_space.shape[0]}")
    print(f"  - lr/min/max           : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - gamma                : {args_cli.gamma}")
    print(f"  - entropy_coef         : {args_cli.entropy_coef}")
    print(f"  - init_log_std         : {args_cli.init_log_std}")
    print(f"  - usd_path             : {env_cfg.usd_path}")
    print(f"  - motion_file          : {env_cfg.motion_file}")
    print(f"  - resume               : {args_cli.resume if args_cli.resume else '<none>'}")
    print(f"  - tensorboard          : tensorboard --logdir={args_cli.log_root}")

    memory = RandomMemory(memory_size=int(cfg["rollouts"]), num_envs=num_envs, device=env.device)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )

    resume_env_steps = 0
    if args_cli.resume:
        resume_path = _resolve_checkpoint_file(args_cli.resume, default_name="g1_task1_model.pt")
        print(f"[INFO] resume skrl checkpoint: {resume_path}")
        agent.load(resume_path)

        metadata_path = Path(resume_path).parent / "train_metadata.pt"
        if metadata_path.exists():
            try:
                meta = torch.load(str(metadata_path), map_location="cpu")
                resume_env_steps = int(meta.get("global_env_steps", 0))
                base_env.global_steps = resume_env_steps
                print(f"[INFO] restored global_env_steps from metadata: {resume_env_steps:,}")
            except Exception as exc:
                print(f"[WARN] metadata 恢复失败: {type(exc).__name__}: {exc}")

    trainer = StepTrainer(
        cfg={
            "timesteps": int(total_vector_steps),
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n🔥 [G1 Task1 skrl PPO 已点火]")
    print("👉 训练目标：站立稳定 -> 原地踏步 -> harness 辅助慢走 -> 降低 harness。")
    print("👉 Policy/Critic 输入：123 维单帧观测 × 5 帧 = 615。")
    print("👉 这是纯 RL baseline，后续 G1 专业路线应与 HoloSoma / OmniRetarget / BeyondMimic 区分。")
    print("👉 日志重点：Target_Vx / Actual_Vx / Base_Height / Fall_Rate / Harness_Ratio / Contact_Count。\n")

    last_save = resume_env_steps
    update_id = 0
    start = time.time()

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=min(resume_env_steps, total_env_steps),
            desc="G1 Task1 pure-RL skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                trainer.train(timestep=t, timesteps=total_vector_steps)

                env_steps = min(resume_env_steps + (t + 1) * num_envs, total_env_steps)
                previous_env_steps = min(resume_env_steps + t * num_envs, total_env_steps)
                pbar.update(env_steps - previous_env_steps)

                pbar.set_postfix(
                    task1_progress_postfix(
                        env_steps=env_steps,
                        start_time=start,
                        reward_mean=stacked_env.last_reward_mean,
                        done_count=stacked_env.last_done_count,
                        info=stacked_env.last_info,
                    )
                )

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1

                    ppo_info = tracking_mean(agent)
                    ppo_info["learning_rate"] = current_lr(agent)

                    writer = getattr(agent, "writer", None)
                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat_dict(stacked_env.last_info), env_steps, "env_info")

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        print_update(
                            pbar,
                            update_id,
                            env_steps,
                            total_env_steps,
                            time.time() - start,
                            num_envs,
                            cfg["rollouts"],
                            stacked_env.last_info,
                            ppo_info,
                            ppo_info["learning_rate"],
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                if env_steps - last_save >= save_freq_env_steps:
                    last_save = env_steps
                    save_dir = os.path.join(log_dir, f"checkpoint_{env_steps}")

                    try:
                        save_agent_checkpoint(agent, save_dir, env_steps, num_envs, base_env, stacked_env)
                        pbar.write(f"\n💾 [G1 Task1 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                if env_steps >= total_env_steps:
                    break

    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存...")
    except Exception:
        print("\n[ERROR] G1 Task1 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, "final_checkpoint")
        final_env_steps = min(resume_env_steps + total_vector_steps * num_envs, total_env_steps)

        try:
            save_agent_checkpoint(agent, final_dir, final_env_steps, num_envs, base_env, stacked_env)
            print(f"✅ G1 Task1 模型与归一化统计已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终模型失败: {type(exc).__name__}: {exc}")

        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ G1 Task1 skrl PPO 训练管线安全退出")


if __name__ == "__main__":
    main()
