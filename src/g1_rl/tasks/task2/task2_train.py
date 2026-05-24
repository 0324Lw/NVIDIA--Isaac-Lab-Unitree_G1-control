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

import torch
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
    description="Train Unitree G1 Task2 omni-directional pure-RL baseline with skrl PPO"
)

# Runtime
parser.add_argument("--total-env-steps", type=int, default=500_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)

# Assets
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK2_MOTION_FILE", ""))

# Warm-start / resume
parser.add_argument(
    "--pretrained-task1",
    "--pretrained",
    dest="pretrained_task1",
    type=str,
    default=os.environ.get("G1_TASK1_PRETRAINED", ""),
    help="Optional Task1 skrl checkpoint for warm-start. Example: logs/task1/<run>/final_checkpoint/g1_task1_model.pt",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Optional Task2 checkpoint file or checkpoint directory to resume.",
)

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", "--epochs", dest="learning_epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=1.0e-4)
parser.add_argument("--min-lr", type=float, default=7.0e-5)
parser.add_argument("--max-lr", type=float, default=2.0e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--kl-threshold", "--target-kl", dest="kl_threshold", type=float, default=0.015)
parser.add_argument("--entropy-coef", type=float, default=0.0025)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--ratio-clip", "--clip-range", dest="ratio_clip", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--init-log-std", type=float, default=-1.35)
parser.add_argument("--min-log-std", type=float, default=-5.0)
parser.add_argument("--max-log-std", type=float, default=0.20)

# Task2 conservative curriculum overrides
parser.add_argument("--stage0-vx-max", type=float, default=0.04)
parser.add_argument("--stage1-vx-min", type=float, default=-0.04)
parser.add_argument("--stage1-vx-max", type=float, default=0.16)
parser.add_argument("--stage1-no-yaw", action="store_true", default=True)
parser.add_argument("--cmd-smoothing-factor", type=float, default=0.08)

# Logging
parser.add_argument("--log-root", type=str, default=os.environ.get("RT_G1_TASK2_LOG_ROOT", default_log_root("task2")))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=10)
parser.add_argument("--tb-log-interval-steps", type=int, default=50)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

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

from g1_rl.common.g1_skrl_models import G1Actor, G1Critic
from g1_rl.common.g1_skrl_wrappers import G1FrameStackWrapper
from g1_rl.common.info_utils import (
    current_lr,
    flat_dict,
    load_normalizers,
    make_table,
    save_normalizers,
    to_float,
    tracking_mean,
    write_scalars,
)
from g1_rl.tasks.task2.task2_config import Task2Config
from g1_rl.tasks.task2.task2_env import G1OmniEnv


def make_log_dir() -> str:
    run_name = args_cli.run_name.strip()
    if not run_name:
        run_name = f"g1_task2_omni_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = os.path.abspath(os.path.join(args_cli.log_root, run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def task2_progress_postfix(env_steps: int, start_time: float, reward_mean: float, done_count: int, info: Dict[str, Any]):
    flat = flat_dict(info)
    fps = env_steps / max(time.time() - start_time, 1e-6)

    return {
        "steps": f"{env_steps:,}",
        "fps": f"{fps:,.0f}",
        "rew": f"{reward_mean:+.3f}",
        "done": int(done_count),
        "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
        "cmd": (
            f"{flat.get('telemetry/Cmd_Vx', 0.0):+.2f},"
            f"{flat.get('telemetry/Cmd_Vy', 0.0):+.2f},"
            f"{flat.get('telemetry/Cmd_Wz', 0.0):+.2f}"
        ),
        "vel": (
            f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f},"
            f"{flat.get('telemetry/Actual_Vy', 0.0):+.2f},"
            f"{flat.get('telemetry/Actual_Wz', 0.0):+.2f}"
        ),
        "linerr": f"{flat.get('telemetry/Lin_Error', 0.0):.3f}",
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
                f"📊 [G1 Task2 Omni pure-RL skrl PPO 更新 {update_id}] 总步数: {env_steps:,} / {total_steps:,} | "
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


def build_skrl_cfg(env, log_dir: str):
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
            "experiment_name": "g1_task2_omni_skrl",
            "write_interval": int(args_cli.skrl_write_interval),
            "checkpoint_interval": int(args_cli.skrl_checkpoint_interval),
            "store_separately": True,
            "wandb": False,
        }
    )

    return cfg


def _resolve_checkpoint_file(path: str, default_name: str) -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / default_name,
            p / "g1_task1_model.pt",
            p / "g1_task2_omni_model.pt",
            p / "agent.pt",
            p / "checkpoint.pt",
            p / "best_agent.pt",
            p / "final_checkpoint" / default_name,
            p / "final_checkpoint" / "g1_task1_model.pt",
            p / "final_checkpoint" / "g1_task2_omni_model.pt",
        ]

        for cand in candidates:
            if cand.exists():
                return str(cand)

    return str(p)


def try_load_pretrained(agent, models, pretrained_path: str, device: str) -> bool:
    if not pretrained_path:
        print("[INFO] 未指定 --pretrained-task1，将从随机初始化开始 Task2。")
        return False

    ckpt_path = _resolve_checkpoint_file(pretrained_path, default_name="g1_task1_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"[WARN] 找不到 Task1 预训练模型: {ckpt_path}")
        print("[WARN] 将从随机初始化开始 Task2。")
        return False

    print("\n" + "=" * 112)
    print(f"🔁 尝试加载 Task1 预训练模型: {ckpt_path}")
    print("=" * 112)

    try:
        agent.load(ckpt_path)
        print("✅ 已通过 agent.load() 成功加载 Task1 预训练模型")
        loaded_norm = load_normalizers(agent, str(Path(ckpt_path).parent))
        print(f"✅ normalizers loaded: {loaded_norm if loaded_norm else '<none>'}")
        return True
    except Exception as exc:
        print(f"⚠️ agent.load() 加载失败，尝试手动加载 state_dict: {type(exc).__name__}: {exc}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict) and "policy" in ckpt and "value" in ckpt:
            models["policy"].load_state_dict(ckpt["policy"], strict=False)
            models["value"].load_state_dict(ckpt["value"], strict=False)
            print("✅ 已从 ckpt['policy'] / ckpt['value'] 手动加载模型权重")
            return True

        if isinstance(ckpt, dict) and "models" in ckpt:
            loaded_any = False
            if "policy" in ckpt["models"]:
                models["policy"].load_state_dict(ckpt["models"]["policy"], strict=False)
                loaded_any = True
            if "value" in ckpt["models"]:
                models["value"].load_state_dict(ckpt["models"]["value"], strict=False)
                loaded_any = True
            if loaded_any:
                print("✅ 已从 ckpt['models'] 手动加载模型权重")
                return True

        if isinstance(ckpt, dict):
            missing, unexpected = models["policy"].load_state_dict(ckpt, strict=False)
            print("⚠️ 仅尝试作为 policy state_dict 加载")
            print(f"   missing={len(missing)}, unexpected={len(unexpected)}")
            return True

    except Exception as exc:
        print(f"[WARN] 手动加载 Task1 预训练模型失败: {type(exc).__name__}: {exc}")

    print("[WARN] Task1 预训练模型加载失败，将从随机初始化开始 Task2。")
    return False


def try_resume_task2(agent, resume_path: str) -> int:
    if not resume_path:
        return 0

    ckpt_path = _resolve_checkpoint_file(resume_path, default_name="g1_task2_omni_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"--resume 指定的 checkpoint 不存在: {ckpt_path}")

    print("\n" + "=" * 112)
    print(f"🔁 恢复 Task2 checkpoint: {ckpt_path}")
    print("=" * 112)

    agent.load(ckpt_path)
    loaded_norm = load_normalizers(agent, str(Path(ckpt_path).parent))
    print(f"✅ normalizers loaded: {loaded_norm if loaded_norm else '<none>'}")

    meta_path = Path(ckpt_path).parent / "train_metadata.pt"
    if meta_path.exists():
        try:
            meta = torch.load(str(meta_path), map_location="cpu")
            return int(meta.get("global_env_steps", 0))
        except Exception as exc:
            print(f"[WARN] 无法读取 resume metadata: {type(exc).__name__}: {exc}")

    return 0


def apply_task2_conservative_overrides(env_cfg: Task2Config) -> None:
    """Make Task2 early curriculum conservative.

    Reason:
        Task2 starts from a pure-RL Task1 baseline. Task1 may stand stably but may
        not be a mature humanoid walking policy. Task2 should not immediately ask
        for large omni-directional commands.
    """
    env_cfg.cmd_vx_stage0 = (0.00, float(args_cli.stage0_vx_max))
    env_cfg.cmd_vy_stage0 = (0.00, 0.00)
    env_cfg.cmd_wz_stage0 = (0.00, 0.00)

    env_cfg.cmd_vx_stage1 = (float(args_cli.stage1_vx_min), float(args_cli.stage1_vx_max))
    env_cfg.cmd_vy_stage1 = (0.00, 0.00)
    if bool(args_cli.stage1_no_yaw):
        env_cfg.cmd_wz_stage1 = (0.00, 0.00)

    env_cfg.cmd_smoothing_factor = float(args_cli.cmd_smoothing_factor)


def save_train_metadata(path, env_steps, total_env_steps, num_envs, base_env, stacked_env, pretrained_path, pretrained_loaded):
    torch.save(
        {
            "stage": "unitree_g1_task2_omni_locomotion_pure_rl_baseline",
            "algorithm": "skrl_ppo",
            "project_positioning": "educational pure-RL baseline, not HoloSoma / BeyondMimic imitation pipeline",
            "global_env_steps": int(env_steps),
            "total_env_steps_target": int(total_env_steps),
            "num_envs": int(num_envs),
            "single_obs_dim": int(base_env.cfg.num_observations),
            "stacked_obs_dim": int(stacked_env.observation_space.shape[0]),
            "num_actions": int(stacked_env.action_space.shape[0]),
            "frame_stack": int(stacked_env.n_stack),
            "motion_file": str(base_env.cfg.motion_file),
            "usd_path": str(base_env.cfg.usd_path),
            "pretrained_task1": str(pretrained_path),
            "pretrained_loaded": bool(pretrained_loaded),
            "controlled_joint_names": list(base_env.cfg.controlled_joint_names),
            "sensor_joint_names": list(base_env.cfg.sensor_joint_names),
            "foot_body_names": list(base_env.cfg.foot_body_names),
            "command_curriculum": {
                "stage0": [base_env.cfg.cmd_vx_stage0, base_env.cfg.cmd_vy_stage0, base_env.cfg.cmd_wz_stage0],
                "stage1": [base_env.cfg.cmd_vx_stage1, base_env.cfg.cmd_vy_stage1, base_env.cfg.cmd_wz_stage1],
                "stage2": [base_env.cfg.cmd_vx_stage2, base_env.cfg.cmd_vy_stage2, base_env.cfg.cmd_wz_stage2],
                "stage3": [base_env.cfg.cmd_vx_stage3, base_env.cfg.cmd_vy_stage3, base_env.cfg.cmd_wz_stage3],
                "stage4": [base_env.cfg.cmd_vx_stage4, base_env.cfg.cmd_vy_stage4, base_env.cfg.cmd_wz_stage4],
            },
        },
        os.path.join(path, "train_metadata.pt"),
    )


def save_agent_checkpoint(agent, save_dir: str, env_steps: int, total_env_steps: int, num_envs: int, base_env, stacked_env, pretrained_path: str, pretrained_loaded: bool):
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, "g1_task2_omni_model.pt"))
    save_normalizers(agent, save_dir)
    save_train_metadata(
        save_dir,
        env_steps=env_steps,
        total_env_steps=total_env_steps,
        num_envs=num_envs,
        base_env=base_env,
        stacked_env=stacked_env,
        pretrained_path=pretrained_path,
        pretrained_loaded=pretrained_loaded,
    )


def main():
    set_seed(int(args_cli.seed))

    log_dir = make_log_dir()

    print("\n" + "=" * 124)
    print("🚀 Unitree G1 Task2: Omni-Directional Command Locomotion Pure-RL skrl PPO 训练启动")
    print("=" * 124)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_dir      = {log_dir}")
    print(f"[INFO] device       = {args_cli.device}")
    print(f"[INFO] TF32 enabled = {getattr(torch.backends.cuda.matmul, 'allow_tf32', False)}")
    print("[INFO] 定位说明：这是 G1 人形机器人 pure-RL baseline，不等同于 HoloSoma / BeyondMimic 专业路线。")

    env_cfg = Task2Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.print_debug_info = False

    if args_cli.usd_path:
        env_cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        env_cfg.motion_file = str(args_cli.motion_file)

    apply_task2_conservative_overrides(env_cfg)

    base_env = G1OmniEnv(env_cfg)

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * base_env.cfg.curriculum_total_steps)
        print(
            f"[INFO] 已设置初始课程进度 start_k={args_cli.start_k:.4f}, "
            f"global_steps={base_env.global_steps:,}"
        )

    stacked_env = G1FrameStackWrapper(
        base_env,
        log_dir=log_dir,
        n_stack=5,
        tb_log_interval_steps=int(args_cli.tb_log_interval_steps),
        use_privileged_obs=False,
    )

    env = wrap_env(stacked_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", stacked_env.num_envs)

    print("\n[DEBUG] G1 Task2 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    if int(env.observation_space.shape[0]) != 615:
        raise RuntimeError(f"G1 Task2 policy input dim should be 615, got {env.observation_space.shape[0]}")
    if int(env.state_space.shape[0]) != 615:
        raise RuntimeError(f"G1 Task2 critic input dim should be 615, got {env.state_space.shape[0]}")
    if int(env.action_space.shape[0]) != 23:
        raise RuntimeError(f"G1 Task2 action dim should be 23, got {env.action_space.shape[0]}")

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

    print("\n[INFO] G1 Task2 训练配置")
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
    print(f"  - entropy_coef         : {args_cli.entropy_coef}")
    print(f"  - init_log_std         : {args_cli.init_log_std}")
    print(f"  - usd_path             : {env_cfg.usd_path}")
    print(f"  - motion_file          : {env_cfg.motion_file}")
    print(f"  - pretrained_task1     : {args_cli.pretrained_task1 if args_cli.pretrained_task1 else '<none>'}")
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

    pretrained_loaded = False
    resume_env_steps = 0

    if args_cli.resume:
        resume_env_steps = try_resume_task2(agent, args_cli.resume)
        base_env.global_steps = int(resume_env_steps)
        print(f"[INFO] 已恢复 Task2 global_env_steps={resume_env_steps:,}")
    else:
        pretrained_loaded = try_load_pretrained(agent, models, args_cli.pretrained_task1, env.device)

    trainer = StepTrainer(
        cfg={
            "timesteps": int(total_vector_steps),
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n🔥 [G1 Task2 Omni skrl PPO 已点火]")
    print("👉 训练目标：small vx -> forward/backward -> yaw -> lateral -> full omni。")
    print("👉 Policy/Critic 输入：123 维单帧观测 × 5 帧 = 615。")
    print("👉 动作维度：23，与 Task1 模型结构兼容。")
    print("👉 数据：g1_omni_walk.pt，要求 joint_names 与 USD 完全一致。")
    print("👉 当前仍然是 pure-RL baseline，不是 HoloSoma / BeyondMimic imitation pipeline。")
    print("👉 日志重点：Cmd_Vx/Vy/Wz、Actual_Vx/Vy/Wz、Lin_Error、Yaw_Error、Fall_Rate。\n")

    last_save = resume_env_steps
    update_id = 0
    start = time.time()

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=min(resume_env_steps, total_env_steps),
            desc="G1 Task2 Omni pure-RL skrl PPO",
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
                    task2_progress_postfix(
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
                        save_agent_checkpoint(
                            agent,
                            save_dir=save_dir,
                            env_steps=env_steps,
                            total_env_steps=total_env_steps,
                            num_envs=num_envs,
                            base_env=base_env,
                            stacked_env=stacked_env,
                            pretrained_path=args_cli.pretrained_task1,
                            pretrained_loaded=pretrained_loaded,
                        )
                        pbar.write(f"\n💾 [G1 Task2 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                if env_steps >= total_env_steps:
                    break

    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存...")
    except Exception:
        print("\n[ERROR] G1 Task2 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, "final_checkpoint")
        final_env_steps = min(resume_env_steps + total_vector_steps * num_envs, total_env_steps)

        try:
            save_agent_checkpoint(
                agent,
                save_dir=final_dir,
                env_steps=final_env_steps,
                total_env_steps=total_env_steps,
                num_envs=num_envs,
                base_env=base_env,
                stacked_env=stacked_env,
                pretrained_path=args_cli.pretrained_task1,
                pretrained_loaded=pretrained_loaded,
            )
            print(f"✅ G1 Task2 模型与归一化统计已保存至 {final_dir}")
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

        print("✅ G1 Task2 Omni skrl PPO 训练管线安全退出")


if __name__ == "__main__":
    main()
