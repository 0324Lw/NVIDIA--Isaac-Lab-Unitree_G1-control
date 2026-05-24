from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Unitree G1 Task3 Whole-Body pure-RL skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK3_MOTION_FILE", ""))
parser.add_argument("--visualize", action="store_true", help="Open Isaac Sim GUI for lightweight visualization")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(getattr(args_cli, "visualize", False))

simulation_app = AppLauncher(args_cli).app

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from g1_rl.common.g1_eval_utils import direct_policy_action, init_agent_compat
from g1_rl.common.g1_skrl_models import G1Actor, G1Critic
from g1_rl.common.g1_skrl_wrappers import G1FrameStackWrapper
from g1_rl.common.info_utils import flat_dict, load_normalizers
from g1_rl.tasks.task3.task3_config import Task3Config
from g1_rl.tasks.task3.task3_env import G1WholeBodyEnv


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
    print("\n" + "=" * 174)
    print("G1 Task3 Whole-Body Pure-RL Model Test Summary")
    print("=" * 174)
    print(f"{'metric':<82} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12}")
    print("-" * 174)

    for key in sorted(summary):
        row = summary[key]
        print(
            f"{key:<82} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 174 + "\n")


def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def build_agent(env):
    models = {
        "policy": G1Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=-1.35,
            min_log_std=-5.0,
            max_log_std=0.20,
        ),
        "value": G1Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": 1,
        "learning_epochs": 1,
        "mini_batches": 1,
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
    }

    for k, v in requested.items():
        if k in cfg:
            cfg[k] = v

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": str(PROJECT_ROOT / "logs" / "task3_eval_tmp"),
            "experiment_name": "eval",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": True,
            "wandb": False,
        }
    )

    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=env.device)

    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def step_env(env, actions):
    out = env.step(actions)
    if len(out) == 5:
        return out
    states, rewards, dones, infos = out
    return states, rewards, dones, dones, infos


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    candidates = [
        p / "g1_task3_whole_body_model.pt",
        p / "g1_task2_omni_model.pt",
        p / "g1_task1_model.pt",
        p / "agent.pt",
        p / "checkpoint.pt",
        p / "best_agent.pt",
        p / "final_checkpoint" / "g1_task3_whole_body_model.pt",
        p / "final_checkpoint" / "g1_task2_omni_model.pt",
        p / "final_checkpoint" / "g1_task1_model.pt",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    return p


def _unwrap_candidates(obj: Any) -> List[Any]:
    out = []

    for attr in ["env", "_env", "unwrapped", "venv", "gym_env", "raw_env", "base_env", "wrapped_env"]:
        try:
            value = getattr(obj, attr, None)
        except Exception:
            value = None

        if value is not None and value is not obj:
            out.append(value)

    return out


def _collect_env_chain(root: Any) -> List[Any]:
    seen: Set[int] = set()
    stack = [root]
    out = []

    while stack:
        obj = stack.pop(0)

        if obj is None:
            continue

        obj_id = id(obj)
        if obj_id in seen:
            continue

        seen.add(obj_id)
        out.append(obj)

        for child in _unwrap_candidates(obj):
            stack.append(child)

    return out


def force_eval_curriculum(env_like: Any, start_k: float = 1.0, label: str = "") -> int:
    try:
        k = float(start_k)
    except Exception:
        k = 1.0

    k = max(0.0, min(1.0, k))

    chain = _collect_env_chain(env_like)
    total_steps = 600_000_000

    for env in chain:
        cfg = getattr(env, "cfg", None)

        for obj in [env, cfg]:
            if obj is None:
                continue

            if hasattr(obj, "curriculum_total_steps"):
                try:
                    total_steps = max(total_steps, int(getattr(obj, "curriculum_total_steps")))
                except Exception:
                    pass

    target_steps = int(k * total_steps)
    changed = 0

    for env in chain:
        if hasattr(env, "global_steps"):
            try:
                setattr(env, "global_steps", target_steps)
                changed += 1
            except Exception:
                pass

    # Important: reset may sample command at old stage.
    # Resample commands after forcing final curriculum.
    for env in chain:
        if hasattr(env, "_resample_commands") and hasattr(env, "target_cmd"):
            try:
                ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
                env._resample_commands(ids)
                if hasattr(env, "smoothed_cmd"):
                    env.smoothed_cmd.copy_(env.target_cmd)
            except Exception as exc:
                print(f"[WARN] force_eval_curriculum command resample failed: {type(exc).__name__}: {exc}")

    prefix = f"[CURRICULUM][{label}]" if label else "[CURRICULUM]"
    print(
        f"{prefix} forced start_k={k:.4f}, target_steps={target_steps:,}, "
        f"total_steps={total_steps:,}, updated_fields={changed}",
        flush=True,
    )

    return target_steps


def main():
    set_seed(int(args_cli.seed))

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.print_debug_info = False

    if args_cli.usd_path:
        cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        cfg.motion_file = str(args_cli.motion_file)

    base_env = G1WholeBodyEnv(cfg)

    force_eval_curriculum(base_env, args_cli.start_k, label="after_env_creation")

    stacked_env = G1FrameStackWrapper(
        base_env,
        log_dir=str(PROJECT_ROOT / "logs" / "task3_eval_tmp"),
        n_stack=5,
        tb_log_interval_steps=0,
        use_privileged_obs=False,
    )

    env = wrap_env(stacked_env, wrapper="isaaclab")

    print("\n[DEBUG] G1 Task3 Eval Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    if int(env.observation_space.shape[0]) != 615:
        raise RuntimeError(f"G1 Task3 policy input dim should be 615, got {env.observation_space.shape[0]}")
    if int(env.state_space.shape[0]) != 615:
        raise RuntimeError(f"G1 Task3 critic input dim should be 615, got {env.state_space.shape[0]}")
    if int(env.action_space.shape[0]) != 23:
        raise RuntimeError(f"G1 Task3 action dim should be 23, got {env.action_space.shape[0]}")

    agent = build_agent(env)
    init_agent_compat(agent)

    checkpoint = resolve_checkpoint(args_cli.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    print(f"[INFO] loading checkpoint: {checkpoint}")
    agent.load(str(checkpoint))

    normalizer_dir = checkpoint.parent
    loaded = load_normalizers(agent, str(normalizer_dir))
    print(f"[INFO] loaded normalizers: {loaded if loaded else '<none>'}")

    try:
        agent.set_running_mode("eval")
    except Exception:
        pass

    force_eval_curriculum(base_env, args_cli.start_k, label="before_rollout_reset")
    states, _ = reset_env(env)
    force_eval_curriculum(base_env, args_cli.start_k, label="after_rollout_reset")

    records = []
    total_terminated = 0
    total_truncated = 0
    total_fall = 0
    total_timeout = 0

    start = time.time()

    print("\n" + "=" * 158)
    print("Unitree G1 Task3 Whole-Body pure-RL skrl model test started")
    print("=" * 158)
    print(f"[INFO] model_test requested start_k = {args_cli.start_k}")
    print(f"checkpoint : {checkpoint}")
    print(f"num_envs   : {env.num_envs}")
    print(f"steps      : {args_cli.steps}")
    print(f"start_k    : {args_cli.start_k}")
    print(f"device     : {env.device}")
    print(f"usd_path   : {cfg.usd_path}")
    print(f"motion     : {cfg.motion_file}")
    print("note       : pure-RL whole-body baseline evaluation, not HoloSoma / BeyondMimic imitation pipeline")
    print("=" * 158 + "\n")

    try:
        with tqdm(total=int(args_cli.steps), desc="G1 Task3 Whole-Body Model Test", dynamic_ncols=True, mininterval=0.5) as pbar:
            for step in range(int(args_cli.steps)):
                if step < 3:
                    print(f"[DEBUG][eval step {step}] before direct_policy_action", flush=True)

                actions = direct_policy_action(agent, states, debug=(step < 3), step=int(step))

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after direct_policy_action", flush=True)
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True)

                states, rewards, terminated, truncated, _ = step_env(env, actions)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after env.step", flush=True)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    flat = flat_dict(stacked_env.last_info)

                    row = {
                        "reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "terminated_rate": float(terminated.float().mean().cpu().item()),
                        "truncated_rate": float(truncated.float().mean().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    total_fall += int(round(flat.get("events/Fall_Rate", 0.0) * int(env.num_envs)))
                    total_timeout += int(round(flat.get("events/Timeout_Rate", 0.0) * int(env.num_envs)))

                    pbar.set_postfix(
                        {
                            "rew": f"{row['reward_mean']:+.3f}",
                            "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
                            "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
                            "arm": f"{flat.get('telemetry/Arm_Action_Gain', 0.0):.2f}",
                            "style": f"{flat.get('telemetry/Style_Scale', 0.0):.2f}",
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
                            "h": f"{flat.get('telemetry/Base_Height', 0.0):.3f}",
                            "armref": f"{flat.get('reward_components/R_Arm_Ref', 0.0):+.3f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ G1 Task3 model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")
        print(f"  approx fall      : {total_fall:,}")
        print(f"  approx timeout   : {total_timeout:,}")

        print_table(summarize(records))

        print("G1 Task3 model test checklist:")
        print("1. Smoke checkpoint 表现差是正常的，重点检查是否稳定推理、无 NaN/Inf。")
        print("2. 默认 start_k=1.0，用最终课程阶段测试，不再错误停留在 Stage 0。")
        print("3. 这是 pure-RL whole-body baseline，不代表专业人形机器人动作控制最终路线。")
        print("4. 重点看 Fall_Rate、Base_Height、Cmd/Actual 误差、Arm_Action_Gain、R_Arm_Ref、R_Arm_Leg_Sync。")
        print("5. 如果 agent.act 卡死问题复现，说明不应恢复 agent.act；本脚本已使用 direct_policy_action。")

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
