from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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

parser = argparse.ArgumentParser(description="Evaluate Unitree G1 Task1 pure-RL skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.10)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK1_MOTION_FILE", ""))
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

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

from g1_rl.common.g1_skrl_models import G1Actor, G1Critic
from g1_rl.common.g1_skrl_wrappers import G1FrameStackWrapper
from g1_rl.common.info_utils import flat_dict, load_normalizers
from g1_rl.tasks.task1.task1_config import Task1Config
from g1_rl.tasks.task1.task1_env import G1Task1Env


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
    print("\n" + "=" * 150)
    print("G1 Task1 Pure-RL Model Test Summary")
    print("=" * 150)
    print(f"{'metric':<68} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12}")
    print("-" * 150)
    for key in sorted(summary):
        row = summary[key]
        print(
            f"{key:<68} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )
    print("=" * 150 + "\n")


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
            "directory": str(PROJECT_ROOT / "logs" / "task1_eval_tmp"),
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


def init_agent_compat(agent):
    """Keep Go2 verified style while supporting current skrl versions.

    Go2 model_test used:
        agent.init(trainer_cfg={"timesteps": 1, "headless": True})

    Some skrl builds expect trainer_cfg to be a dataclass, not a dict.
    In that case, evaluation does not need trainer_cfg, so fallback to agent.init().
    """
    try:
        agent.init(trainer_cfg={"timesteps": 1, "headless": True})
    except TypeError as exc:
        if "asdict" not in str(exc) and "dataclass" not in str(exc):
            raise
        print("[WARN] agent.init(trainer_cfg=dict) is not supported by this skrl build; fallback to agent.init().")
        agent.init()


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



def extract_policy_tensor(states):
    """Extract policy observation tensor from skrl IsaacLab wrapper output."""
    if isinstance(states, dict):
        if "policy" in states:
            return states["policy"]
        if "observations" in states:
            return states["observations"]
        if "states" in states:
            return states["states"]
        # Defensive fallback: use first tensor value in dict.
        for value in states.values():
            if torch.is_tensor(value):
                return value

    if torch.is_tensor(states):
        return states

    raise RuntimeError(f"Cannot extract policy tensor from states type={type(states)}")


@torch.no_grad()
def direct_policy_action(agent, states):
    """Safe deterministic evaluation action.

    Why this exists:
        Training uses skrl PPO normally.
        Go2 eval used agent.act successfully.
        In G1 eval, agent.act can block before returning on this skrl build.
        For model testing, we bypass skrl's high-level act() and call the loaded
        policy model directly. This keeps checkpoint loading and model weights
        unchanged, but avoids eval-only agent.act blocking.

    Output:
        action tensor in [-1, 1], finite.
    """
    obs = extract_policy_tensor(states)

    # Prefer loaded observation preprocessor if it is available and callable.
    # If a skrl version has incompatible preprocessor call signature, fallback to raw obs.
    prep = (
        getattr(agent, "_observation_preprocessor", None)
        or getattr(agent, "observation_preprocessor", None)
    )

    if prep is not None:
        try:
            obs = prep(obs, train=False)
        except TypeError:
            try:
                obs = prep(obs)
            except Exception:
                pass
        except Exception:
            pass

    obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    obs = torch.clamp(obs, -10.0, 10.0)

    policy = None
    try:
        policy = agent.models["policy"]
    except Exception:
        pass

    if policy is None:
        policy = getattr(agent, "policy", None)

    if policy is None:
        raise RuntimeError("Cannot find policy model from skrl agent.")

    # Our G1Actor.compute returns: mean, {"log_std": ...}
    out = policy.compute({"observations": obs, "states": obs}, role="policy")

    if isinstance(out, tuple):
        actions = out[0]
    else:
        actions = out

    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = torch.clamp(actions, -1.0, 1.0)

    return actions


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    candidates = [
        p / "g1_task1_model.pt",
        p / "agent.pt",
        p / "checkpoint.pt",
        p / "best_agent.pt",
        p / "final_checkpoint" / "g1_task1_model.pt",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    return p


def main():
    set_seed(int(args_cli.seed))

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.print_debug_info = False

    if args_cli.usd_path:
        cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        cfg.motion_file = str(args_cli.motion_file)

    base_env = G1Task1Env(cfg)

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * cfg.curriculum_total_steps)
        print(
            f"[INFO] Evaluation start_k={args_cli.start_k:.4f}, "
            f"global_steps={base_env.global_steps:,}"
        )

    stacked_env = G1FrameStackWrapper(
        base_env,
        log_dir=str(PROJECT_ROOT / "logs" / "task1_eval_tmp"),
        n_stack=5,
        tb_log_interval_steps=0,
        use_privileged_obs=False,
    )

    env = wrap_env(stacked_env, wrapper="isaaclab")

    print("\n[DEBUG] G1 Task1 Eval Spaces")
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

    states, _ = reset_env(env)

    records = []
    total_terminated = 0
    total_truncated = 0
    total_fall = 0
    total_timeout = 0

    start = time.time()

    print("\n" + "=" * 130)
    print("Unitree G1 Task1 pure-RL skrl model test started")
    print("=" * 130)
    print(f"checkpoint : {checkpoint}")
    print(f"num_envs   : {env.num_envs}")
    print(f"steps      : {args_cli.steps}")
    print(f"start_k    : {args_cli.start_k}")
    print(f"device     : {env.device}")
    print(f"usd_path   : {cfg.usd_path}")
    print(f"motion     : {cfg.motion_file}")
    print("note       : pure-RL baseline evaluation, not HoloSoma / BeyondMimic imitation pipeline")
    print("=" * 130 + "\n")

    try:
        with tqdm(total=int(args_cli.steps), desc="G1 Task1 Model Test", dynamic_ncols=True, mininterval=0.5) as pbar:
            for step in range(int(args_cli.steps)):
                if step < 3:
                    print(f"[DEBUG][eval step {step}] before direct_policy_action", flush=True)

                act_t0 = time.time()
                actions = direct_policy_action(agent, states)

                if step < 3:
                    print(
                        f"[DEBUG][eval step {step}] after direct_policy_action, "
                        f"dt={time.time() - act_t0:.4f}s, "
                        f"min={actions.min().item():+.4f}, max={actions.max().item():+.4f}",
                        flush=True,
                    )

                if step < 3:
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True)

                step_t0 = time.time()
                states, rewards, terminated, truncated, _ = step_env(env, actions)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after env.step, dt={time.time() - step_t0:.4f}s", flush=True)

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
                            "stage": f"{flat.get('telemetry/Curriculum_Stage', 0.0):.0f}",
                            "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f}",
                            "target": f"{flat.get('telemetry/Target_Vx', 0.0):+.2f}",
                            "h": f"{flat.get('telemetry/Base_Height', 0.0):.3f}",
                            "har": f"{flat.get('telemetry/Harness_Ratio', 0.0):.2f}",
                            "ct": f"{flat.get('telemetry/Contact_Count', 0.0):.2f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ G1 Task1 model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")
        print(f"  approx fall      : {total_fall:,}")
        print(f"  approx timeout   : {total_timeout:,}")

        print_table(summarize(records))

        print("G1 Task1 model test checklist:")
        print("1. Smoke checkpoint 表现差是正常的，重点检查是否稳定推理、无 NaN/Inf。")
        print("2. 这是 pure-RL baseline，不代表专业人形机器人动作控制最终路线。")
        print("3. 正式训练 checkpoint 应逐步看到 Base_Height 稳定、Fall_Rate 下降。")
        print("4. Harness_Ratio 下降后如果机器人快速摔倒，说明纯 RL gait 还没学稳。")
        print("5. Actual_Vx 接近 Target_Vx 之前，先确保站立和 contact 稳定。")

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
