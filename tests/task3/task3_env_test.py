# Unitree G1 Task3 whole-body locomotion environment test.
#
# Usage:
#   cd /home/lw/unitree_g1_isaaclab_rl
#   bash scripts/ubuntu/test_task3_env.sh
#
# Important:
#   task3_env.py depends on IsaacLab / pxr-dependent modules through task2/task1.
#   Therefore AppLauncher must be launched before importing G1WholeBodyEnv.

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Unitree G1 Task3 Whole-Body Env Test")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK3_MOTION_FILE", ""))
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--collect-interval", type=int, default=40)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from g1_rl.tasks.task3.task3_config import Task3Config
from g1_rl.tasks.task3.task3_env import G1WholeBodyEnv


def heading(title: str) -> None:
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f" ⚠️ {msg}", flush=True)


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


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor, got {type(x)}"
    assert torch.isfinite(x).all(), f"{name} contains NaN/Inf"


def flatten_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (info or {}).items():
        if key == "terminal_observation":
            continue

        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_info(value, name))
        else:
            val = to_float(value)
            if val is not None and math.isfinite(val):
                out[name] = val

    return out


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({key for row in records for key in row.keys()})
    out: Dict[str, Dict[str, float]] = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    return out


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print_warn("没有收集到有效统计字段")
        return

    print("\n" + "=" * 184)
    print(" " * 58 + "G1 Task3 Whole-Body 环境统计报告")
    print("=" * 184)
    print(
        f"{'metric':<78} | {'mean':>12} | {'std':>12} | {'min':>12} | "
        f"{'p25':>12} | {'p50':>12} | {'p75':>12} | {'max':>12}"
    )
    print("-" * 184)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<78} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['p25']:>12.6f} | "
            f"{row['p50']:>12.6f} | "
            f"{row['p75']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 184 + "\n")


def get_env_origins(env: G1WholeBodyEnv) -> torch.Tensor:
    if hasattr(env, "env_origins"):
        return env.env_origins
    return env.scene.env_origins


def quat_from_roll_pitch_yaw(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    device: str = "cuda:0",
) -> torch.Tensor:
    # Isaac / USD quaternion order: wxyz
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy

    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def force_root_pose(
    env: G1WholeBodyEnv,
    env_ids: torch.Tensor,
    height: float | None = None,
    quat: torch.Tensor | None = None,
    zero_vel: bool = True,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    origins = get_env_origins(env)

    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:2] = origins[env_ids, 0:2]

    if height is None:
        root_state[:, 2] = origins[env_ids, 2] + float(env.cfg.target_height)
    else:
        root_state[:, 2] = origins[env_ids, 2] + float(height)

    if quat is None:
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    else:
        quat = torch.as_tensor(quat, dtype=torch.float32, device=env.device)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0).repeat(env_ids.numel(), 1)
        root_state[:, 3:7] = quat

    if zero_vel:
        root_state[:, 7:13] = 0.0

    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.update(dt=0.0)


def check_project_files() -> None:
    heading("[测试 0] G1 Task3 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task3_whole_body.yaml",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task3" / "task3_config.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task3" / "task3_env.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task2" / "task2_env.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task1" / "task1_env.py",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task3 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("G1 Task3 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task3Config 基础配置检测")

    cfg = Task3Config()
    cfg.validate()

    assert cfg.num_actions == 23
    assert cfg.num_observations == 123
    assert cfg.frame_stack == 5
    assert cfg.stacked_obs_dim == 615
    assert cfg.leg_action_scale > 0
    assert cfg.shoulder_action_scale > 0
    assert cfg.elbow_action_scale > 0
    assert cfg.wrist_action_scale > 0

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"stacked_obs_dim = {cfg.stacked_obs_dim}")
    print_ok(f"curriculum_total_steps = {cfg.curriculum_total_steps:,}")
    print_ok(f"leg_action_scale = {cfg.leg_action_scale}")
    print_ok(f"shoulder_action_scale = {cfg.shoulder_action_scale}")
    print_ok(f"elbow_action_scale = {cfg.elbow_action_scale}")
    print_ok(f"wrist_action_scale = {cfg.wrist_action_scale}")
    print_ok("Task3Config 基础配置正常")


def check_assets(cfg: Task3Config) -> None:
    heading("[测试 2] G1 USD / Task3 whole-body motion 文件检查")

    print(f"usd_path    = {cfg.usd_path}")
    print(f"motion_file = {cfg.motion_file}")

    if not Path(cfg.usd_path).exists():
        raise FileNotFoundError(
            f"G1 USD 不存在: {cfg.usd_path}\n"
            "请设置 G1_USD_PATH 或修改 Task3Config.usd_path。"
        )

    if not Path(cfg.motion_file).exists():
        raise FileNotFoundError(
            f"G1 Task3 whole-body motion 文件不存在: {cfg.motion_file}\n"
            "请设置 G1_TASK3_MOTION_FILE，或先生成 g1_whole_body_walk.pt。"
        )

    print_ok("G1 USD 文件存在")
    print_ok("G1 Task3 whole-body motion 文件存在")


def check_obs_shape_and_values(env: G1WholeBodyEnv, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)

    assert torch.is_tensor(obs), f"obs 必须是 torch.Tensor，当前为 {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape 错误: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"obs 超出 clamp 范围: {obs.abs().max().item():.6f}"


def check_observation_layout(env: G1WholeBodyEnv, obs: torch.Tensor) -> None:
    # Same 123-D layout as Task1 / Task2.
    layout = [
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("command", 3),
        ("q_err", 23),
        ("qd", 23),
        ("last_action", 23),
        ("action_delta", 23),
        ("foot_contact", 2),
        ("foot_rel_pos", 6),
        ("foot_vel_xy", 4),
        ("base_acc", 3),
        ("sin_phase", 1),
        ("cos_phase", 1),
        ("harness_ratio", 1),
        ("root_height", 1),
    ]

    cursor = 0
    slices = {}

    for name, dim in layout:
        slices[name] = obs[:, cursor : cursor + dim]
        cursor += dim

    assert cursor == env.cfg.num_observations, f"obs layout cursor={cursor}, expected={env.cfg.num_observations}"

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(slices["projected_gravity"].abs() <= 1.25)
    assert torch.all(slices["command"].abs() <= 10.0)
    assert torch.all(slices["last_action"].abs() <= 1.0001)
    assert torch.all(slices["action_delta"].abs() <= 2.0001)
    assert torch.all(slices["foot_contact"] >= -1e-5)
    assert torch.all(slices["foot_contact"] <= 1.0 + 1e-5)
    assert torch.all(slices["harness_ratio"] >= -1e-5)
    assert torch.all(slices["harness_ratio"] <= 1.0 + 1e-5)

    h_mean = slices["root_height"].mean().item()
    assert 0.30 <= h_mean <= 1.20, f"root_height mean 异常: {h_mean:.4f}"


def check_whole_body_motion_tensor(cfg: Task3Config, env: G1WholeBodyEnv) -> None:
    heading("[测试 5] g1_whole_body_walk.pt 全身参考库格式与 G1 模型匹配检测")

    data = torch.load(cfg.motion_file, map_location=cfg.device)

    required_keys = [
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

    for key in required_keys:
        assert key in data, f"motion 文件缺少字段: {key}"

    pos = data["pos"].to(cfg.device)
    vel = data["vel"].to(cfg.device)
    cmd = data["cmd"].to(cfg.device)
    phase = data["phase"].to(cfg.device)
    contact_ref = data["contact_ref"].to(cfg.device)
    mode_id = data["mode_id"].to(cfg.device)
    arm_swing_ref = data["arm_swing_ref"].to(cfg.device)

    joint_names = list(data["joint_names"])
    mode_names = list(data["mode_names"])
    num_frames = int(data["num_frames"])

    assert pos.shape == vel.shape, f"pos/vel shape 不一致: {tuple(pos.shape)} vs {tuple(vel.shape)}"
    assert pos.shape == (num_frames, env.robot.num_joints), (
        f"pos shape 应为 [T, {env.robot.num_joints}]，当前 {tuple(pos.shape)}"
    )
    assert cmd.shape == (num_frames, 3), f"cmd shape 应为 [T, 3]，当前 {tuple(cmd.shape)}"
    assert phase.shape == (num_frames,), f"phase shape 应为 [T]，当前 {tuple(phase.shape)}"
    assert contact_ref.shape == (num_frames, 2), f"contact_ref shape 应为 [T, 2]，当前 {tuple(contact_ref.shape)}"
    assert mode_id.shape == (num_frames,), f"mode_id shape 应为 [T]，当前 {tuple(mode_id.shape)}"
    assert arm_swing_ref.ndim == 2, f"arm_swing_ref 应为 2-D，当前 {tuple(arm_swing_ref.shape)}"
    assert arm_swing_ref.shape[0] == num_frames, "arm_swing_ref 第一维必须等于 num_frames"

    assert joint_names == env.robot_joint_names, "motion joint_names 与 G1 USD robot.joint_names 不一致"

    assert torch.isfinite(pos).all()
    assert torch.isfinite(vel).all()
    assert torch.isfinite(cmd).all()
    assert torch.isfinite(phase).all()
    assert torch.isfinite(contact_ref).all()
    assert torch.isfinite(mode_id.float()).all()
    assert torch.isfinite(arm_swing_ref).all()

    sensor_ids = [env.robot_joint_names.index(name) for name in cfg.sensor_joint_names]
    assert pos[:, sensor_ids].abs().max().item() < 1e-6, "sensor joint pos 应保持 0"
    assert vel[:, sensor_ids].abs().max().item() < 1e-6, "sensor joint vel 应保持 0"

    unique_cmd = torch.unique(cmd, dim=0)
    unique_mode = torch.unique(mode_id)

    assert unique_cmd.shape[0] >= 4, "Task3 whole-body 数据应至少包含多个 command 模式"
    assert len(mode_names) == int(unique_mode.numel()), "mode_names 数量应与 mode_id 唯一值数量匹配"

    print_ok(f"motion 文件存在: {cfg.motion_file}")
    print_ok(f"pos/vel shape: {tuple(pos.shape)}")
    print_ok(f"cmd shape: {tuple(cmd.shape)}")
    print_ok(f"phase shape: {tuple(phase.shape)}")
    print_ok(f"contact_ref shape: {tuple(contact_ref.shape)}")
    print_ok(f"mode_id shape: {tuple(mode_id.shape)}")
    print_ok(f"arm_swing_ref shape: {tuple(arm_swing_ref.shape)}")
    print_ok(f"unique commands: {unique_cmd.shape[0]}")
    print_ok(f"unique modes: {int(unique_mode.numel())}")
    print_ok("joint_names 与 G1 USD 完全匹配，支持 Task3 whole-body 训练")

    print("\nmode 分布:")
    for i, name in enumerate(mode_names):
        mask = mode_id == i
        if mask.any():
            mean_cmd = cmd[mask].mean(dim=0).detach().cpu().tolist()
            print(f"  {i:02d}: {str(name):<24} frames={int(mask.sum().item()):6d} cmd={mean_cmd}")


def check_task3_curriculum(env: G1WholeBodyEnv, cfg: Task3Config) -> None:
    heading("[测试 6] Task3 command / arm gain / style / harness 课程检测")

    old_steps = int(env.global_steps)

    probe_ks = [0.00, 0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 1.00]
    records = []

    for k in probe_ks:
        env.global_steps = int(k * cfg.curriculum_total_steps)

        stage = env._command_stage()
        vx_range, vy_range, wz_range = env._command_ranges()
        rsi_prob = env._rsi_probability()
        style_scale = env._style_weight_scale()
        ref_scale = env._reference_reset_scale()
        harness = env._harness_ratio()
        arm_gain = env._arm_action_gain()

        records.append(
            {
                "K": k,
                "Stage": stage,
                "Vx_Min": vx_range[0],
                "Vx_Max": vx_range[1],
                "Vy_Min": vy_range[0],
                "Vy_Max": vy_range[1],
                "Wz_Min": wz_range[0],
                "Wz_Max": wz_range[1],
                "RSI_Prob": rsi_prob,
                "StyleScale": style_scale,
                "RefResetScale": ref_scale,
                "Harness": harness,
                "ArmGain": arm_gain,
            }
        )

    env.global_steps = old_steps

    print(
        f"{'K':>6} | {'Stage':>5} | {'Vx':>17} | {'Vy':>17} | {'Wz':>17} | "
        f"{'RSI':>7} | {'Style':>7} | {'RefReset':>8} | {'Harness':>8} | {'ArmGain':>8}"
    )
    print("-" * 130)

    for row in records:
        print(
            f"{row['K']:>6.2f} | "
            f"{row['Stage']:>5} | "
            f"{row['Vx_Min']:>7.3f}~{row['Vx_Max']:<7.3f} | "
            f"{row['Vy_Min']:>7.3f}~{row['Vy_Max']:<7.3f} | "
            f"{row['Wz_Min']:>7.3f}~{row['Wz_Max']:<7.3f} | "
            f"{row['RSI_Prob']:>7.3f} | "
            f"{row['StyleScale']:>7.3f} | "
            f"{row['RefResetScale']:>8.3f} | "
            f"{row['Harness']:>8.3f} | "
            f"{row['ArmGain']:>8.3f}"
        )

    assert records[0]["Stage"] == 0, "K=0 应为 Stage 0"
    assert records[-1]["Stage"] == 4, "K=1 应为 Stage 4"
    assert records[-1]["Vx_Max"] > records[0]["Vx_Max"], "vx command 范围没有扩大"
    assert records[-1]["Vy_Max"] >= records[0]["Vy_Max"], "vy command 范围异常"
    assert records[-1]["Wz_Max"] > records[0]["Wz_Max"], "wz command 范围没有扩大"
    assert records[-1]["StyleScale"] > records[0]["StyleScale"], "StyleScale 没有打开"
    assert records[-1]["ArmGain"] > records[0]["ArmGain"], "ArmGain 没有随课程解冻"
    assert records[-1]["Harness"] <= records[0]["Harness"], "Harness 没有逐步关闭"

    print_ok("Task3 command / arm gain / style / harness 课程函数正常")


def check_action_scale_and_arm_gate(env: G1WholeBodyEnv) -> None:
    heading("[测试 7] action scale / arm action gate 检测")

    assert env.action_scale.shape == (env.num_actions,)
    assert env.ema_alpha_tensor.shape == (env.cfg.num_envs, env.num_actions)

    assert env.leg_action_ids_t.numel() > 0
    assert env.arm_action_ids_t.numel() > 0
    assert env.waist_action_ids_t.numel() > 0

    leg_scale_mean = env.action_scale[env.leg_action_ids_t].mean().item()
    arm_scale_mean = env.action_scale[env.arm_action_ids_t].mean().item()
    waist_scale_mean = env.action_scale[env.waist_action_ids_t].mean().item()

    assert leg_scale_mean > arm_scale_mean, "腿部 action_scale 应大于上肢 action_scale"
    assert waist_scale_mean <= leg_scale_mean, "腰部 action_scale 应不大于腿部"

    old_steps = int(env.global_steps)

    env.global_steps = 0
    early_gain = env._arm_action_gain()

    env.global_steps = int(env.cfg.curriculum_total_steps)
    final_gain = env._arm_action_gain()

    env.global_steps = old_steps

    assert abs(early_gain - 0.0) < 1e-6, f"早期 arm gain 应为 0，当前 {early_gain}"
    assert final_gain > 0.95, f"最终 arm gain 应接近 1，当前 {final_gain}"

    print_ok(f"leg_action_ids = {env.leg_action_ids_t.detach().cpu().tolist()}")
    print_ok(f"waist_action_ids = {env.waist_action_ids_t.detach().cpu().tolist()}")
    print_ok(f"arm_action_ids = {env.arm_action_ids_t.detach().cpu().tolist()}")
    print_ok(f"leg_scale_mean = {leg_scale_mean:.4f}")
    print_ok(f"waist_scale_mean = {waist_scale_mean:.4f}")
    print_ok(f"arm_scale_mean = {arm_scale_mean:.4f}")
    print_ok(f"early arm_gain = {early_gain:.4f}, final arm_gain = {final_gain:.4f}")


def check_forced_events(env: G1WholeBodyEnv, cfg: Task3Config) -> None:
    heading("[测试 10] 终局事件检测：摔倒 / 倾斜 / 超时截断")

    zero_actions = torch.zeros((cfg.num_envs, env.num_actions), dtype=torch.float32, device=env.device)

    env.reset()
    fall_ids = torch.arange(min(16, cfg.num_envs), device=env.device)
    force_root_pose(env, fall_ids, height=cfg.fall_height - 0.08)

    obs, rewards, terminated, truncated, info = env.step(zero_actions)
    fall_hit = int(terminated[fall_ids].sum().item())

    assert fall_hit > 0, "强制降低 root height 后没有触发摔倒 terminated"
    flat = flatten_info(info)

    print_ok(f"摔倒事件触发正常: {fall_hit}/{len(fall_ids)}")
    print_ok(f"Fall_Rate = {flat.get('events/Fall_Rate', 0.0):.6f}")
    print_ok(f"Event_Fall = {flat.get('reward_components/Event_Fall', 0.0):.6f}")

    env.reset()
    tilt_ids = torch.arange(min(16, cfg.num_envs), device=env.device)
    bad_quat = quat_from_roll_pitch_yaw(roll=1.35, pitch=0.0, yaw=0.0, device=env.device)
    force_root_pose(env, tilt_ids, height=cfg.target_height, quat=bad_quat)

    obs, rewards, terminated, truncated, info = env.step(zero_actions)
    tilt_hit = int(terminated[tilt_ids].sum().item())

    assert tilt_hit > 0, "强制大幅倾斜后没有触发 terminated"
    print_ok(f"倾斜事件触发正常: {tilt_hit}/{len(tilt_ids)}")

    env.reset()
    env.episode_steps[:] = cfg.max_episode_length - 1

    obs, rewards, terminated, truncated, info = env.step(zero_actions)
    timeout_count = int(truncated.sum().item())

    assert timeout_count > 0, "episode_steps 到达 max_episode_length 后没有触发 truncated"
    print_ok(f"超时截断触发正常: truncated={timeout_count}")


def run_tests() -> None:
    heading("G1 Task3 Whole-Body 环境 / 课程 / 奖励 / 全身参考库 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.num_envs = min(int(args_cli.num_envs), 4)
        args_cli.steps = min(int(args_cli.steps), 80)
        args_cli.collect_interval = min(int(args_cli.collect_interval), 20)

    check_project_files()
    check_config()

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.print_debug_info = bool(args_cli.print_names)

    if args_cli.usd_path:
        cfg.usd_path = str(args_cli.usd_path)

    if args_cli.motion_file:
        cfg.motion_file = str(args_cli.motion_file)

    check_assets(cfg)

    env: G1WholeBodyEnv | None = None

    try:
        heading("[测试 3] 环境初始化 / 模型信息 / 名称映射检测")
        env = G1WholeBodyEnv(cfg)

        print_ok(f"device = {device}")
        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {env.num_actions}")
        print_ok(f"num_observations = {cfg.num_observations}")
        print_ok(f"robot_mass = {env.robot_mass:.3f} kg")
        print_ok(f"motion frames = {env.motion.num_frames}")
        print_ok(f"motion modes = {env.motion.mode_names}")
        print_ok(f"arm_swing_ref dim = {env.motion.arm_swing_ref.shape[-1]}")

        assert env.robot.num_joints == 25, f"预期 G1 有 25 个关节，当前 {env.robot.num_joints}"
        assert env.num_actions == 23
        assert cfg.num_observations == 123
        assert len(env.sensor_joint_ids) == 2
        assert len(env.foot_body_ids) == 2
        assert len(env.contact_foot_ids) == 2

        if args_cli.print_names:
            print("\nrobot.joint_names:")
            for i, name in enumerate(env.robot_joint_names):
                print(f"  {i:02d}: {name}")

            print("\nrobot.body_names:")
            for i, name in enumerate(env.robot_body_names):
                print(f"  {i:02d}: {name}")

            print("\ncontact.body_names:")
            for i, name in enumerate(env.contact.body_names):
                print(f"  {i:02d}: {name}")

        heading("[测试 4] reset / obs / action space / command buffer 检测")
        obs, info = env.reset(seed=args_cli.seed)

        check_obs_shape_and_values(env, obs)
        check_observation_layout(env, obs)

        assert env.observation_space.shape == (cfg.num_observations,)
        assert env.action_space.shape == (env.num_actions,)
        assert env.target_cmd.shape == (cfg.num_envs, 3)
        assert env.smoothed_cmd.shape == (cfg.num_envs, 3)
        assert env.current_arm_gain.shape == (cfg.num_envs,)

        assert torch.isfinite(env.target_cmd).all()
        assert torch.isfinite(env.smoothed_cmd).all()
        assert torch.isfinite(env.current_arm_gain).all()

        print_ok(f"observation_space = {env.observation_space}")
        print_ok(f"action_space = {env.action_space}")
        print_ok(f"reset obs shape = {tuple(obs.shape)}")
        print_ok(f"target_cmd shape = {tuple(env.target_cmd.shape)}")
        print_ok(f"smoothed_cmd shape = {tuple(env.smoothed_cmd.shape)}")
        print_ok(f"current_arm_gain shape = {tuple(env.current_arm_gain.shape)}")
        print_ok(f"obs finite，范围 min={obs.min().item():.4f}, max={obs.max().item():.4f}")

        check_whole_body_motion_tensor(cfg, env)
        check_task3_curriculum(env, cfg)
        check_action_scale_and_arm_gate(env)

        heading("[测试 8] 随机动作控制链路检测：动作是否能驱动 G1 关节变化")
        env.reset()
        q0 = env.robot.data.joint_pos[:, env.controlled_joint_ids_t].clone()
        sensor_q0 = env.robot.data.joint_pos[:, env.sensor_joint_ids_t].clone()

        test_action = torch.rand((cfg.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0

        latest_info: Dict[str, Any] = {}

        for _ in range(20):
            obs, rewards, terminated, truncated, latest_info = env.step(test_action)

        q1 = env.robot.data.joint_pos[:, env.controlled_joint_ids_t].clone()
        sensor_q1 = env.robot.data.joint_pos[:, env.sensor_joint_ids_t].clone()

        q_delta = torch.norm(q1 - q0, dim=-1).mean().item()
        sensor_delta = torch.norm(sensor_q1 - sensor_q0, dim=-1).mean().item()

        assert q_delta > 1e-5, "控制动作没有引起关节明显变化"
        assert sensor_delta < 5e-3, f"sensor joints moved too much: {sensor_delta:.6f}"
        assert_finite_tensor("control rewards", rewards)
        check_obs_shape_and_values(env, obs)
        check_observation_layout(env, obs)

        flat = flatten_info(latest_info)

        required_info_keys = [
            "reward_components/R_Cmd_Lin",
            "reward_components/R_Cmd_Speed",
            "reward_components/R_Cmd_Yaw",
            "reward_components/R_Arm_Ref",
            "reward_components/R_Arm_Vel_Ref",
            "reward_components/R_Arm_Leg_Sync",
            "reward_components/R_Arm_Cross",
            "reward_components/Event_Fall",
            "events/Fall_Rate",
            "events/Timeout_Rate",
            "telemetry/Command_Stage",
            "telemetry/Cmd_Vx",
            "telemetry/Cmd_Vy",
            "telemetry/Cmd_Wz",
            "telemetry/Actual_Vx",
            "telemetry/Actual_Vy",
            "telemetry/Actual_Wz",
            "telemetry/Arm_Action_Gain",
            "telemetry/Style_Scale",
            "telemetry/Arm_Ref_Raw",
            "telemetry/Arm_Leg_Sync_Raw",
            "debug/Obs_Dim",
            "debug/Action_Dim",
            "debug/Motion_Arm_Ref_Dim",
        ]

        for key in required_info_keys:
            assert key in flat, f"info 缺少字段: {key}"

        print_ok(f"controlled joint 平均位移范数 = {q_delta:.6f}")
        print_ok(f"sensor joint 平均位移范数 = {sensor_delta:.8f}")
        print_ok("随机动作控制链路正常")
        print_ok("Task3 whole-body reward/info 字段完整")

        heading("[测试 9] 向量化 step 返回结构检测")
        rand_actions = torch.rand((cfg.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(rand_actions)

        check_obs_shape_and_values(env, obs)
        check_observation_layout(env, obs)

        assert rewards.shape == (cfg.num_envs,)
        assert terminated.shape == (cfg.num_envs,)
        assert truncated.shape == (cfg.num_envs,)
        assert_finite_tensor("rewards", rewards)

        print_ok(f"obs shape = {tuple(obs.shape)}")
        print_ok(f"reward shape = {tuple(rewards.shape)}")
        print_ok(f"terminated shape = {tuple(terminated.shape)}")
        print_ok(f"truncated shape = {tuple(truncated.shape)}")
        print_ok(
            f"reward range: min={rewards.min().item():.4f}, "
            f"mean={rewards.mean().item():.4f}, max={rewards.max().item():.4f}"
        )

        check_forced_events(env, cfg)

        heading(f"[测试 11] 随机策略 rollout 稳定性检测：{args_cli.steps} 步")
        env.global_steps = int(0.75 * cfg.curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        info_history: List[Dict[str, float]] = []
        total_falls = 0
        total_timeouts = 0
        start_time = time.time()

        for step in range(int(args_cli.steps)):
            actions = torch.rand((cfg.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0
            obs, rewards, terminated, truncated, info = env.step(actions)

            total_falls += int(terminated.sum().item())
            total_timeouts += int(truncated.sum().item())

            if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                check_obs_shape_and_values(env, obs)
                check_observation_layout(env, obs)
                assert_finite_tensor("rollout rewards", rewards)
                assert torch.isfinite(env.target_cmd).all()
                assert torch.isfinite(env.smoothed_cmd).all()
                assert torch.isfinite(env.current_arm_gain).all()

                flat = flatten_info(info)
                flat["test/step"] = float(step)
                flat["test/reward_mean"] = float(rewards.mean().item())
                flat["test/reward_min"] = float(rewards.min().item())
                flat["test/reward_max"] = float(rewards.max().item())
                flat["test/terminated_count"] = float(terminated.sum().item())
                flat["test/truncated_count"] = float(truncated.sum().item())
                info_history.append(flat)

                print(
                    f"step={step + 1:>5}/{args_cli.steps} | "
                    f"Reward={flat.get('test/reward_mean', 0.0):+8.4f} | "
                    f"Fall={flat.get('events/Fall_Rate', 0.0):6.3f} | "
                    f"Stage={flat.get('telemetry/Command_Stage', 0.0):4.1f} | "
                    f"Cmd=({flat.get('telemetry/Cmd_Vx', 0.0):+.3f}, "
                    f"{flat.get('telemetry/Cmd_Vy', 0.0):+.3f}, "
                    f"{flat.get('telemetry/Cmd_Wz', 0.0):+.3f}) | "
                    f"Vel=({flat.get('telemetry/Actual_Vx', 0.0):+.3f}, "
                    f"{flat.get('telemetry/Actual_Vy', 0.0):+.3f}, "
                    f"{flat.get('telemetry/Actual_Wz', 0.0):+.3f}) | "
                    f"LinErr={flat.get('telemetry/Lin_Error', 0.0):.4f} | "
                    f"YawErr={flat.get('telemetry/Yaw_Error', 0.0):.4f} | "
                    f"H={flat.get('telemetry/Base_Height', 0.0):.3f} | "
                    f"Contact={flat.get('telemetry/Contact_Count', 0.0):.3f} | "
                    f"ArmGain={flat.get('telemetry/Arm_Action_Gain', 0.0):.3f} | "
                    f"Style={flat.get('telemetry/Style_Scale', 0.0):.3f} | "
                    f"R_Arm={flat.get('reward_components/R_Arm_Ref', 0.0):+.3f} | "
                    f"Sync={flat.get('reward_components/R_Arm_Leg_Sync', 0.0):+.3f}",
                    flush=True,
                )

        elapsed = time.time() - start_time
        fps = int(args_cli.steps) * int(cfg.num_envs) / max(elapsed, 1e-6)

        print_ok(f"随机策略 rollout 完成: {args_cli.steps} steps")
        print_ok(f"总 transitions: {int(args_cli.steps) * int(cfg.num_envs):,}")
        print_ok(f"吞吐约: {fps:,.2f} env steps/s")
        print_ok(f"累计 terminated: {total_falls:,}")
        print_ok(f"累计 truncated: {total_timeouts:,}")

        heading("[测试 12] 奖励组件 / 遥测 / 事件统计分析")
        print_summary_table(summarize_records(info_history))

        print("G1 Task3 whole-body training pre-check guide:")
        print("1. action_dim 应为 23，obs_dim 应为 123，才能和 Task1/Task2 架构兼容。")
        print("2. g1_whole_body_walk.pt 必须包含 arm_swing_ref。")
        print("3. joint_names 必须与 G1 USD robot.joint_names 完全一致。")
        print("4. 早期 Arm_Action_Gain 应接近 0，后期应逐步接近 1。")
        print("5. 随机策略下 Fall_Rate 可以偏高，但不能出现 NaN/Inf。")
        print("6. 训练时重点看 Cmd/Actual 误差、Fall_Rate、Arm_Ref、Arm_Leg_Sync。")
        print("7. 这是 pure-RL whole-body baseline，不等同于 HoloSoma / BeyondMimic 专业路线。")

        heading("G1 Task3 Whole-Body 环境测试全部通过")

    except Exception as exc:
        print("\n❌ G1 Task3 Whole-Body 环境测试失败：")
        print(type(exc).__name__, ":", exc)
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
