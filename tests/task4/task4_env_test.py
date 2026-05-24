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

parser = argparse.ArgumentParser(description="Unitree G1 Task4 Sim2Real Env Test")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--usd-path", type=str, default=os.environ.get("G1_USD_PATH", ""))
parser.add_argument("--motion-file", type=str, default=os.environ.get("G1_TASK4_MOTION_FILE", os.environ.get("G1_TASK2_MOTION_FILE", "")))
parser.add_argument("--collect-interval", type=int, default=40)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from g1_rl.tasks.task4.task4_config import Task4Config
from g1_rl.tasks.task4.task4_env import G1Sim2RealEnv


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
    print(" " * 55 + "G1 Task4 Sim2Real 环境统计报告")
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


def get_env_origins(env: G1Sim2RealEnv) -> torch.Tensor:
    if hasattr(env, "env_origins"):
        return env.env_origins
    return env.scene.env_origins


def quat_from_roll_pitch_yaw(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    device: str = "cuda:0",
) -> torch.Tensor:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy

    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def force_root_pose(
    env: G1Sim2RealEnv,
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
    heading("[测试 0] G1 Task4 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task4_sim2real.yaml",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task4" / "task4_config.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task4" / "task4_env.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task2" / "task2_env.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task1" / "task1_env.py",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task4 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("G1 Task4 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task4Config 基础配置检测")

    cfg = Task4Config()
    cfg.validate()

    assert cfg.num_actions == 23
    assert cfg.num_observations == 123
    assert cfg.num_privileged_obs == 162
    assert cfg.stacked_obs_dim == 615
    assert cfg.action_delay_steps_max >= 0
    assert cfg.obs_delay_steps_max >= 0
    assert cfg.curriculum_total_steps == 400_000_000

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
    print_ok(f"stacked_obs_dim = {cfg.stacked_obs_dim}")
    print_ok(f"curriculum_total_steps = {cfg.curriculum_total_steps:,}")
    print_ok(f"action_delay_steps_max = {cfg.action_delay_steps_max}")
    print_ok(f"obs_delay_steps_max = {cfg.obs_delay_steps_max}")
    print_ok("Task4Config 基础配置正常")


def check_assets(cfg: Task4Config) -> None:
    heading("[测试 2] G1 USD / Task4 motion 文件检查")

    print(f"usd_path    = {cfg.usd_path}")
    print(f"motion_file = {cfg.motion_file}")

    if not Path(cfg.usd_path).exists():
        raise FileNotFoundError(
            f"G1 USD 不存在: {cfg.usd_path}\n"
            "请设置 G1_USD_PATH 或修改 Task4Config.usd_path。"
        )

    if not Path(cfg.motion_file).exists():
        raise FileNotFoundError(
            f"G1 Task4 motion 文件不存在: {cfg.motion_file}\n"
            "Task4 默认复用 Task2 g1_omni_walk.pt。请设置 G1_TASK4_MOTION_FILE 或 G1_TASK2_MOTION_FILE。"
        )

    print_ok("G1 USD 文件存在")
    print_ok("Task4 motion 文件存在")


def check_obs_shape_and_values(env: G1Sim2RealEnv, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)

    assert torch.is_tensor(obs), f"obs 必须是 torch.Tensor，当前为 {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape 错误: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"obs 超出 clamp 范围: {obs.abs().max().item():.6f}"


def check_state_shape_and_values(env: G1Sim2RealEnv, state: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_privileged_obs)

    assert torch.is_tensor(state), f"state 必须是 torch.Tensor，当前为 {type(state)}"
    assert tuple(state.shape) == expected, f"state shape 错误: {tuple(state.shape)} != {expected}"
    assert_finite_tensor("state", state)
    assert state.abs().max().item() <= 20.0001, f"state 超出 clamp 范围: {state.abs().max().item():.6f}"


def check_observation_layout(env: G1Sim2RealEnv, obs: torch.Tensor) -> None:
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
        ("dr_scale", 1),
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
    assert torch.all(slices["dr_scale"] >= -1e-5)
    assert torch.all(slices["dr_scale"] <= 1.0 + 1e-5)

    h_mean = slices["root_height"].mean().item()
    assert 0.30 <= h_mean <= 1.20, f"root_height mean 异常: {h_mean:.4f}"


def check_curriculum_and_dr(env: G1Sim2RealEnv, cfg: Task4Config) -> None:
    heading("[测试 6] Task4 command / DR scale 课程检测")

    old_steps = int(env.global_steps)

    probe_ks = [0.00, 0.05, 0.08, 0.25, 0.50, 0.75, 1.00]
    records = []

    for k in probe_ks:
        env.global_steps = int(k * cfg.curriculum_total_steps)

        stage = env._command_stage()
        dr = env._dr_scale()
        vx_range, vy_range, wz_range = env._command_ranges()

        records.append(
            {
                "K": k,
                "Stage": stage,
                "DR": dr,
                "Vx_Min": vx_range[0],
                "Vx_Max": vx_range[1],
                "Vy_Min": vy_range[0],
                "Vy_Max": vy_range[1],
                "Wz_Min": wz_range[0],
                "Wz_Max": wz_range[1],
            }
        )

    env.global_steps = old_steps

    print(
        f"{'K':>6} | {'Stage':>5} | {'DR':>7} | {'Vx':>17} | {'Vy':>17} | {'Wz':>17}"
    )
    print("-" * 88)

    for row in records:
        print(
            f"{row['K']:>6.2f} | "
            f"{row['Stage']:>5} | "
            f"{row['DR']:>7.3f} | "
            f"{row['Vx_Min']:>7.3f}~{row['Vx_Max']:<7.3f} | "
            f"{row['Vy_Min']:>7.3f}~{row['Vy_Max']:<7.3f} | "
            f"{row['Wz_Min']:>7.3f}~{row['Wz_Max']:<7.3f}"
        )

    assert records[0]["Stage"] == 0
    assert records[-1]["Stage"] == 4
    assert records[0]["DR"] <= records[-1]["DR"]
    assert records[-1]["DR"] > 0.95
    assert records[-1]["Vx_Max"] >= records[0]["Vx_Max"]
    assert records[-1]["Wz_Max"] > records[0]["Wz_Max"]

    print_ok("Task4 command / DR scale 课程函数正常")


def check_dr_buffers(env: G1Sim2RealEnv, cfg: Task4Config) -> None:
    heading("[测试 7] Sim2Real DR buffer / privileged obs 检测")

    env.global_steps = int(cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    state = env.get_privileged_observations()

    check_obs_shape_and_values(env, obs)
    check_observation_layout(env, obs)
    check_state_shape_and_values(env, state)

    assert env.current_dr_scale.shape == (cfg.num_envs,)
    assert env.dr_motor_eff.shape == (cfg.num_envs, env.num_actions)
    assert env.dr_alpha_scale.shape == (cfg.num_envs, 1)
    assert env.dr_action_deadzone.shape == (cfg.num_envs, 1)
    assert env.dr_action_noise_std.shape == (cfg.num_envs, 1)
    assert env.action_delay_buffer.shape == (cfg.num_envs, cfg.action_delay_steps_max + 1, env.num_actions)
    assert env.obs_delay_buffer.shape == (cfg.num_envs, cfg.obs_delay_steps_max + 1, cfg.num_observations)

    assert torch.all(env.current_dr_scale >= -1e-5)
    assert torch.all(env.current_dr_scale <= 1.0 + 1e-5)
    assert torch.all(env.dr_motor_eff > 0.0)
    assert torch.all(env.dr_action_deadzone >= -1e-6)
    assert torch.all(env.dr_action_noise_std >= -1e-6)
    assert torch.all(env.action_delay_steps >= 0)
    assert torch.all(env.action_delay_steps <= cfg.action_delay_steps_max)
    assert torch.all(env.obs_delay_steps >= 0)
    assert torch.all(env.obs_delay_steps <= cfg.obs_delay_steps_max)

    print_ok(f"current_dr_scale mean = {env.current_dr_scale.mean().item():.4f}")
    print_ok(f"motor_eff range = {env.dr_motor_eff.min().item():.4f} ~ {env.dr_motor_eff.max().item():.4f}")
    print_ok(f"action_deadzone mean = {env.dr_action_deadzone.mean().item():.6f}")
    print_ok(f"action_noise_std mean = {env.dr_action_noise_std.mean().item():.6f}")
    print_ok(f"payload_mass mean = {env.dr_payload_mass.mean().item():.6f}")
    print_ok(f"friction_proxy range = {env.dr_friction.min().item():.4f} ~ {env.dr_friction.max().item():.4f}")
    print_ok(f"action_delay range = {int(env.action_delay_steps.min().item())} ~ {int(env.action_delay_steps.max().item())}")
    print_ok(f"obs_delay range = {int(env.obs_delay_steps.min().item())} ~ {int(env.obs_delay_steps.max().item())}")
    print_ok("privileged obs shape / finite / clamp 正常")


def check_action_chain(env: G1Sim2RealEnv, cfg: Task4Config) -> Dict[str, Any]:
    heading("[测试 8] action delay / deadzone / noise / motor efficiency 控制链路检测")

    env.global_steps = int(cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

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

    assert_finite_tensor("action chain rewards", rewards)
    check_obs_shape_and_values(env, obs)
    check_observation_layout(env, obs)

    flat = flatten_info(latest_info)

    required_info_keys = [
        "reward_components/R_Cmd_Lin",
        "reward_components/R_Cmd_Speed",
        "reward_components/R_Cmd_Yaw",
        "reward_components/R_Recovery",
        "reward_components/R_Push_Survival",
        "reward_components/P_Motor_Temp",
        "reward_components/Event_Fall",
        "events/Fall_Rate",
        "events/Timeout_Rate",
        "events/Push_Active_Rate",
        "telemetry/DR_Scale",
        "telemetry/Motor_Eff_Mean",
        "telemetry/Action_Deadzone",
        "telemetry/Action_Noise_Std",
        "telemetry/Payload_Mass",
        "telemetry/Friction_Proxy",
        "telemetry/Action_Delay",
        "telemetry/Obs_Delay",
        "debug/Obs_Dim",
        "debug/Privileged_Obs_Dim",
        "debug/Action_Dim",
    ]

    for key in required_info_keys:
        assert key in flat, f"info 缺少字段: {key}"

    print_ok(f"controlled joint 平均位移范数 = {q_delta:.6f}")
    print_ok(f"sensor joint 平均位移范数 = {sensor_delta:.8f}")
    print_ok("Task4 action latency / DR 控制链路正常")
    print_ok("Task4 reward/info 字段完整")

    return latest_info


def check_forced_events(env: G1Sim2RealEnv, cfg: Task4Config) -> None:
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
    heading("G1 Task4 Sim2Real Robustness 环境 / DR / privileged obs 全量测试启动")

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

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.print_debug_info = bool(args_cli.print_names)

    if args_cli.usd_path:
        cfg.usd_path = str(args_cli.usd_path)
    if args_cli.motion_file:
        cfg.motion_file = str(args_cli.motion_file)

    check_assets(cfg)

    env: G1Sim2RealEnv | None = None

    try:
        heading("[测试 3] 环境初始化 / 模型信息 / 名称映射检测")
        env = G1Sim2RealEnv(cfg)

        print_ok(f"device = {device}")
        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {env.num_actions}")
        print_ok(f"num_observations = {cfg.num_observations}")
        print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
        print_ok(f"robot_mass = {env.robot_mass:.3f} kg")

        assert env.robot.num_joints == 25
        assert env.num_actions == 23
        assert cfg.num_observations == 123
        assert cfg.num_privileged_obs == 162
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

        heading("[测试 4] reset / obs / privileged obs / spaces 检测")
        obs, info = env.reset(seed=args_cli.seed)

        check_obs_shape_and_values(env, obs)
        check_observation_layout(env, obs)

        state = env.get_privileged_observations()
        check_state_shape_and_values(env, state)

        assert env.observation_space.shape == (cfg.num_observations,)
        assert env.state_space.shape == (cfg.num_privileged_obs,)
        assert env.action_space.shape == (env.num_actions,)

        print_ok(f"observation_space = {env.observation_space}")
        print_ok(f"state_space = {env.state_space}")
        print_ok(f"action_space = {env.action_space}")
        print_ok(f"reset obs shape = {tuple(obs.shape)}")
        print_ok(f"privileged obs shape = {tuple(state.shape)}")
        print_ok(f"obs finite，范围 min={obs.min().item():.4f}, max={obs.max().item():.4f}")
        print_ok(f"state finite，范围 min={state.min().item():.4f}, max={state.max().item():.4f}")

        check_curriculum_and_dr(env, cfg)
        check_dr_buffers(env, cfg)
        check_action_chain(env, cfg)

        heading("[测试 9] 向量化 step 返回结构检测")
        rand_actions = torch.rand((cfg.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(rand_actions)

        check_obs_shape_and_values(env, obs)
        check_observation_layout(env, obs)

        state = env.get_privileged_observations()
        check_state_shape_and_values(env, state)

        assert rewards.shape == (cfg.num_envs,)
        assert terminated.shape == (cfg.num_envs,)
        assert truncated.shape == (cfg.num_envs,)
        assert_finite_tensor("rewards", rewards)

        print_ok(f"obs shape = {tuple(obs.shape)}")
        print_ok(f"state shape = {tuple(state.shape)}")
        print_ok(f"reward shape = {tuple(rewards.shape)}")
        print_ok(f"terminated shape = {tuple(terminated.shape)}")
        print_ok(f"truncated shape = {tuple(truncated.shape)}")
        print_ok(
            f"reward range: min={rewards.min().item():.4f}, "
            f"mean={rewards.mean().item():.4f}, max={rewards.max().item():.4f}"
        )

        check_forced_events(env, cfg)

        heading(f"[测试 11] 随机策略 rollout 稳定性检测：{args_cli.steps} 步")
        env.global_steps = int(0.85 * cfg.curriculum_total_steps)
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

                state = env.get_privileged_observations()
                check_state_shape_and_values(env, state)

                assert torch.isfinite(env.target_cmd).all()
                assert torch.isfinite(env.smoothed_cmd).all()
                assert torch.isfinite(env.current_dr_scale).all()
                assert torch.isfinite(env.dr_motor_eff).all()
                assert torch.isfinite(env.obs_delay_buffer).all()
                assert torch.isfinite(env.action_delay_buffer).all()

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
                    f"DR={flat.get('telemetry/DR_Scale', 0.0):.3f} | "
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
                    f"DelayA={flat.get('telemetry/Action_Delay', 0.0):.2f} | "
                    f"DelayO={flat.get('telemetry/Obs_Delay', 0.0):.2f} | "
                    f"Push={flat.get('events/Push_Active_Rate', 0.0):.3f}",
                    flush=True,
                )

        elapsed = time.time() - start_time
        fps = int(args_cli.steps) * int(cfg.num_envs) / max(elapsed, 1e-6)

        print_ok(f"随机策略 rollout 完成: {args_cli.steps} steps")
        print_ok(f"总 transitions: {int(args_cli.steps) * int(cfg.num_envs):,}")
        print_ok(f"吞吐约: {fps:,.2f} env steps/s")
        print_ok(f"累计 terminated: {total_falls:,}")
        print_ok(f"累计 truncated: {total_timeouts:,}")

        heading("[测试 12] 奖励组件 / DR 遥测 / 事件统计分析")
        print_summary_table(summarize_records(info_history))

        print("G1 Task4 Sim2Real training pre-check guide:")
        print("1. action_dim 应为 23，actor obs_dim 应为 123。")
        print("2. privileged obs_dim 应为 162，用于后续非对称 critic。")
        print("3. DR_Scale 应随课程从 0 逐步到 1。")
        print("4. action_delay / obs_delay / motor_eff / deadzone / payload / friction 都应有限。")
        print("5. 随机策略下 Fall_Rate 可以偏高，但不能出现 NaN/Inf。")
        print("6. 正式训练重点看 Fall_Rate、Cmd/Actual 误差、DR_Scale、Push_Active_Rate。")
        print("7. 这是 pure-RL Sim2Real robustness baseline，不是 HoloSoma / BeyondMimic。")

        heading("G1 Task4 Sim2Real Robustness 环境测试全部通过")

    except Exception as exc:
        print("\n❌ G1 Task4 Sim2Real 环境测试失败：")
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
