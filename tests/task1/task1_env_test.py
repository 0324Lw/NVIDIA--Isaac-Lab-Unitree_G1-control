# Unitree G1 Task1 assisted locomotion environment test.
#
# Usage:
#   cd /home/lw/unitree_g1_isaaclab_rl
#   bash scripts/ubuntu/test_task1_env.sh
#
# Important:
#   task1_env.py imports IsaacLab / pxr-dependent modules.
#   Therefore AppLauncher must be launched before importing G1Task1Env.

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


parser = argparse.ArgumentParser(description="Unitree G1 Task1 Assisted Locomotion Env Test")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--rollout-k", type=float, default=0.10)
parser.add_argument("--collect-interval", type=int, default=40)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from g1_rl.tasks.task1.task1_config import Task1Config
from g1_rl.tasks.task1.task1_env import G1Task1Env


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f" ⚠️ {msg}", flush=True)


def heading(title: str) -> None:
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor, got {type(x)}"
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def tensor_to_float(x: Any):
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


def flatten_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (info or {}).items():
        if key == "terminal_observation":
            continue

        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_info(value, name))
        else:
            val = tensor_to_float(value)
            if val is not None and math.isfinite(val):
                out[name] = val

    return out


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({k for row in records for k in row.keys()})
    summary: Dict[str, Dict[str, float]] = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return summary


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print_warn("No valid records collected.")
        return

    print("\n" + "=" * 168)
    print(" " * 54 + "Unitree G1 Task1 Environment Statistics")
    print("=" * 168)
    print(f"{'metric':<76} | {'mean':>14} | {'std':>14} | {'min':>14} | {'max':>14}")
    print("-" * 168)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<76} | "
            f"{row['mean']:>14.6f} | "
            f"{row['std']:>14.6f} | "
            f"{row['min']:>14.6f} | "
            f"{row['max']:>14.6f}"
        )

    print("=" * 168 + "\n")


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
    z = cr * cp * sy - sr * sp * cy

    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def check_project_files() -> None:
    heading("[测试 0] G1 Task1 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task1_assisted_locomotion.yaml",
        PROJECT_ROOT / "src" / "g1_rl" / "common" / "paths.py",
        PROJECT_ROOT / "src" / "g1_rl" / "common" / "info_utils.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task1" / "task1_config.py",
        PROJECT_ROOT / "src" / "g1_rl" / "tasks" / "task1" / "task1_env.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required G1 Task1 files:\n" + "\n".join(missing)

    for p in required:
        print_ok(str(p.relative_to(PROJECT_ROOT)))

    print_ok("G1 Task1 工程文件结构正常")


def check_assets(cfg: Task1Config) -> None:
    heading("[测试 1] G1 USD / motion file 资源检查")

    print(f"usd_path    = {cfg.usd_path}")
    print(f"motion_file = {cfg.motion_file}")

    if not Path(cfg.usd_path).exists():
        raise FileNotFoundError(
            f"G1 USD 不存在: {cfg.usd_path}\n"
            "请设置环境变量 G1_USD_PATH，或修改 Task1Config.usd_path。\n"
            "示例：export G1_USD_PATH=/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd"
        )

    if not Path(cfg.motion_file).exists():
        raise FileNotFoundError(
            f"G1 motion 文件不存在: {cfg.motion_file}\n"
            "请设置环境变量 G1_TASK1_MOTION_FILE，或修改 Task1Config.motion_file。\n"
            "示例：export G1_TASK1_MOTION_FILE=/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt"
        )

    print_ok("G1 USD 文件存在")
    print_ok("G1 motion 文件存在")


def check_obs_shape_and_values(env: G1Task1Env, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)

    assert torch.is_tensor(obs), f"obs must be torch.Tensor, got {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape wrong: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"obs out of clamp range: {obs.abs().max().item():.6f}"


def check_observation_slices(env: G1Task1Env, obs: torch.Tensor) -> None:
    cursor = 0
    slices: Dict[str, torch.Tensor] = {}

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

    for name, dim in layout:
        slices[name] = obs[:, cursor:cursor + dim]
        cursor += dim

    assert cursor == env.cfg.num_observations, f"obs slice cursor={cursor}, expected={env.cfg.num_observations}"

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(slices["projected_gravity"].abs() <= 1.25)
    assert torch.all(slices["last_action"].abs() <= 1.0001)
    assert torch.all(slices["action_delta"].abs() <= 2.0001)
    assert torch.all(slices["foot_contact"] >= -1e-5)
    assert torch.all(slices["foot_contact"] <= 1.0 + 1e-5)
    assert torch.all(slices["sin_phase"].abs() <= 1.0001)
    assert torch.all(slices["cos_phase"].abs() <= 1.0001)
    assert torch.all(slices["harness_ratio"] >= -1e-5)
    assert torch.all(slices["harness_ratio"] <= 1.0 + 1e-5)

    height_mean = slices["root_height"].mean().item()
    assert 0.30 <= height_mean <= 1.20, f"root_height mean abnormal: {height_mean:.4f}"


def force_root_pose(
    env: G1Task1Env,
    env_ids: torch.Tensor,
    height: float | None = None,
    quat: torch.Tensor | None = None,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    if env_ids.numel() == 0:
        return

    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:2] = env.env_origins[env_ids, 0:2]

    if height is None:
        root_state[:, 2] = float(env.cfg.target_height)
    else:
        root_state[:, 2] = float(height)

    if quat is None:
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    else:
        quat = torch.as_tensor(quat, dtype=torch.float32, device=env.device)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0).repeat(env_ids.numel(), 1)
        root_state[:, 3:7] = quat

    root_state[:, 7:13] = 0.0

    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.update(dt=0.0)


def check_config() -> None:
    heading("[测试 2] Task1Config 基础配置检测")

    cfg = Task1Config()
    cfg.validate()

    assert cfg.num_actions == 23
    assert cfg.num_observations == 123
    assert cfg.frame_stack == 5
    assert cfg.stacked_obs_dim == 615
    assert len(cfg.all_joint_names) == 25
    assert len(cfg.sensor_joint_names) == 2
    assert len(cfg.controlled_joint_names) == 23
    assert "xl330_joint" in cfg.sensor_joint_names
    assert "d455_joint" in cfg.sensor_joint_names
    assert cfg.control_dt == cfg.sim_dt * cfg.decimation

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"stacked_obs_dim = {cfg.stacked_obs_dim}")
    print_ok(f"control_dt = {cfg.control_dt}")
    print_ok("Task1Config 基础配置正常")


def check_curriculum(env: G1Task1Env, cfg: Task1Config) -> None:
    heading("[测试 5] curriculum / harness / target_vx 检测")

    old_steps = int(env.global_steps)

    rows = []

    checks = [
        (0.00, 0),
        (0.10, 0),
        (0.15, 1),
        (0.35, 2),
        (0.65, 3),
        (0.90, 4),
        (1.00, 4),
    ]

    for k, expected_stage in checks:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        target_vx, harness, stage = env._curriculum_values()

        assert stage == expected_stage, f"k={k} stage wrong: got {stage}, expected {expected_stage}"
        assert 0.0 <= target_vx <= cfg.target_vx_final + 1e-5
        assert 0.0 <= harness <= cfg.harness_start + 1e-5

        rows.append((k, stage, target_vx, harness, env._reference_reset_scale(), env._style_weight_scale()))

    env.global_steps = old_steps

    for k, stage, target_vx, harness, ref_scale, style_scale in rows:
        print_ok(
            f"k={k:.2f} | stage={stage} | target_vx={target_vx:.3f} | "
            f"harness={harness:.3f} | ref_reset_scale={ref_scale:.3f} | style_scale={style_scale:.3f}"
        )

    print_ok("curriculum / harness / target_vx 正常")


def check_forced_events(env: G1Task1Env, cfg: Task1Config) -> None:
    heading("[测试 8] 强制 low-height / tilt / high-height / timeout 事件检测")

    zero_action = torch.zeros((cfg.num_envs, env.num_actions), dtype=torch.float32, device=env.device)

    # Low height fall.
    env.global_steps = 0
    env.reset(seed=args_cli.seed)

    low_ids = torch.arange(min(4, cfg.num_envs), dtype=torch.long, device=env.device)
    force_root_pose(env, low_ids, height=cfg.fall_height * 0.5)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    low_hit = int(terminated[low_ids].sum().item())
    assert low_hit > 0, "forced low height did not trigger fall termination"
    print_ok(f"低高度 fall 事件触发正常: {low_hit}/{len(low_ids)}")

    # Tilt fall.
    env.reset(seed=args_cli.seed)
    tilt_ids = torch.arange(min(4, cfg.num_envs), dtype=torch.long, device=env.device)
    bad_quat = quat_from_roll_pitch_yaw(roll=1.35, pitch=0.0, yaw=0.0, device=env.device)
    force_root_pose(env, tilt_ids, height=cfg.target_height, quat=bad_quat)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    tilt_hit = int(terminated[tilt_ids].sum().item())
    assert tilt_hit > 0, "forced tilt did not trigger fall termination"
    print_ok(f"倾斜 fall 事件触发正常: {tilt_hit}/{len(tilt_ids)}")

    # High height / abnormal jump.
    env.reset(seed=args_cli.seed)
    high_ids = torch.arange(min(4, cfg.num_envs), dtype=torch.long, device=env.device)
    force_root_pose(env, high_ids, height=cfg.jump_height + 0.20)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    high_hit = int(terminated[high_ids].sum().item())
    assert high_hit > 0, "forced high height did not trigger termination"
    print_ok(f"高高度 jump/fall 事件触发正常: {high_hit}/{len(high_ids)}")

    # Timeout.
    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = cfg.max_episode_length - 1

    obs, reward, terminated, truncated, info = env.step(zero_action)
    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "max_episode_length did not trigger truncated"

    flat = flatten_info(info)
    assert "events/Timeout_Rate" in flat
    assert "events/Fall_Rate" in flat

    print_ok(f"timeout 截断触发正常: truncated={timeout_count}")


def run_tests() -> None:
    heading("Unitree G1 Task1 Assisted Locomotion Env 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.steps = min(int(args_cli.steps), 80)
        args_cli.num_envs = min(int(args_cli.num_envs), 4)

    check_project_files()
    check_config()

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.print_debug_info = bool(args_cli.print_names)

    check_assets(cfg)

    env: G1Task1Env | None = None

    try:
        heading("[测试 3] G1Task1Env 初始化 / 关节映射 / 空间维度检测")
        env = G1Task1Env(cfg)

        print_ok(f"device = {device}")
        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {env.num_actions}")
        print_ok(f"num_observations = {cfg.num_observations}")
        print_ok(f"observation_space = {env.observation_space}")
        print_ok(f"action_space = {env.action_space}")
        print_ok(f"controlled_joint_ids = {env.controlled_joint_ids}")
        print_ok(f"sensor_joint_ids = {env.sensor_joint_ids}")
        print_ok(f"foot_body_ids = {env.foot_body_ids}")
        print_ok(f"contact_foot_ids = {env.contact_foot_ids}")
        print_ok(f"motion frames = {env.motion.num_frames}")
        print_ok(f"robot mass = {env.robot_mass:.3f} kg")

        assert env.robot.num_joints == 25, f"expected 25 robot joints, got {env.robot.num_joints}"
        assert env.num_actions == 23
        assert len(env.controlled_joint_ids) == 23
        assert len(env.sensor_joint_ids) == 2
        assert len(env.foot_body_ids) == 2
        assert len(env.contact_foot_ids) == 2
        assert env.observation_space.shape == (cfg.num_observations,)
        assert env.state_space.shape == (cfg.num_observations,)
        assert env.action_space.shape == (env.num_actions,)

        sensor_names = [env.robot_joint_names[i] for i in env.sensor_joint_ids]
        assert "xl330_joint" in sensor_names
        assert "d455_joint" in sensor_names

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

        heading("[测试 4] reset / observation shape / observation layout 检测")
        obs, info = env.reset(seed=args_cli.seed)

        check_obs_shape_and_values(env, obs)
        check_observation_slices(env, obs)

        print_ok(f"reset obs shape = {tuple(obs.shape)}")
        print_ok(f"obs range = {obs.min().item():.4f} ~ {obs.max().item():.4f}")

        root_h = env.robot.data.root_pos_w[:, 2] - env.env_origins[:, 2]
        assert torch.isfinite(root_h).all()
        assert 0.30 <= root_h.mean().item() <= 1.20

        print_ok(f"root height mean = {root_h.mean().item():.6f}")

        check_curriculum(env, cfg)

        heading("[测试 6] action control / sensor joints fixed / info fields 检测")
        env.global_steps = int(0.20 * cfg.curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        sensor_q0 = env.robot.data.joint_pos[:, env.sensor_joint_ids_t].clone()
        q0 = env.robot.data.joint_pos[:, env.controlled_joint_ids_t].clone()

        test_action = torch.empty((cfg.num_envs, env.num_actions), dtype=torch.float32, device=env.device).uniform_(-1.0, 1.0)

        latest_info: Dict[str, Any] = {}

        for _ in range(20):
            obs, reward, terminated, truncated, latest_info = env.step(test_action)

        q1 = env.robot.data.joint_pos[:, env.controlled_joint_ids_t].clone()
        sensor_q1 = env.robot.data.joint_pos[:, env.sensor_joint_ids_t].clone()

        q_delta = torch.norm(q1 - q0, dim=-1).mean().item()
        sensor_delta = torch.norm(sensor_q1 - sensor_q0, dim=-1).mean().item()

        assert q_delta > 1e-5, "action did not change controlled joint positions"
        assert sensor_delta < 5e-3, f"sensor joints moved too much: {sensor_delta:.6f}"

        assert reward.shape == (cfg.num_envs,)
        assert terminated.shape == (cfg.num_envs,)
        assert truncated.shape == (cfg.num_envs,)
        assert_finite_tensor("reward", reward)

        check_obs_shape_and_values(env, obs)
        check_observation_slices(env, obs)

        flat = flatten_info(latest_info)
        required_info_keys = [
            "reward_components/Total",
            "reward_components/R_Vx",
            "reward_components/R_Upright",
            "reward_components/R_Height",
            "reward_components/P_Foot_Slip",
            "events/Fall_Rate",
            "events/Timeout_Rate",
            "telemetry/Curriculum_K",
            "telemetry/Curriculum_Stage",
            "telemetry/Target_Vx",
            "telemetry/Actual_Vx",
            "telemetry/Base_Height",
            "telemetry/Harness_Ratio",
            "telemetry/Contact_Count",
            "debug/Obs_Dim",
            "debug/Action_Dim",
        ]

        for key in required_info_keys:
            assert key in flat, f"info missing field: {key}"

        print_ok(f"controlled joints 平均位移范数 = {q_delta:.6f}")
        print_ok(f"sensor joints 平均位移范数 = {sensor_delta:.8f}")
        print_ok("action control / sensor joints fixed / info fields 正常")

        heading("[测试 7] contact sensor / reference motion 接口检测")
        contact, normal_force = env._get_feet_contact()

        assert contact.shape == (cfg.num_envs, 2)
        assert normal_force.shape == (cfg.num_envs, 2)
        assert_finite_tensor("contact", contact)
        assert_finite_tensor("normal_force", normal_force)

        ref_pos, ref_vel, ref_contact = env.motion.get_reference_by_phase(env.phase)

        assert ref_pos.shape == (cfg.num_envs, 23)
        assert ref_vel.shape == (cfg.num_envs, 23)
        assert ref_contact.shape == (cfg.num_envs, 2)
        assert_finite_tensor("ref_pos", ref_pos)
        assert_finite_tensor("ref_vel", ref_vel)
        assert_finite_tensor("ref_contact", ref_contact)
        assert torch.all(ref_contact >= -1e-5)
        assert torch.all(ref_contact <= 1.0 + 1e-5)

        print_ok(f"contact shape = {tuple(contact.shape)}")
        print_ok(f"normal_force shape = {tuple(normal_force.shape)}")
        print_ok(f"ref_pos shape = {tuple(ref_pos.shape)}")
        print_ok(f"ref_vel shape = {tuple(ref_vel.shape)}")
        print_ok(f"ref_contact shape = {tuple(ref_contact.shape)}")
        print_ok("contact sensor / reference motion 接口正常")

        check_forced_events(env, cfg)

        heading("[测试 9] 随机策略 rollout 稳定性检测")
        env.global_steps = int(float(args_cli.rollout_k) * cfg.curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        records: List[Dict[str, float]] = []
        total_terminated = 0
        total_truncated = 0

        start_time = time.time()

        for step in range(int(args_cli.steps)):
            action = torch.empty(
                (cfg.num_envs, env.num_actions),
                dtype=torch.float32,
                device=env.device,
            ).uniform_(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            total_terminated += int(terminated.sum().item())
            total_truncated += int(truncated.sum().item())

            if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                check_obs_shape_and_values(env, obs)
                check_observation_slices(env, obs)
                assert_finite_tensor("rollout_reward", reward)

                flat = flatten_info(info)

                row = {
                    "test/step": float(step),
                    "test/reward_mean": float(reward.detach().mean().cpu().item()),
                    "test/terminated_rate": float(terminated.float().mean().cpu().item()),
                    "test/truncated_rate": float(truncated.float().mean().cpu().item()),
                }
                row.update(flat)
                records.append(row)

                print(
                    f"step={step + 1:>5}/{args_cli.steps} | "
                    f"reward={row.get('test/reward_mean', 0.0):>8.4f} | "
                    f"k={row.get('telemetry/Curriculum_K', 0.0):>5.3f} | "
                    f"stage={row.get('telemetry/Curriculum_Stage', 0.0):>4.1f} | "
                    f"target_vx={row.get('telemetry/Target_Vx', 0.0):>6.3f} | "
                    f"vx={row.get('telemetry/Actual_Vx', 0.0):>7.3f} | "
                    f"fall={row.get('events/Fall_Rate', 0.0):>6.3f} | "
                    f"harness={row.get('telemetry/Harness_Ratio', 0.0):>6.3f} | "
                    f"h={row.get('telemetry/Base_Height', 0.0):>5.3f} | "
                    f"ct={row.get('telemetry/Contact_Count', 0.0):>4.2f}",
                    flush=True,
                )

        elapsed = time.time() - start_time
        fps = int(args_cli.steps) * int(cfg.num_envs) / max(elapsed, 1e-6)

        print_ok(f"随机策略 rollout 完成: {args_cli.steps} control steps")
        print_ok(f"总 transitions: {args_cli.steps * cfg.num_envs:,}")
        print_ok(f"吞吐约: {fps:,.2f} env steps/s")
        print_ok(f"累计 terminated: {total_terminated:,}")
        print_ok(f"累计 truncated: {total_truncated:,}")

        heading("[测试 10] 奖励组件 / 事件 / 遥测统计报告")
        print_summary_table(summarize_records(records))

        print("G1 Task1 training pre-check guide:")
        print("1. 单帧 obs 必须为 123，action 必须为 23。")
        print("2. xl330_joint 和 d455_joint 是传感器关节，不应由策略控制。")
        print("3. 随机策略下 fall 可以出现，但不能出现 NaN/Inf。")
        print("4. Task1 是人形机器人纯 RL 早期 baseline，不代表专业动作控制最终路线。")
        print("5. 正式训练时重点看 Target_Vx / Actual_Vx、Base_Height、Fall_Rate、Harness_Ratio、Contact_Count。")
        print("6. 如果前期 Fall_Rate 高，优先检查 harness、height/upright 奖励、action scale 和 reset 姿态。")

        heading("Unitree G1 Task1 Assisted Locomotion Env 测试全部通过")

    except Exception as exc:
        print("\n❌ G1 Task1 环境测试失败：")
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
