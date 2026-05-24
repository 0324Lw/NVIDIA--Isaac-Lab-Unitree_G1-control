from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from g1_rl.common.info_utils import to_float, write_scalars


class G1FrameStackWrapper(gym.Env):
    """Reusable G1 frame-stack wrapper for skrl.

    Raw env obs:
        Task1 = 123

    Stacked obs:
        Task1 = 123 * 5 = 615

    Return format:
        {
            "policy": stacked_obs,
            "critic": stacked_obs,
        }

    This matches the verified Go2 model_test / train style:
        raw_env -> FrameStackWrapper -> skrl.wrap_env(...)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        log_dir: str,
        n_stack: int = 5,
        tb_log_interval_steps: int = 50,
        use_privileged_obs: bool = False,
    ):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device
        self.use_privileged_obs = bool(use_privileged_obs)

        self.single_dim = int(env.observation_space.shape[0])
        self.stacked_dim = self.single_dim * self.n_stack

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_dim,),
            dtype=np.float32,
        )

        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_dim,),
            dtype=np.float32,
        )

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

        self.writer = SummaryWriter(log_dir) if int(tb_log_interval_steps) != 0 else None
        self.tb_log_interval_steps = int(tb_log_interval_steps)

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
