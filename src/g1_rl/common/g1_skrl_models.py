from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class G1GaussianPolicy(GaussianMixin, Model):
    """Gaussian actor for G1 pure-RL baseline.

    This model is intentionally simple and stable:
        MLP: input -> 512 -> 256 -> 128 -> action_dim

    It is suitable for educational pure-RL baselines and skrl PPO.
    """

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        *,
        init_log_std: float = -1.35,
        min_log_std: float = -5.0,
        max_log_std: float = 0.20,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=float(min_log_std),
            max_log_std=float(max_log_std),
            reduction="sum",
        )

        obs_dim = int(observation_space.shape[0])
        act_dim = int(action_space.shape[0])

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )

        self.log_std_parameter = nn.Parameter(
            torch.ones(act_dim, dtype=torch.float32) * float(init_log_std)
        )

        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def compute(self, inputs, role):
        x = inputs.get("observations", inputs.get("states"))
        return self.net(x), {"log_std": self.log_std_parameter}


class G1DeterministicValue(DeterministicMixin, Model):
    """Deterministic value model for G1 skrl PPO."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        DeterministicMixin.__init__(self, clip_actions=False)

        state_dim = int(state_space.shape[0])

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def compute(self, inputs, role):
        return self.net(inputs.get("states")), {}


# Backward-compatible aliases
G1Actor = G1GaussianPolicy
G1Critic = G1DeterministicValue
