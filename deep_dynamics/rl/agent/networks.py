"""
Phase 5: policy and Q networks for SAC (Gaussian actor + twin critics).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _init_linear(m: nn.Linear) -> None:
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)


class QNetwork(nn.Module):
    """(obs, action) → scalar Q."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        for m in (self.fc1, self.fc2, self.fc3):
            if isinstance(m, nn.Linear):
                _init_linear(m)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TwinQNetwork(nn.Module):
    """Two independent Q-networks for clipped double-Q learning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.minimum(q1, q2)


class GaussianActor(nn.Module):
    """MLP Gaussian policy with tanh squashing and affine map to action bounds."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        self.action_dim = action_dim
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        # a = bias + scale * tanh(u),  u ~ Normal(mean, std)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        for m in (self.fc1, self.fc2, self.mean, self.log_std):
            if isinstance(m, nn.Linear):
                _init_linear(m)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            pre_tanh = mean
        else:
            dist = Normal(mean, std)
            pre_tanh = dist.rsample()

        action = torch.tanh(pre_tanh)
        scaled = self.action_bias + self.action_scale * action

        if not with_log_prob:
            return scaled, None

        if deterministic:
            return scaled, None

        dist = Normal(mean, std)
        log_prob_u = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        squash_corr = torch.log(1.0 - action.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )
        # SAC's target entropy is specified in normalized tanh-action units.
        # Including the affine physical-action scale here adds a large constant
        # when command ranges are tiny (especially steering), which makes
        # automatic entropy tuning drive alpha upward without improving policy
        # shape. The constant has no useful actor gradient, so leave it out.
        log_prob = log_prob_u - squash_corr
        return scaled, log_prob
