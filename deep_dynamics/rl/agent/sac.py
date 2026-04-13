"""
Phase 5: Soft Actor-Critic with automatic entropy tuning (learnable log α).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from deep_dynamics.rl.agent.networks import GaussianActor, TwinQNetwork
from deep_dynamics.rl.agent.replay_buffer import ReplayBuffer


def _to_device(
    batch: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in batch.items()
    }


def _torch_load_trusted(path: Union[str, Path], map_location: torch.device) -> Any:
    """Load SAC checkpoint dict (numpy + tensors). PyTorch ≥2.6 defaults ``weights_only=True``."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SAC:
    """SAC agent: Gaussian actor, twin Q, soft target update, entropy coefficient."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        cfg = config or {}
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        hidden_dim = int(cfg.get("hidden_dim", 256))
        lr_actor = float(cfg.get("lr_actor", 3e-4))
        lr_critic = float(cfg.get("lr_critic", 3e-4))
        lr_alpha = float(cfg.get("lr_alpha", 3e-4))
        self.target_entropy = cfg.get("target_entropy", None)
        if self.target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(self.target_entropy)

        al = np.asarray(action_low, dtype=np.float32).reshape(-1)
        ah = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.action_low = al
        self.action_high = ah

        self.actor = GaussianActor(
            obs_dim, action_dim, hidden_dim, al, ah
        ).to(self.device)
        self.critic = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_alpha = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def select_action(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            a, _ = self.actor(
                o, deterministic=deterministic, with_log_prob=False
            )
            return a.cpu().numpy().reshape(-1)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """One gradient step on critic, actor, α; soft-update targets."""
        b = _to_device(batch, self.device)
        obs, actions, rewards, next_obs, dones = (
            b["obs"],
            b["actions"],
            b["rewards"],
            b["next_obs"],
            b["dones"],
        )

        self.alpha = float(self.log_alpha.exp().detach().cpu())

        with torch.no_grad():
            next_actions, next_log_pi = self.actor(
                next_obs, deterministic=False, with_log_prob=True
            )
            assert next_log_pi is not None
            target_q = self.critic_target.q_min(next_obs, next_actions)
            target = rewards + (1.0 - dones) * self.gamma * (
                target_q - self.alpha * next_log_pi
            )

        q1, q2 = self.critic(obs, actions)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.optim_critic.zero_grad()
        loss_q.backward()
        self.optim_critic.step()

        new_actions, log_pi = self.actor(
            obs, deterministic=False, with_log_prob=True
        )
        assert log_pi is not None
        q_min_online = self.critic.q_min(obs, new_actions)
        loss_actor = (self.alpha * log_pi - q_min_online).mean()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        loss_alpha = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()

        self._soft_update(self.critic, self.critic_target)

        return {
            "loss_q": float(loss_q.detach().cpu()),
            "loss_actor": float(loss_actor.detach().cpu()),
            "loss_alpha": float(loss_alpha.detach().cpu()),
            "alpha": float(self.log_alpha.exp().detach().cpu()),
        }

    def _soft_update(self, online: TwinQNetwork, target: TwinQNetwork) -> None:
        with torch.no_grad():
            for p_t, p in zip(target.parameters(), online.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "optim_actor": self.optim_actor.state_dict(),
                "optim_critic": self.optim_critic.state_dict(),
                "optim_alpha": self.optim_alpha.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "gamma": self.gamma,
                "tau": self.tau,
                "target_entropy": self.target_entropy,
            },
            path,
        )

    def load(self, path: Union[str, Path], load_optimizers: bool = True) -> None:
        ckpt = _torch_load_trusted(path, self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        la = ckpt["log_alpha"]
        la_t = la if isinstance(la, torch.Tensor) else torch.as_tensor(
            la, dtype=torch.float32, device=self.device
        )
        self.log_alpha.data.copy_(la_t.to(self.device).reshape_as(self.log_alpha.data))
        if load_optimizers:
            self.optim_actor.load_state_dict(ckpt["optim_actor"])
            self.optim_critic.load_state_dict(ckpt["optim_critic"])
            self.optim_alpha.load_state_dict(ckpt["optim_alpha"])
        self.alpha = float(self.log_alpha.exp().detach().cpu())
