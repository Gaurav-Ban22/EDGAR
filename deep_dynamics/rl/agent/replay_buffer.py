"""
Phase 5: circular replay buffer for off-policy RL.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class ReplayBuffer:
    """Stores ``(obs, action, reward, next_obs, done)``; samples uniform batches."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
    ):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._ptr = 0
        self._size = 0

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self._ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[self._ptr] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self._ptr, 0] = float(reward)
        self.next_obs[self._ptr] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[self._ptr, 0] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size > self._size:
            raise ValueError(
                f"batch_size {batch_size} > buffer size {self._size}"
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx].copy(),
            "actions": self.actions[idx].copy(),
            "rewards": self.rewards[idx].copy(),
            "next_obs": self.next_obs[idx].copy(),
            "dones": self.dones[idx].copy(),
        }
