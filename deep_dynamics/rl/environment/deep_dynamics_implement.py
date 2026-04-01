"""
Phase 2: Dynamics wrapper — loads a trained Deep Dynamics checkpoint
and provides a step(throttle_cmd, steering_cmd) interface that manages
the rolling history buffer and scaler normalization.

Pose integration is handled separately by PoseIntegrator (Phase 3).
Track-relative observations are built by RacingEnv (Phase 4).
"""

from __future__ import annotations

import pickle
from typing import Tuple

import numpy as np
import torch
import yaml

from deep_dynamics.model.models import string_to_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsWrapper:
    """Wraps a trained DeepDynamics (or DeepPacejka) checkpoint.

    Manages:
      * rolling ``(horizon, n_features)`` history buffer (raw + normalised)
      * the sklearn StandardScaler used during training
      * RNN hidden state (if the model is recurrent)

    Does NOT manage world-frame pose — that is PoseIntegrator's job.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        scaler_path: str,
        device: torch.device = device,
    ):
        with open(config_path, "r") as f:
            self.param_dict = yaml.load(f, Loader=yaml.SafeLoader)

        self.device = device

        self.model = string_to_model[self.param_dict["MODEL"]["NAME"]](
            self.param_dict, eval=True
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.horizon = self.model.horizon
        self.timestep = self.model.timestep
        self.n_states = len(self.param_dict["STATE"])
        self.n_actions = len(self.param_dict["ACTIONS"])
        self.n_features = self.n_states + self.n_actions  # 7 for IAC DeepDynamics

        # Runtime state (populated by reset)
        self.history: np.ndarray | None = None
        self.h: torch.Tensor | None = None  # RNN hidden state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        vx: float = 10.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
        throttle_fb: float = 0.0,
        steering_fb: float = 0.0,
    ) -> None:
        """Initialise the history buffer with a steady-state."""
        # [VX, VY, YAW_RATE, THROTTLE_FB, STEERING_FB, THROTTLE_CMD, STEERING_CMD]
        self.history = np.zeros((self.horizon, self.n_features), dtype=np.float64)
        self.history[:, 0] = vx
        self.history[:, 1] = vy
        self.history[:, 2] = yaw_rate
        self.history[:, 3] = throttle_fb
        self.history[:, 4] = steering_fb

        if self.model.is_rnn:
            self.h = self.model.init_hidden(1)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, throttle_cmd: float, steering_cmd: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one timestep.

        Parameters
        ----------
        throttle_cmd, steering_cmd : float
            **Incremental** commands (same semantics as training data).

        Returns
        -------
        prediction : np.ndarray
            ``[vx, vy, yaw_rate]`` predicted by the physics head.
        sysid : np.ndarray
            Raw system-identification parameter vector from the network.
        """
        # 1. Build the new feature row
        prev = self.history[-1].copy()
        new_row = np.zeros(self.n_features, dtype=np.float64)
        new_row[0] = prev[0]                  # VX  (overwritten below)
        new_row[1] = prev[1]                  # VY
        new_row[2] = prev[2]                  # YAW_RATE
        new_row[3] = prev[3] + throttle_cmd   # THROTTLE_FB accumulates
        new_row[4] = prev[4] + steering_cmd   # STEERING_FB accumulates
        new_row[5] = throttle_cmd              # THROTTLE_CMD (delta)
        new_row[6] = steering_cmd              # STEERING_CMD (delta)

        # Slide window: drop oldest, append new
        self.history = np.vstack([self.history[1:], new_row[np.newaxis, :]])

        # 2. Normalise and prepare tensors — model.forward(x_raw, x_norm, h0)
        x_raw = torch.from_numpy(self.history[np.newaxis]).float().to(self.device)

        history_flat = self.history.reshape(-1, self.n_features)
        history_norm = self.scaler.transform(history_flat).reshape(
            1, self.horizon, self.n_features
        )
        x_norm = torch.from_numpy(history_norm).float().to(self.device)

        # 3. Forward pass
        with torch.no_grad():
            if self.model.is_rnn:
                self.h = self.h.data
                prediction, self.h, sysid = self.model(x_raw, x_norm, self.h)
            else:
                prediction, _, sysid = self.model(x_raw, x_norm)

        pred = prediction.cpu().numpy().squeeze()       # (3,)
        sysid_np = sysid.cpu().numpy().squeeze()        # (n_params,)

        # 4. Write predicted velocities back into history for next step
        self.history[-1, 0] = pred[0]
        self.history[-1, 1] = pred[1]
        self.history[-1, 2] = pred[2]

        return pred, sysid_np
