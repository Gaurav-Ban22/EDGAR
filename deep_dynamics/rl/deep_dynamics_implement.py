
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml

from deep_dynamics.model.models import string_to_model
from deep_dynamics.rl.track import Track

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsWrapper:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        scaler_path: str,
        track: Track,
        device: torch.device = device,
    ):
        with open(config_path, "r") as f:
            self.param_dict = yaml.load(f, Loader=yaml.SafeLoader)

        self.device = device
        self.track = track

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
        self.n_features = self.n_states + self.n_actionss

        self.history: np.ndarray | None = None
        self.pose: np.ndarray | None = None  # [x, y, psi]
        self.h: torch.Tensor | None = None  # RNN hidden state

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
        vx: float = 10.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
        throttle_fb: float = 0.0,
        steering_fb: float = 0.0,
    ) -> dict:
        self.history = np.zeros((self.horizon, self.n_features), dtype=np.float64)
        self.history[:, 0] = vx
        self.history[:, 1] = vy
        self.history[:, 2] = yaw_rate
        self.history[:, 3] = throttle_fb
        self.history[:, 4] = steering_fb

        self.pose = np.array([x, y, psi], dtype=np.float64)

        if self.model.is_rnn:
            self.h = self.model.init_hidden(1)

        return self._build_obs()

    def step(
        self, throttle_cmd: float, steering_cmd: float
    ) -> Tuple[dict, np.ndarray]:

        prev = self.history[-1].copy()
        new_row = np.zeros(self.n_features, dtype=np.float64)
        new_row[0] = prev[0]  # VX  (will be overwritten by prediction below)
        new_row[1] = prev[1]  # VY
        new_row[2] = prev[2]  # YAW_RATE
        new_row[3] = prev[3] + throttle_cmd  # THROTTLE_FB accumulates
        new_row[4] = prev[4] + steering_cmd  # STEERING_FB accumulates
        new_row[5] = throttle_cmd             # THROTTLE_CMD (this step's delta)
        new_row[6] = steering_cmd             # STEERING_CMD (this step's delta)

        self.history = np.vstack([self.history[1:], new_row[np.newaxis, :]])

        x_raw = torch.from_numpy(self.history[np.newaxis]).float().to(self.device)

        history_flat = self.history.reshape(-1, self.n_features)
        history_norm = self.scaler.transform(history_flat).reshape(
            1, self.horizon, self.n_features
        )
        x_norm = torch.from_numpy(history_norm).float().to(self.device)

        with torch.no_grad():
            if self.model.is_rnn:
                self.h = self.h.data
                prediction, self.h, sysid = self.model(x_raw, x_norm, self.h)
            else:
                prediction, _, sysid = self.model(x_raw, x_norm)

        pred = prediction.cpu().numpy().squeeze()  # (3,)

        self.history[-1, 0] = pred[0]
        self.history[-1, 1] = pred[1]
        self.history[-1, 2] = pred[2]

        vx, vy, omega = pred
        psi = self.pose[2]
        self.pose[0] += (vx * np.cos(psi) - vy * np.sin(psi)) * self.timestep
        self.pose[1] += (vx * np.sin(psi) + vy * np.cos(psi)) * self.timestep
        self.pose[2] += omega * self.timestep

        return self._build_obs(), pred


    def _build_obs(self) -> dict:
        """Return a dict that an RL policy can consume."""
        idx, s, e_lat, track_heading = self.track.project(
            self.pose[0], self.pose[1]
        )
        heading_error = self.pose[2] - track_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        return {
            "vx": self.history[-1, 0],
            "vy": self.history[-1, 1],
            "yaw_rate": self.history[-1, 2],
            "throttle_fb": self.history[-1, 3],
            "steering_fb": self.history[-1, 4],
            "x": self.pose[0],
            "y": self.pose[1],
            "psi": self.pose[2],
            "s": s,
            "e_lat": e_lat,
            "heading_error": heading_error,
            "track_heading": track_heading,
            "curvature": self.track.curvature_at(s),
            "is_inside": self.track.is_inside(self.pose[0], self.pose[1]),
            "progress": self.track.normalized_progress(s),
        }
