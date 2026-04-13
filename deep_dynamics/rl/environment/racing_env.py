"""
Phase 4: Gymnasium racing environment — Deep Dynamics + pose + track geometry.

Config dict (minimum):

* ``track_dir`` + ``track_name`` **or** ``inner_csv`` + ``outer_csv``
* ``model_config_path``, ``checkpoint_path``, ``scaler_path``
* optional: ``device``, ``max_steps``, ``initial_speed``, reward weights,
  action bounds, ``n_curvature_points``, termination penalties, etc.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from deep_dynamics.rl.environment.deep_dynamics_implement import DynamicsWrapper
from deep_dynamics.rl.environment.pose_integrator import PoseIntegrator, wrap_to_pi
from deep_dynamics.rl.environment.track import Track


def _normalize_angle(angle: float) -> float:
    return wrap_to_pi(angle)


def _as_torch_device(dev: Union[str, torch.device]) -> torch.device:
    if isinstance(dev, torch.device):
        return dev
    return torch.device(dev)


class RacingEnv(gym.Env):
    """SAC-ready loop track env using a frozen Deep Dynamics model."""

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._config = config

        if "inner_csv" in config and "outer_csv" in config:
            self.track = Track.from_csv(
                config["inner_csv"],
                config["outer_csv"],
                name=config.get("track_name"),
            )
        elif "track_dir" in config and "track_name" in config:
            self.track = Track.from_track_dir(
                config["track_dir"], config["track_name"]
            )
        else:
            raise ValueError(
                "config must provide either (inner_csv, outer_csv) or "
                "(track_dir, track_name)"
            )

        self.dynamics = DynamicsWrapper(
            config_path=config["model_config_path"],
            checkpoint_path=config["checkpoint_path"],
            scaler_path=config["scaler_path"],
            device=_as_torch_device(config.get("device", "cpu")),
        )
        self.pose = PoseIntegrator(self.dynamics.Ts)

        self.max_steps = int(config.get("max_steps", 5000))
        self.initial_speed = float(config.get("initial_speed", 15.0))
        self.boundary_margin = float(config.get("boundary_margin", 0.0))

        self.n_curvature_points = int(config.get("n_curvature_points", 10))
        cl_space = float(config.get("curvature_lookahead_spacing", 5.0))
        self.curvature_lookaheads = np.array(
            [cl_space * (i + 1) for i in range(self.n_curvature_points)],
            dtype=np.float64,
        )

        # Reward weights (plan defaults)
        self._alpha_d = float(config.get("alpha_d", 0.1))
        self._alpha_h = float(config.get("alpha_h", 0.5))
        self._alpha_a = float(config.get("alpha_a", 0.01))
        self._alpha_v = float(config.get("alpha_v", 0.0))

        self._vx_max = float(config.get("vx_max", 50.0))
        self._alpha_overspeed = float(config.get("alpha_overspeed", 1.0))

        self.crash_penalty = float(config.get("crash_penalty", 100.0))
        self.stuck_vx_threshold = float(config.get("stuck_vx_threshold", 1.0))
        self.stuck_steps_threshold = int(config.get("stuck_steps_threshold", 50))
        self.stuck_penalty = float(config.get("stuck_penalty", 50.0))
        self.lap_bonus = float(config.get("lap_bonus", 0.0))

        low = np.array(config.get("action_low", [-0.5, -0.05]), dtype=np.float32)
        high = np.array(config.get("action_high", [0.5, 0.05]), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.prev_s: float = 0.0
        self.step_count: int = 0
        self.lap_count: int = 0
        self._slow_steps: int = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        # Cached state for observations (updated in reset/step)
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._s = 0.0
        self._d = 0.0
        self._track_heading = 0.0
        self._heading_error = 0.0
        self._delta_s = 0.0
        self._throttle_fb = 0.0
        self._steering_fb = 0.0
        self._sysid: np.ndarray = np.array([])

    def _compute_obs_dim(self) -> int:
        # 3 vel + 3 track-relative + n curvature + 4 controls
        return 10 + self.n_curvature_points

    def _sync_feedback_from_dynamics(self) -> None:
        assert self.dynamics.history is not None
        row = self.dynamics.history[-1]
        self._throttle_fb = float(row[3])
        self._steering_fb = float(row[4])

    def _get_obs(self) -> np.ndarray:
        curv = self.track.lookahead_curvatures(self._s, self.curvature_lookaheads)
        progress_speed = self._delta_s / self.dynamics.Ts
        parts = [
            np.array(
                [
                    self._vx,
                    self._vy,
                    self._yaw_rate,
                    self._d,
                    self._heading_error,
                    progress_speed,
                ],
                dtype=np.float32,
            ),
            curv.astype(np.float32),
            np.array(
                [
                    self._throttle_fb,
                    self._steering_fb,
                    self._last_action[0],
                    self._last_action[1],
                ],
                dtype=np.float32,
            ),
        ]
        return np.concatenate(parts, axis=0)

    def _compute_reward(
        self,
        vx: float,
        delta_s: float,
        d: float,
        heading_error: float,
        action: np.ndarray,
    ) -> float:
        r_progress = delta_s / self.dynamics.Ts
        r_lateral = -self._alpha_d * (d * d)
        r_heading = -self._alpha_h * (heading_error * heading_error)
        r_smooth = -self._alpha_a * float(action[0] ** 2 + action[1] ** 2)
        r_speed = self._alpha_v * vx

        r_overspeed = 0.0
        if self._vx_max > 0.0 and vx > self._vx_max:
            r_overspeed = -self._alpha_overspeed * (vx - self._vx_max) ** 2

        return float(
            r_progress + r_lateral + r_heading + r_smooth + r_speed + r_overspeed
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        n = len(self.track.centerline)
        if "start_idx" in options and options["start_idx"] is not None:
            start_idx = int(options["start_idx"]) % n
        elif options.get("randomize_start", False):
            start_idx = int(self.np_random.integers(0, n))
        else:
            start_idx = 0

        x0 = float(self.track.centerline[start_idx, 0])
        y0 = float(self.track.centerline[start_idx, 1])
        heading0 = float(self.track.headings[start_idx])

        self.pose.reset(x0, y0, heading0)
        self.dynamics.reset(
            vx=self.initial_speed,
            vy=0.0,
            yaw_rate=0.0,
            throttle_fb=float(options.get("initial_throttle_fb", 0.0)),
            steering_fb=float(options.get("initial_steering_fb", 0.0)),
        )

        self.prev_s = float(self.track.cumulative_s[start_idx])
        self.step_count = 0
        self.lap_count = 0
        self._slow_steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        self._x = self.pose.x
        self._y = self.pose.y
        self._heading = self.pose.heading
        self._s, self._d, self._track_heading = self.track.cartesian_to_frenet(
            self._x, self._y
        )
        self._heading_error = _normalize_angle(self._heading - self._track_heading)
        self._vx = self.initial_speed
        self._vy = 0.0
        self._yaw_rate = 0.0
        self._delta_s = 0.0
        self._sync_feedback_from_dynamics()
        self._sysid = np.array([])

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        throttle_cmd = float(action[0])
        steering_cmd = float(action[1])

        pred, sysid = self.dynamics.step(throttle_cmd, steering_cmd)
        vx, vy, yaw_rate = float(pred[0]), float(pred[1]), float(pred[2])

        x, y, heading = self.pose.step(vx, vy, yaw_rate)
        s, d, track_heading = self.track.cartesian_to_frenet(x, y)
        heading_error = _normalize_angle(heading - track_heading)
        delta_s, lap_complete = self.track.update_progress(self.prev_s, s)
        self.prev_s = s
        self.step_count += 1
        self._last_action = action.copy()

        reward = self._compute_reward(vx, delta_s, d, heading_error, action)

        terminated = False
        in_bounds = self.track.is_within_bounds(x, y, margin=self.boundary_margin)
        if not in_bounds:
            terminated = True
            reward -= self.crash_penalty
        else:
            if vx < self.stuck_vx_threshold:
                self._slow_steps += 1
                if self._slow_steps >= self.stuck_steps_threshold:
                    terminated = True
                    reward -= self.stuck_penalty
            else:
                self._slow_steps = 0

        if lap_complete:
            self.lap_count += 1
            reward += self.lap_bonus

        truncated = self.step_count >= self.max_steps

        term_reason: Optional[str] = None
        if terminated:
            term_reason = "crash" if not in_bounds else "stuck"
        elif truncated:
            term_reason = "timeout"

        self._vx, self._vy, self._yaw_rate = vx, vy, yaw_rate
        self._x, self._y = x, y
        self._heading = heading
        self._s, self._d, self._track_heading = s, d, track_heading
        self._heading_error = heading_error
        self._delta_s = delta_s
        self._sysid = np.asarray(sysid, dtype=np.float64).reshape(-1)
        self._sync_feedback_from_dynamics()

        info: Dict[str, Any] = {
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "x": x,
            "y": y,
            "heading": heading,
            "s": s,
            "d": d,
            "heading_error": heading_error,
            "lap_count": self.lap_count,
            "delta_s": delta_s,
            "sysid": self._sysid,
            "throttle_fb": self._throttle_fb,
            "steering_fb": self._steering_fb,
            "step_reward": float(reward),
            "termination_reason": term_reason,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None
