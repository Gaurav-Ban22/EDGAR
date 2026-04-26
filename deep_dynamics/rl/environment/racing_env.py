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
from deep_dynamics.rl.environment.raceline import Raceline
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
        if "throttle_fb_range" in config:
            lo, hi = config["throttle_fb_range"]
            self.dynamics.throttle_fb_range = (float(lo), float(hi))
        if "steering_fb_range" in config:
            lo, hi = config["steering_fb_range"]
            self.dynamics.steering_fb_range = (float(lo), float(hi))
        self.pose = PoseIntegrator(self.dynamics.Ts)

        self.max_steps = int(config.get("max_steps", 5000))
        self.initial_speed = float(config.get("initial_speed", 15.0))
        self._initial_speed_from_reference = bool(
            config.get("initial_speed_from_reference", False)
        )
        self._initial_speed_scale = float(config.get("initial_speed_scale", 1.0))
        self._initial_speed_min = float(config.get("initial_speed_min", 5.0))
        self._initial_speed_max = float(
            config.get("initial_speed_max", config.get("vx_max", 75.0))
        )
        self.boundary_margin = float(config.get("boundary_margin", 0.0))

        # Curvature look-ahead distances (metres). Two config shapes supported:
        #   (preferred) curvature_lookahead_distances: [20, 40, 60, 80, ...]
        #   (legacy)    n_curvature_points + curvature_lookahead_spacing
        # The paper ("Formula RL", Remonda et al. 2019) shows that longer,
        # non-uniform look-aheads (20/40/60/80 m) matter much more than a
        # dense short-range grid — the car needs to SEE the next corner, not
        # just the 5 m in front of it.
        cl_list = config.get("curvature_lookahead_distances")
        if cl_list:
            self.curvature_lookaheads = np.asarray(cl_list, dtype=np.float64)
            self.n_curvature_points = int(len(self.curvature_lookaheads))
        else:
            self.n_curvature_points = int(config.get("n_curvature_points", 10))
            cl_space = float(config.get("curvature_lookahead_spacing", 5.0))
            self.curvature_lookaheads = np.array(
                [cl_space * (i + 1) for i in range(self.n_curvature_points)],
                dtype=np.float64,
            )

        # Optional expert raceline built from real telemetry. When present,
        # the Frenet frame used for reward/observation is the raceline's,
        # not the centerline's. Track is still used for boundary (crash)
        # detection and for the initial lap segmentation inside Raceline.
        raceline_csv = config.get("raceline_csv")
        self._use_raceline = bool(raceline_csv) and bool(
            config.get("use_raceline", True)
        )
        self.raceline: Optional[Raceline] = None
        if self._use_raceline:
            self.raceline = Raceline.from_telemetry(
                str(raceline_csv),
                track=self.track,
                smooth_window_xy=int(config.get("raceline_smooth_xy", 15)),
                smooth_window_vx=int(config.get("raceline_smooth_vx", 25)),
                resample_ds=float(config.get("raceline_resample_ds", 2.0)),
                min_lap_coverage=float(
                    config.get("raceline_min_lap_coverage", 0.90)
                ),
            )
            print(f"[RacingEnv] {self.raceline!r}")
        # Whether the agent's speed reference should track the raceline's
        # vx(s) profile (true) or the legacy curvature-adaptive vx_max (false).
        self._use_raceline_vx_ref = bool(config.get("use_raceline_vx_ref", True))

        # Reward mode. Two schemes supported:
        #   "balanced"   — additive progress + lateral + heading + aux terms
        #                  (quadratic-inside / linear-outside lateral penalty
        #                  so the gradient never saturates past d_cap).
        #   "formula_rl" — multiplicative vx * (cos h_err − |sin h_err| − |d|/d_ref)
        #                  from Remonda et al. 2019. Speed is intrinsically
        #                  coupled to tracking accuracy: you can't earn reward
        #                  by driving fast off the line.
        self._reward_mode = str(config.get("reward_mode", "balanced")).lower()
        if self._reward_mode not in ("balanced", "formula_rl"):
            raise ValueError(
                f"reward.mode must be 'balanced' or 'formula_rl', "
                f"got {self._reward_mode!r}"
            )

        # Reward weights — see configs/default.yaml for design rationale.
        self._alpha_progress = float(config.get("alpha_progress", 1.0))
        self._alpha_d = float(config.get("alpha_d", 0.3))
        self._d_cap = float(config.get("d_cap", 3.0))
        # d_ref = normalization distance for the formula_rl multiplicative term:
        # at |d|=d_ref the lateral cost fully cancels the cos(h_err)=1 speed bonus.
        self._d_ref = float(config.get("d_ref", 1.0))
        self._alpha_h = float(config.get("alpha_h", 0.5))
        self._alpha_a = float(config.get("alpha_a", 0.01))
        self._alpha_v = float(config.get("alpha_v", 0.0))
        self._alpha_slip = float(config.get("alpha_slip", 2.0))
        self._alpha_steerspeed = float(config.get("alpha_steerspeed", 0.0))
        self._alpha_underspeed = float(config.get("alpha_underspeed", 0.0))
        self._underspeed_tolerance = float(config.get("underspeed_tolerance", 2.0))

        # Speed ceiling: combine the raceline vx profile with a conservative
        # curvature target v_ref = sqrt(a_lat_max / |kappa|). The latter is
        # what teaches the agent to brake before sharp turns even when the
        # telemetry speed profile is nearly flat.
        self._vx_max = float(config.get("vx_max", 35.0))
        self._alpha_overspeed = float(config.get("alpha_overspeed", 1.0))
        self._use_curvature_vxmax = bool(config.get("use_curvature_vxmax", True))
        self._use_curvature_vx_ref = bool(config.get("use_curvature_vx_ref", True))
        self._use_lookahead_overspeed = bool(config.get("use_lookahead_overspeed", True))
        self._overspeed_lookahead_max_distance = float(
            config.get("overspeed_lookahead_max_distance", 160.0)
        )
        self._braking_decel = float(config.get("braking_decel", 4.0))
        self._a_lat_max = float(config.get("a_lat_max", 15.0))
        self._kappa_min = float(config.get("kappa_min", 5e-3))

        self.crash_penalty = float(config.get("crash_penalty", 100.0))
        self.stuck_vx_threshold = float(config.get("stuck_vx_threshold", 1.0))
        self.stuck_steps_threshold = int(config.get("stuck_steps_threshold", 50))
        self.stuck_penalty = float(config.get("stuck_penalty", 50.0))
        self.lap_bonus = float(config.get("lap_bonus", 0.0))

        low = np.array(config.get("action_low", [-0.5, -0.05]), dtype=np.float32)
        high = np.array(config.get("action_high", [0.5, 0.05]), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._normalize_observations = bool(config.get("normalize_observations", True))
        self._obs_clip = float(config.get("obs_clip", 10.0))
        self._obs_v_scale = float(config.get("obs_v_scale", max(self._vx_max, 1.0)))
        self._obs_yaw_rate_scale = float(config.get("obs_yaw_rate_scale", 2.0))
        self._obs_d_scale = float(config.get("obs_d_scale", max(self._d_cap, 10.0)))
        self._obs_curvature_scale = float(config.get("obs_curvature_scale", 0.02))

        obs_dim = self._compute_obs_dim()
        self._obs_scale = self._build_obs_scale()
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

    # ------------------------------------------------------------------
    # Reference-curve helpers (raceline if loaded, else centerline)
    # ------------------------------------------------------------------

    def _ref_cartesian_to_frenet(
        self, x: float, y: float
    ) -> Tuple[float, float, float]:
        """Project onto the active reference curve (raceline or centerline)."""
        if self._use_raceline and self.raceline is not None:
            return self.raceline.cartesian_to_frenet(x, y)
        return self.track.cartesian_to_frenet(x, y)

    def _ref_update_progress(
        self, s_prev: float, s_new: float
    ) -> Tuple[float, bool]:
        if self._use_raceline and self.raceline is not None:
            return self.raceline.update_progress(s_prev, s_new)
        return self.track.update_progress(s_prev, s_new)

    def _ref_total_length(self) -> float:
        if self._use_raceline and self.raceline is not None:
            return self.raceline.total_length
        return self.track.total_length

    def _ref_lookahead_curvatures(self, s: float) -> np.ndarray:
        if self._use_raceline and self.raceline is not None:
            return self.raceline.lookahead_curvatures(s, self.curvature_lookaheads)
        return self.track.lookahead_curvatures(s, self.curvature_lookaheads)

    def _curvature_vx_from_curvature(self, curvatures: np.ndarray) -> np.ndarray:
        """Conservative speed target from local curvature samples."""
        kappa = np.maximum(np.abs(curvatures), self._kappa_min)
        return np.minimum(np.sqrt(self._a_lat_max / kappa), self._vx_max)

    def _vx_ref_at(self, s: float) -> float:
        """Speed target at a specific arc length."""
        refs = []
        if self._use_raceline_vx_ref and self.raceline is not None:
            refs.append(float(self.raceline.vx_at(s)))
        if self._use_curvature_vx_ref:
            if self._use_raceline and self.raceline is not None:
                curv = float(self.raceline.curvature_at(s))
            else:
                curv = float(self.track.curvature_at(s))
            refs.append(float(self._curvature_vx_from_curvature(np.array([curv]))[0]))
        if not refs:
            refs.append(self._vx_max)
        return min(refs)

    def _ref_lookahead_vx(self, s: float) -> np.ndarray:
        """Upcoming speed targets, including curvature-limited corner speeds."""
        refs = np.full(self.n_curvature_points, self._vx_max, dtype=np.float64)
        if self._use_raceline_vx_ref and self.raceline is not None:
            refs = np.minimum(
                refs,
                self.raceline.lookahead_vx_refs(s, self.curvature_lookaheads),
            )
        if self._use_curvature_vx_ref:
            refs = np.minimum(
                refs,
                self._curvature_vx_from_curvature(self._ref_lookahead_curvatures(s)),
            )
        return refs

    def _compute_obs_dim(self) -> int:
        # Base: 3 vel + (d, heading_error, progress_speed) + n curvatures +
        #       (throttle_fb, steering_fb, prev_throttle_cmd, prev_steering_cmd)
        base = 10 + self.n_curvature_points
        # When raceline is active, add: vx_error_vs_ref (=vx - vx_ref(s))
        # and n_curvature_points reference-speed lookaheads so the policy can
        # *see* where it should slow down, not just be penalized for going fast.
        if self._use_raceline:
            base += 1 + self.n_curvature_points
        return base

    def _build_obs_scale(self) -> np.ndarray:
        """Fixed physical-unit scales so the policy sees O(1) inputs."""
        action_scale = np.maximum(
            np.maximum(np.abs(self.action_space.low), np.abs(self.action_space.high)),
            1e-6,
        )
        throttle_fb_scale = max(
            abs(self.dynamics.throttle_fb_range[0]),
            abs(self.dynamics.throttle_fb_range[1]),
            1e-6,
        )
        steering_fb_scale = max(
            abs(self.dynamics.steering_fb_range[0]),
            abs(self.dynamics.steering_fb_range[1]),
            1e-6,
        )
        scale = [
            self._obs_v_scale,
            self._obs_v_scale,
            self._obs_yaw_rate_scale,
            self._obs_d_scale,
            np.pi,
            self._obs_v_scale,
        ]
        scale.extend([self._obs_curvature_scale] * self.n_curvature_points)
        scale.extend(
            [
                throttle_fb_scale,
                steering_fb_scale,
                float(action_scale[0]),
                float(action_scale[1]),
            ]
        )
        if self._use_raceline:
            scale.append(self._obs_v_scale)
            scale.extend([self._obs_v_scale] * self.n_curvature_points)
        return np.asarray(scale, dtype=np.float32)

    def _sync_feedback_from_dynamics(self) -> None:
        assert self.dynamics.history is not None
        row = self.dynamics.history[-1]
        self._throttle_fb = float(row[3])
        self._steering_fb = float(row[4])

    def _get_obs(self) -> np.ndarray:
        curv = self._ref_lookahead_curvatures(self._s)
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
        if self._use_raceline:
            # vx_error_vs_ref + upcoming reference-speed lookaheads give the
            # policy a direct view of "am I too fast / too slow for what's
            # ahead", instead of inferring it only from the overspeed penalty.
            vx_ref_now = self._braking_vx_ref(self._s)
            vx_ref_ahead = self._ref_lookahead_vx(self._s).astype(np.float32)
            parts.append(
                np.array(
                    [self._vx - vx_ref_now], dtype=np.float32
                )
            )
            parts.append(vx_ref_ahead)
        obs = np.concatenate(parts, axis=0).astype(np.float32)
        if self._normalize_observations:
            obs = obs / self._obs_scale
            if self._obs_clip > 0.0:
                obs = np.clip(obs, -self._obs_clip, self._obs_clip)
        return obs.astype(np.float32)

    def _current_vx_ref(self) -> float:
        """Speed reference at the agent's current arc length."""
        if not self._use_curvature_vxmax and not self._use_raceline_vx_ref:
            return self._vx_max
        return self._vx_ref_at(self._s)

    def _braking_vx_ref(self, s: float) -> float:
        """Most restrictive current/upcoming speed target for braking reward."""
        current_ref = self._vx_ref_at(s)
        if not self._use_lookahead_overspeed:
            return current_ref
        lookahead_mask = self.curvature_lookaheads <= self._overspeed_lookahead_max_distance
        if not np.any(lookahead_mask):
            return current_ref
        lookahead_refs = self._ref_lookahead_vx(s)[lookahead_mask]
        if lookahead_refs.size == 0:
            return current_ref
        lookahead_distances = self.curvature_lookaheads[lookahead_mask]
        braking_refs = np.sqrt(
            np.maximum(
                lookahead_refs * lookahead_refs
                + 2.0 * self._braking_decel * lookahead_distances,
                0.0,
            )
        )
        return min(current_ref, float(np.min(braking_refs)))

    def _compute_reward(
        self,
        vx: float,
        vy: float,
        delta_s: float,
        d: float,
        heading_error: float,
        action: np.ndarray,
        vx_ref: float,
    ) -> float:
        # --- Auxiliary terms (shared across modes) --------------------------
        beta = float(np.arctan2(vy, max(abs(vx), 1e-3)))
        r_slip = -self._alpha_slip * (beta * beta)
        r_smooth = -self._alpha_a * float(action[0] ** 2 + action[1] ** 2)
        r_steerspeed = -self._alpha_steerspeed * vx * (
            self._steering_fb * self._steering_fb
        )
        r_overspeed = 0.0
        if vx_ref > 0.0 and vx > vx_ref:
            r_overspeed = -self._alpha_overspeed * (vx - vx_ref) ** 2
        r_underspeed = 0.0
        if self._alpha_underspeed > 0.0 and vx_ref > 0.0:
            speed_shortfall = max(vx_ref - vx - self._underspeed_tolerance, 0.0)
            r_underspeed = -self._alpha_underspeed * (speed_shortfall ** 2)

        if self._reward_mode == "formula_rl":
            # Paper's multiplicative coupling (Remonda et al. 2019), ADAPTED
            # with two fixes for the failure mode we actually saw:
            #
            #   (1) CLAMP |d|/d_ref at 1.0. The original unbounded form made
            #       r → -vx * |d|/d_ref at large |d|, which meant the optimal
            #       response to "being off-line" was to BRAKE (less negative)
            #       rather than steer back. Clamping bounds the negative
            #       contribution at -vx and removes that perverse incentive.
            #
            #   (2) Add an ADDITIVE progress term α_progress·(Δs/Ts) and a
            #       speed-independent lateral penalty α_d·d_penalty.
            #       Additive progress keeps "accelerate along the line" a
            #       strict gradient signal even when the coupling term is
            #       zeroed out by a bad heading/offset during exploration
            #       — without this, the agent learns to sit at initial_speed
            #       going straight because any transient wobble costs vx.
            #       The additive lateral (independent of vx) closes the
            #       brake-to-escape loophole entirely: slowing down no
            #       longer reduces the cost of being off-line.
            abs_h = abs(heading_error)
            lateral_frac = min(abs(d) / max(self._d_ref, 1e-6), 1.0)
            r_coupling = vx * (
                float(np.cos(heading_error))
                - float(np.sin(abs_h))
                - lateral_frac
            )
            r_progress = self._alpha_progress * (delta_s / self.dynamics.Ts)

            abs_d = abs(d)
            if abs_d <= self._d_cap:
                d_penalty = abs_d * abs_d
            else:
                d_penalty = (
                    self._d_cap * self._d_cap
                    + 2.0 * self._d_cap * (abs_d - self._d_cap)
                )
            r_lateral = -self._alpha_d * d_penalty

            return float(
                r_coupling
                + r_progress
                + r_lateral
                + r_slip
                + r_smooth
                + r_steerspeed
                + r_overspeed
                + r_underspeed
            )

        # --- "balanced" mode (default) --------------------------------------
        # Dominant progress term: Δs/Ts ~ projected forward speed along track.
        r_progress = self._alpha_progress * (delta_s / self.dynamics.Ts)

        # Quadratic-inside / linear-outside lateral penalty. Inside [−d_cap,
        # d_cap] it is α·d² (gentle, smooth near zero). Beyond d_cap it
        # switches to a linear tail with slope 2·α·d_cap so the gradient
        # is always ≥ 2·α·d_cap and never vanishes — this fixes the bug
        # where a saturated cap at |d|>cap left the agent with no incentive
        # to return to the line.
        abs_d = abs(d)
        if abs_d <= self._d_cap:
            d_penalty = abs_d * abs_d
        else:
            d_penalty = (
                self._d_cap * self._d_cap
                + 2.0 * self._d_cap * (abs_d - self._d_cap)
            )
        r_lateral = -self._alpha_d * d_penalty

        r_heading = -self._alpha_h * (heading_error * heading_error)
        r_speed = self._alpha_v * vx

        return float(
            r_progress
            + r_lateral
            + r_heading
            + r_slip
            + r_smooth
            + r_steerspeed
            + r_speed
            + r_overspeed
            + r_underspeed
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        # Spawn on the active reference (raceline when enabled, else centerline).
        # Previously the car was always placed on the geometric centerline with
        # the centerline heading, even when the reward Frenet frame was the
        # raceline. On a banked oval, centerline ≠ raceline by several metres,
        # so every episode started with |d| >> d_ref and h_err != 0 — the
        # formula_rl reward was already deeply negative before the agent acted,
        # and most rollouts were short crashes of "bad state → worse state".
        # Spawning on the raceline puts the agent in the regime where the
        # reward gradient actually pushes toward "stay on line, accelerate".
        if self._use_raceline and self.raceline is not None:
            ref_xy = self.raceline.xy
            ref_heading = self.raceline.heading
        else:
            ref_xy = self.track.centerline
            ref_heading = self.track.headings
        n = len(ref_xy)

        if "start_idx" in options and options["start_idx"] is not None:
            start_idx = int(options["start_idx"]) % n
        elif options.get("randomize_start", False):
            start_idx = int(self.np_random.integers(0, n))
        else:
            start_idx = 0

        x0 = float(ref_xy[start_idx, 0])
        y0 = float(ref_xy[start_idx, 1])
        heading0 = float(ref_heading[start_idx])

        self.pose.reset(x0, y0, heading0)

        # prev_s must be in the *reference* arc-length frame (raceline or
        # centerline). Use the active reference's projection of the start
        # pose so the first delta_s of the next step isn't a huge jump.
        self._x = self.pose.x
        self._y = self.pose.y
        self._heading = self.pose.heading
        self._s, self._d, self._track_heading = self._ref_cartesian_to_frenet(
            self._x, self._y
        )
        initial_vx = float(options.get("initial_speed", self.initial_speed))
        if self._initial_speed_from_reference and "initial_speed" not in options:
            initial_vx = self._initial_speed_scale * self._braking_vx_ref(self._s)
            initial_vx = float(
                np.clip(initial_vx, self._initial_speed_min, self._initial_speed_max)
            )
        self.dynamics.reset(
            vx=initial_vx,
            vy=0.0,
            yaw_rate=0.0,
            throttle_fb=float(options.get("initial_throttle_fb", 0.0)),
            steering_fb=float(options.get("initial_steering_fb", 0.0)),
        )
        self.prev_s = float(self._s)
        self.step_count = 0
        self.lap_count = 0
        self._slow_steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        self._heading_error = _normalize_angle(self._heading - self._track_heading)
        self._vx = initial_vx
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
        s, d, track_heading = self._ref_cartesian_to_frenet(x, y)
        heading_error = _normalize_angle(heading - track_heading)
        delta_s, lap_complete = self._ref_update_progress(self.prev_s, s)
        self.prev_s = s
        self.step_count += 1
        self._last_action = action.copy()

        # Update cached frenet state + feedback before reward so curvature
        # lookahead uses the *current* s and the steering-speed coupling term
        # sees the *current* steering_fb.
        self._s = s
        self._sync_feedback_from_dynamics()
        vx_ref = self._current_vx_ref()
        vx_ref_brake = self._braking_vx_ref(s)
        reward = self._compute_reward(
            vx, vy, delta_s, d, heading_error, action, vx_ref_brake
        )

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
            "vx_ref": float(vx_ref),
            "vx_ref_brake": float(vx_ref_brake),
            "reference": "raceline" if self._use_raceline else "centerline",
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None
