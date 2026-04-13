"""
Pose integrator: converts body-frame velocities (vx, vy, yaw_rate)
to global-frame pose (x, y, heading) via Euler integration.

Phase 3 (rl_implementation_plan.md): same Ts as the dynamics model; body axes
are x forward, y left; heading is global yaw (rad, CCW positive).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap *angle* to (−π, π] (handy for heading-error terms in the env)."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class PoseIntegrator:
    def __init__(self, Ts: float, *, wrap_heading: bool = False):
        """
        Parameters
        ----------
        Ts
            Simulation timestep (s); must match Deep Dynamics ``timestep``.
        wrap_heading
            If True, ``heading`` is wrapped to (−π, π] after each ``step``.
        """
        self.Ts = float(Ts)
        self.wrap_heading = wrap_heading
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

    def reset(self, x: float, y: float, heading: float):
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        if self.wrap_heading:
            self.heading = wrap_to_pi(self.heading)

    def body_to_global_velocity(self, vx: float, vy: float) -> Tuple[float, float]:
        """Body-frame (vx, vy) → global (ẋ, ẏ) at the current ``heading``."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        x_dot = vx * c - vy * s
        y_dot = vx * s + vy * c
        return float(x_dot), float(y_dot)

    def step(self, vx: float, vy: float, yaw_rate: float) -> Tuple[float, float, float]:
        """Integrate one Euler step; returns updated ``(x, y, heading)``.

        Uses the heading at the start of the step to map body velocity into
        global frame, then updates position and yaw (equivalent to the plan’s
        order when derivatives are fixed at the beginning of the interval).
        """
        x_dot, y_dot = self.body_to_global_velocity(vx, vy)

        self.x += x_dot * self.Ts
        self.y += y_dot * self.Ts
        self.heading += float(yaw_rate) * self.Ts

        if self.wrap_heading:
            self.heading = wrap_to_pi(self.heading)

        return self.x, self.y, self.heading
