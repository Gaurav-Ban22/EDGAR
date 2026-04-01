"""
Pose integrator: converts body-frame velocities (vx, vy, yaw_rate)
to global-frame pose (x, y, heading) via Euler integration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class PoseIntegrator:
    def __init__(self, Ts: float):
        self.Ts = Ts
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

    def reset(self, x: float, y: float, heading: float):
        self.x = x
        self.y = y
        self.heading = heading

    def step(self, vx: float, vy: float, yaw_rate: float) -> Tuple[float, float, float]:
        """Integrate body-frame velocities to global pose (Euler step)."""
        x_dot = vx * np.cos(self.heading) - vy * np.sin(self.heading)
        y_dot = vx * np.sin(self.heading) + vy * np.cos(self.heading)

        self.x += x_dot * self.Ts
        self.y += y_dot * self.Ts
        self.heading += yaw_rate * self.Ts

        return self.x, self.y, self.heading
