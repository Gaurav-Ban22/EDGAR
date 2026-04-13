#!/usr/bin/env python3
"""
Sanity checks for PoseIntegrator (implementation plan Step 4).

Run:
    python -m deep_dynamics.rl.environment.run_pose_integrator_demo
"""

from __future__ import annotations

import math
import sys

from deep_dynamics.rl.environment.pose_integrator import PoseIntegrator, wrap_to_pi


def _straight_line_constant_vx() -> None:
    """Drive straight at vx=30 m/s for 1 s → x should advance ~30 m."""
    Ts = 0.04
    n = int(round(1.0 / Ts))
    pose = PoseIntegrator(Ts)
    pose.reset(0.0, 0.0, 0.0)
    for _ in range(n):
        pose.step(30.0, 0.0, 0.0)
    assert abs(pose.y) < 1e-9, pose.y
    assert abs(pose.heading) < 1e-9, pose.heading
    err = abs(pose.x - 30.0)
    assert err < 1e-6, f"expected x≈30 after 1s, got x={pose.x} (|error|={err})"


def _constant_yaw_rate() -> None:
    """Zero translation, constant yaw_rate → heading change = ω * T."""
    Ts = 0.04
    omega = 0.5
    pose = PoseIntegrator(Ts)
    pose.reset(0.0, 0.0, 0.0)
    pose.step(0.0, 0.0, omega)
    assert abs(pose.x) < 1e-12 and abs(pose.y) < 1e-12
    assert abs(pose.heading - omega * Ts) < 1e-12


def _wrap_to_pi() -> None:
    assert abs(wrap_to_pi(3 * math.pi) - math.pi) < 1e-12
    assert abs(wrap_to_pi(-math.pi)) < 1e-12


def main() -> int:
    _straight_line_constant_vx()
    _constant_yaw_rate()
    _wrap_to_pi()
    print("pose_integrator checks: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
