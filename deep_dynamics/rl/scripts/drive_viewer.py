#!/usr/bin/env python3
"""
Interactive top-down view of the SAC policy driving the sim.

* Track boundaries + centerline (and raceline if the env uses it).
* Live car position, heading arrow, trajectory trace.
* **Restart** button: new random spawn along the centerline (same as training/eval).

From ``EDGAR/``::

    python -m deep_dynamics.rl.scripts.drive_viewer \\
        --config deep_dynamics/rl/configs/default.yaml

    python -m deep_dynamics.rl.scripts.drive_viewer \\
        --config deep_dynamics/rl/configs/default.yaml \\
        --checkpoint deep_dynamics/rl_runs/checkpoints/sac_latest.pt

    Old 20-dim checkpoints::

        python -m deep_dynamics.rl.scripts.drive_viewer --legacy-policy ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from deep_dynamics.rl.agent.sac import SAC
from deep_dynamics.rl.configs import load_rl_config
from deep_dynamics.rl.environment.racing_env import RacingEnv


def _heading_arrow_xy(
    x: float, y: float, heading: float, length: float = 6.0
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array([x, x + length * float(np.cos(heading))], dtype=np.float64),
        np.array([y, y + length * float(np.sin(heading))], dtype=np.float64),
    )


class DriveViewer:
    def __init__(
        self,
        env: RacingEnv,
        agent: SAC,
        *,
        arrow_length: float,
        steps_per_frame: int,
        frame_interval_ms: int,
    ) -> None:
        self.env = env
        self.agent = agent
        self.arrow_length = arrow_length
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.frame_interval_ms = max(1, int(frame_interval_ms))

        self.fig, self.ax = plt.subplots(figsize=(11, 10))
        self.fig.subplots_adjust(bottom=0.11)
        self.fig.canvas.manager.set_window_title("RL drive viewer")

        env.track.plot(
            ax=self.ax,
            show_centerline=True,
            show_boundaries=True,
        )
        if getattr(env, "_use_raceline", False) and env.raceline is not None:
            xy = env.raceline.xy
            self.ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="tab:orange",
                lw=0.7,
                alpha=0.45,
                label="raceline",
            )
        self.ax.legend(loc="upper right", fontsize=8)

        (self.traj_line,) = self.ax.plot([], [], color="tab:blue", lw=1.2, alpha=0.75, zorder=4)
        (self.car_dot,) = self.ax.plot([], [], "o", color="tab:red", ms=9, zorder=6)
        (self.heading_line,) = self.ax.plot([], [], "-", color="darkred", lw=2.0, zorder=5)

        self.status_text = self.fig.text(
            0.02,
            0.98,
            "",
            transform=self.fig.transFigure,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        ax_btn = self.fig.add_axes((0.38, 0.02, 0.24, 0.045))
        self.btn = Button(ax_btn, "Restart (random spawn)")
        self.btn.on_clicked(self._on_restart)

        self.obs: Optional[np.ndarray] = None
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.episode_done = True
        self.pending_restart = True
        self._last_reason = ""
        self._step_idx = 0

        self.anim: Optional[FuncAnimation] = None

    def _on_restart(self, _event: Any) -> None:
        self.pending_restart = True
        self.episode_done = False

    def _reset_episode(self) -> None:
        seed = int(np.random.randint(0, 2**31 - 1))
        self.obs, _ = self.env.reset(
            seed=seed,
            options={"randomize_start": True},
        )
        self.xs = [float(self.env.pose.x)]
        self.ys = [float(self.env.pose.y)]
        self.episode_done = False
        self._step_idx = 0
        self._last_reason = "Driving…"
        self._sync_artists_from_pose()

    def _sync_artists_from_pose(self) -> None:
        x, y = float(self.env.pose.x), float(self.env.pose.y)
        h = float(self.env.pose.heading)
        xa, ya = _heading_arrow_xy(x, y, h, self.arrow_length)
        self.car_dot.set_data([x], [y])
        self.heading_line.set_data(xa, ya)
        self.traj_line.set_data(self.xs, self.ys)

    def _set_status(self, extra: str = "") -> None:
        vx = float(getattr(self.env, "_vx", 0.0))
        d = float(getattr(self.env, "_d", 0.0))
        ref = "raceline" if getattr(self.env, "_use_raceline", False) else "centerline"
        self.status_text.set_text(
            f"{self._last_reason}  |  step {self._step_idx}  |  vx={vx:5.1f} m/s  "
            f"d={d:+6.2f} m  |  ref={ref}\n{extra}"
        )

    def _update(self, _frame: int) -> Tuple[Any, ...]:
        if self.pending_restart:
            self._reset_episode()
            self.pending_restart = False
            self._set_status()
            return self.traj_line, self.car_dot, self.heading_line

        if self.episode_done or self.obs is None:
            self._set_status("(episode ended — click Restart)")
            return self.traj_line, self.car_dot, self.heading_line

        for _ in range(self.steps_per_frame):
            action = self.agent.select_action(self.obs, deterministic=True)
            self.obs, _r, term, trunc, info = self.env.step(action)
            self._step_idx += 1
            self.xs.append(float(info["x"]))
            self.ys.append(float(info["y"]))
            if term or trunc:
                reason = info.get("termination_reason") or (
                    "timeout" if trunc else "done"
                )
                self._last_reason = f"Ended: {reason}"
                self.episode_done = True
                break

        self._sync_artists_from_pose()
        self._set_status()
        return self.traj_line, self.car_dot, self.heading_line

    def run(self) -> None:
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=self.frame_interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def main() -> int:
    here = Path(__file__).resolve()
    default_cfg = here.parent.parent / "configs" / "default.yaml"
    p = argparse.ArgumentParser(
        description="Interactive matplotlib viewer: SAC policy driving with random restarts"
    )
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="SAC weights (default: training.sac_policy_checkpoint from YAML)",
    )
    p.add_argument(
        "--legacy-policy",
        action="store_true",
        help="20-dim obs (disable raceline) for old checkpoints",
    )
    p.add_argument(
        "--arrow-length",
        type=float,
        default=6.0,
        help="Metres — body heading arrow on the map",
    )
    p.add_argument(
        "--steps-per-frame",
        type=int,
        default=1,
        help="Sim steps each animation frame (>1 speeds up playback)",
    )
    p.add_argument(
        "--frame-ms",
        type=int,
        default=None,
        help="Milliseconds between frames (default: model Ts * 1000)",
    )
    args = p.parse_args()

    env_overrides = {"use_raceline": False} if args.legacy_policy else None
    cfg = load_rl_config(
        args.config.expanduser().resolve(), env_overrides=env_overrides
    )

    ckpt = args.checkpoint
    if ckpt is None:
        default_ckpt = cfg.training.get("sac_policy_checkpoint")
        if not default_ckpt:
            p.error("Pass --checkpoint or set training.sac_policy_checkpoint in YAML")
        ckpt = Path(default_ckpt)
    ckpt = ckpt.expanduser().resolve()
    if not ckpt.is_file():
        p.error(f"checkpoint not found: {ckpt}")

    env = RacingEnv(cfg.env)
    agent = SAC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        config=cfg.sac_agent,
        device=cfg.device,
    )
    agent.load(ckpt, load_optimizers=False)

    ts_ms = int(round(float(env.dynamics.Ts) * 1000.0))
    frame_ms = int(args.frame_ms) if args.frame_ms is not None else max(ts_ms, 15)

    print(f"checkpoint: {ckpt}")
    print(f"obs_dim={env.observation_space.shape[0]}  frame_ms={frame_ms}  Ts={env.dynamics.Ts}")
    print("Close the window or Ctrl+C to quit. Use Restart for a new random spawn.")

    viewer = DriveViewer(
        env,
        agent,
        arrow_length=float(args.arrow_length),
        steps_per_frame=int(args.steps_per_frame),
        frame_interval_ms=frame_ms,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
