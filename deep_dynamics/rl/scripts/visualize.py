#!/usr/bin/env python3
"""
Phase 8.2: one rollout + track/trajectory plot (speed-colored) + time series.

From ``EDGAR/``::

    python -m deep_dynamics.rl.scripts.visualize \\
        --config deep_dynamics/rl/configs/default.yaml \\
        --checkpoint deep_dynamics/rl_runs/checkpoints/sac_latest.pt \\
        --output /tmp/rl_viz.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deep_dynamics.rl.agent.sac import SAC
from deep_dynamics.rl.configs import load_rl_config
from deep_dynamics.rl.environment.racing_env import RacingEnv


def collect_episode(
    env: RacingEnv,
    agent: SAC,
    *,
    seed: int,
    randomize_start: bool = False,
) -> Dict[str, np.ndarray]:
    opts = {"randomize_start": True} if randomize_start else None
    obs, _ = env.reset(seed=seed, options=opts)
    xs: List[float] = []
    ys: List[float] = []
    vxs: List[float] = []
    vys: List[float] = []
    yaws: List[float] = []
    th: List[float] = []
    st: List[float] = []
    ds: List[float] = []
    rs: List[float] = []

    while True:
        action = agent.select_action(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        xs.append(float(info["x"]))
        ys.append(float(info["y"]))
        vxs.append(float(info["vx"]))
        vys.append(float(info["vy"]))
        yaws.append(float(info["yaw_rate"]))
        th.append(float(info["throttle_fb"]))
        st.append(float(info["steering_fb"]))
        ds.append(float(info["d"]))
        rs.append(float(info["step_reward"]))
        if term or trunc:
            break

    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "yaw_rate": np.array(yaws),
        "throttle_fb": np.array(th),
        "steering_fb": np.array(st),
        "d": np.array(ds),
        "reward": np.array(rs),
    }


def main() -> int:
    here = Path(__file__).resolve()
    default_cfg = here.parent.parent / "configs" / "default.yaml"
    p = argparse.ArgumentParser(description="Visualize one SAC rollout")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("rl_trajectory.png"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--randomize-start",
        action="store_true",
        help="Random start on centerline (default: fixed start for reproducible plot)",
    )
    args = p.parse_args()

    cfg = load_rl_config(args.config.expanduser().resolve())
    env = RacingEnv(cfg.env)
    agent = SAC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        config=cfg.sac_agent,
        device=cfg.device,
    )
    agent.load(args.checkpoint.expanduser().resolve(), load_optimizers=False)

    traj = collect_episode(
        env, agent, seed=args.seed, randomize_start=args.randomize_start
    )
    t_axis = np.arange(len(traj["vx"])) * float(env.dynamics.Ts)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    env.track.plot(ax=ax0, show_centerline=True, show_boundaries=True)
    sc = ax0.scatter(
        traj["x"],
        traj["y"],
        c=traj["vx"],
        cmap="viridis",
        s=8,
        zorder=5,
    )
    plt.colorbar(sc, ax=ax0, label="vx (m/s)")
    ax0.set_title("Trajectory (color = vx)")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_axis, traj["vx"], label="vx")
    ax1.plot(t_axis, traj["vy"], label="vy", alpha=0.8)
    ax1.plot(t_axis, traj["yaw_rate"], label="yaw_rate", alpha=0.8)
    ax1.set_xlabel("time (s)")
    ax1.legend(fontsize=8)
    ax1.set_title("Velocities / yaw rate")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_axis, traj["throttle_fb"], label="throttle_fb")
    ax2.plot(t_axis, traj["steering_fb"], label="steering_fb", alpha=0.8)
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    ax2.set_title("Integrated controls")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_axis, traj["d"], label="lateral offset d (m)", color="C0")
    ax3.plot(t_axis, traj["reward"], label="step reward", color="C1", alpha=0.85)
    ax3.set_xlabel("time (s)")
    ax3.legend(fontsize=8)
    ax3.set_title("Lateral offset d and reward")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
