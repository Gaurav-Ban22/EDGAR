#!/usr/bin/env python3
"""
Phase 8.1: run N evaluation episodes (deterministic policy), print aggregate stats.

From ``EDGAR/``::

    python -m deep_dynamics.rl.scripts.evaluate \\
        --config deep_dynamics/rl/configs/default.yaml \\
        --checkpoint deep_dynamics/rl_runs/checkpoints/sac_latest.pt \\
        --episodes 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from deep_dynamics.rl.agent.sac import SAC
from deep_dynamics.rl.configs import load_rl_config
from deep_dynamics.rl.environment.racing_env import RacingEnv


@dataclass
class EpisodeEval:
    return_sum: float
    steps: int
    laps: int
    duration_s: float
    mean_vx: float
    max_abs_d: float
    crashed: bool
    stuck: bool
    timeout: bool


def run_eval_episode_detailed(
    env: RacingEnv,
    agent: SAC,
    *,
    seed: int,
    randomize_start: bool = False,
) -> EpisodeEval:
    opts = {"randomize_start": True} if randomize_start else None
    obs, _ = env.reset(seed=seed, options=opts)
    ts = float(env.dynamics.Ts)
    vx_list: List[float] = []
    d_list: List[float] = []
    total_r = 0.0
    steps = 0
    crashed = stuck = timeout = False

    while True:
        action = agent.select_action(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += float(r)
        steps += 1
        vx_list.append(float(info["vx"]))
        d_list.append(abs(float(info["d"])))
        reason = info.get("termination_reason")
        if term or trunc:
            if reason == "crash":
                crashed = True
            elif reason == "stuck":
                stuck = True
            elif reason == "timeout":
                timeout = True
            break

    laps = int(info.get("lap_count", 0))
    duration_s = steps * ts
    mean_vx = float(np.mean(vx_list)) if vx_list else 0.0
    max_abs_d = float(np.max(d_list)) if d_list else 0.0

    return EpisodeEval(
        return_sum=total_r,
        steps=steps,
        laps=laps,
        duration_s=duration_s,
        mean_vx=mean_vx,
        max_abs_d=max_abs_d,
        crashed=crashed,
        stuck=stuck,
        timeout=timeout,
    )


def main() -> int:
    here = Path(__file__).resolve()
    default_cfg = here.parent.parent / "configs" / "default.yaml"
    p = argparse.ArgumentParser(description="Evaluate SAC policy on RacingEnv")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to sac_*.pt from training",
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--randomize-start",
        action="store_true",
        help="Random centerline start each episode (otherwise same start → identical eval)",
    )
    args = p.parse_args()

    cfg = load_rl_config(args.config.expanduser().resolve())
    env = RacingEnv(cfg.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(
        obs_dim,
        action_dim,
        env.action_space.low,
        env.action_space.high,
        config=cfg.sac_agent,
        device=cfg.device,
    )
    agent.load(args.checkpoint.expanduser().resolve(), load_optimizers=False)

    rows: List[EpisodeEval] = []
    for i in range(args.episodes):
        rows.append(
            run_eval_episode_detailed(
                env,
                agent,
                seed=args.seed + i,
                randomize_start=args.randomize_start,
            )
        )

    n_crash = sum(1 for e in rows if e.crashed)
    n_stuck = sum(1 for e in rows if e.stuck)
    n_timeout = sum(1 for e in rows if e.timeout)
    returns = [e.return_sum for e in rows]
    lap_times = [
        e.duration_s / max(e.laps, 1) if e.laps > 0 else float("nan") for e in rows
    ]
    lap_times_finite = [x for x in lap_times if not np.isnan(x)]

    print(f"checkpoint: {args.checkpoint}")
    print(f"episodes:   {args.episodes}")
    std_ret = float(np.std(returns, ddof=0)) if len(returns) > 1 else 0.0
    print(
        f"return:     mean={float(np.mean(returns)):.2f}  "
        f"std={std_ret:.2f}  "
        f"min={min(returns):.2f}  max={max(returns):.2f}"
    )
    print(
        f"steps:      mean={float(np.mean([e.steps for e in rows])):.1f}  "
        f"laps total={sum(e.laps for e in rows)}"
    )
    if lap_times_finite:
        print(
            f"lap time:   mean={float(np.mean(lap_times_finite)):.2f}s  "
            f"(over episodes with ≥1 lap, n={len(lap_times_finite)})"
        )
    else:
        print("lap time:   n/a (no episode completed ≥1 lap)")
    print(
        f"mean |v_x|: mean over eps = "
        f"{float(np.mean([e.mean_vx for e in rows])):.2f} m/s"
    )
    print(
        f"max |d|:    mean over eps = "
        f"{float(np.mean([e.max_abs_d for e in rows])):.2f} m  "
        f"(worst single ep {max(e.max_abs_d for e in rows):.2f} m)"
    )
    print(
        f"terminations: crash={n_crash}  stuck={n_stuck}  timeout={n_timeout}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
