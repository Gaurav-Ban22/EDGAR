#!/usr/bin/env python3
"""
Phase 8.1: run N evaluation episodes (deterministic policy), print aggregate stats.

From ``EDGAR/`` (uses ``training.sac_policy_checkpoint`` in the YAML if you omit
``--checkpoint``)::

    python -m deep_dynamics.rl.scripts.evaluate \\
        --config deep_dynamics/rl/configs/default.yaml \\
        --episodes 20

    python -m deep_dynamics.rl.scripts.evaluate \\
        --config deep_dynamics/rl/configs/default.yaml \\
        --checkpoint deep_dynamics/rl_runs/checkpoints_old/sac_episode_200.pt \\
        --episodes 20

    Checkpoints trained on the old 20-dim observation (centerline only) must
    load with a 20-dim env — use ``--legacy-policy`` (disables raceline extra
    obs so the actor/critic shapes match).     New policies trained with raceline
    enabled use 31-dim obs; omit ``--legacy-policy`` for those.

    Defaults for ``--seed`` and ``--randomize-start`` come from ``training.eval_seed``
    and ``training.eval_randomize_start`` in the YAML (see ``default.yaml``).
    Use ``--verbose`` to print final ``(x,y,s,d)`` each episode (crash mapping).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    # Terminal state (last step before done) — for crash localization.
    final_x: float = 0.0
    final_y: float = 0.0
    final_s: float = 0.0
    final_d: float = 0.0
    final_heading_err: float = 0.0
    final_vx: float = 0.0
    termination: str = ""


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
    last_info: Dict[str, Any] = {}

    while True:
        action = agent.select_action(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        last_info = dict(info)
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

    laps = int(last_info.get("lap_count", 0))
    duration_s = steps * ts
    mean_vx = float(np.mean(vx_list)) if vx_list else 0.0
    max_abs_d = float(np.max(d_list)) if d_list else 0.0

    term = ""
    if crashed:
        term = "crash"
    elif stuck:
        term = "stuck"
    elif timeout:
        term = "timeout"

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
        final_x=float(last_info.get("x", 0.0)),
        final_y=float(last_info.get("y", 0.0)),
        final_s=float(last_info.get("s", 0.0)),
        final_d=float(last_info.get("d", 0.0)),
        final_heading_err=float(last_info.get("heading_error", 0.0)),
        final_vx=float(last_info.get("vx", 0.0)),
        termination=term,
    )


def main() -> int:
    here = Path(__file__).resolve()
    default_cfg = here.parent.parent / "configs" / "default.yaml"
    p = argparse.ArgumentParser(description="Evaluate SAC policy on RacingEnv")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to sac_*.pt (default: training.sac_policy_checkpoint from config)",
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; episode i uses seed+i (default: training.eval_seed from config)",
    )
    p.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Random centerline spawn each episode (default: training.eval_randomize_start)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print one line per episode: termination, steps, final x,y,s,d (crash map)",
    )
    p.add_argument(
        "--legacy-policy",
        action="store_true",
        help="Disable raceline (20-dim obs) so checkpoints trained before the "
        "raceline expansion still load. Required for sac_episode_*.pt when "
        "default.yaml has raceline enabled.",
    )
    args = p.parse_args()

    env_overrides = None
    if args.legacy_policy:
        env_overrides = {"use_raceline": False}
    cfg = load_rl_config(
        args.config.expanduser().resolve(), env_overrides=env_overrides
    )
    base_seed = (
        args.seed
        if args.seed is not None
        else int(cfg.training.get("eval_seed", 0))
    )
    if args.randomize_start is None:
        randomize_start = bool(cfg.training.get("eval_randomize_start", False))
    else:
        randomize_start = bool(args.randomize_start)

    ckpt = args.checkpoint
    if ckpt is None:
        default_ckpt = cfg.training.get("sac_policy_checkpoint")
        if not default_ckpt:
            p.error(
                "No --checkpoint given and config has no training.sac_policy_checkpoint"
            )
        ckpt = Path(default_ckpt)
    ckpt = ckpt.expanduser().resolve()
    if not ckpt.is_file():
        p.error(f"checkpoint not found: {ckpt}")

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
    agent.load(ckpt, load_optimizers=False)

    rows: List[EpisodeEval] = []
    for i in range(args.episodes):
        rows.append(
            run_eval_episode_detailed(
                env,
                agent,
                seed=base_seed + i,
                randomize_start=randomize_start,
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

    print(f"checkpoint: {ckpt}")
    print(
        f"obs_dim={obs_dim}  seed={base_seed}  "
        f"randomize_start={randomize_start}  "
        f"(override: --seed / --randomize-start / --no-randomize-start)"
    )
    if args.legacy_policy:
        print(
            "legacy-policy: obs_dim=20 (raceline disabled; matches "
            "pre-raceline SAC checkpoints)"
        )
    print(f"episodes:   {args.episodes}")
    if args.verbose:
        print("per-episode (terminal state at crash/timeout/stuck):")
        for i, e in enumerate(rows):
            print(
                f"  ep{i:3d}  {e.termination:7s}  steps={e.steps:4d}  "
                f"R={e.return_sum:8.1f}  "
                f"x={e.final_x:9.2f}  y={e.final_y:9.2f}  "
                f"s={e.final_s:9.1f}  d={e.final_d:+7.3f}  "
                f"h_err={e.final_heading_err:+6.3f}  vx={e.final_vx:6.2f}  "
                f"laps={e.laps}"
            )
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
