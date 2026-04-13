#!/usr/bin/env python3
"""
Phase 6: SAC training loop on ``RacingEnv``.

Phase 7: all hyperparameters and paths come from ``load_rl_config``.

From ``EDGAR/``:

    python -m deep_dynamics.rl.scripts.train --config deep_dynamics/rl/configs/default.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from deep_dynamics.rl.agent.replay_buffer import ReplayBuffer
from deep_dynamics.rl.agent.sac import SAC
from deep_dynamics.rl.configs import RLConfig, load_rl_config
from deep_dynamics.rl.environment.racing_env import RacingEnv
from deep_dynamics.rl.utils.logger import TrainLogger


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_eval_episode(
    env: RacingEnv,
    agent: SAC,
    *,
    seed: int,
) -> Tuple[float, int, float, int]:
    """Return (episode_return, steps, max_vx, terminal_lap_count)."""
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    steps = 0
    max_vx = 0.0
    lap_count = 0
    while True:
        action = agent.select_action(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += float(r)
        steps += 1
        max_vx = max(max_vx, float(info.get("vx", 0.0)))
        lap_count = int(info.get("lap_count", lap_count))
        if term or trunc:
            break
    return total_r, steps, max_vx, lap_count


def train(cfg: RLConfig, *, max_episodes_override: Optional[int] = None) -> None:
    tr = dict(cfg.training)
    if max_episodes_override is not None:
        tr["max_episodes"] = int(max_episodes_override)
    eval_every = tr["eval_every"]
    save_every = tr["save_every"]
    log_every = tr["log_every"]
    eval_seed = tr["eval_seed"]
    train_seed = tr["train_seed"]
    batch_size = tr["batch_size"]
    buffer_size = tr["buffer_size"]
    initial_random_steps = tr["initial_random_steps"]

    ckpt_root = Path(tr["checkpoint_dir"])
    log_root = Path(tr["log_dir"])
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    if train_seed is not None:
        _set_global_seeds(int(train_seed))

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
    buffer = ReplayBuffer(obs_dim, action_dim, capacity=buffer_size)
    logger = TrainLogger(log_root)

    total_steps = 0
    print(
        f"Config: {cfg.path}\n"
        f"Training: max_episodes={int(tr['max_episodes'])}  "
        f"obs_dim={obs_dim} action_dim={action_dim} "
        f"buffer={buffer_size} random_steps={initial_random_steps}"
    )

    for episode in range(1, int(tr["max_episodes"]) + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        max_vx = 0.0
        lap_count = 0
        ep_len = 0

        while True:
            if total_steps < initial_random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, float(reward), next_obs, done)

            if total_steps >= initial_random_steps and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                metrics = agent.update(batch)
                if log_every > 0 and total_steps % log_every == 0:
                    logger.log_scalars("train", metrics, total_steps)

            obs = next_obs
            episode_reward += float(reward)
            ep_len += 1
            total_steps += 1
            max_vx = max(max_vx, float(info.get("vx", 0.0)))
            lap_count = int(info.get("lap_count", lap_count))

            if done:
                break

        row = {
            "episode": episode,
            "reward": round(episode_reward, 4),
            "length": ep_len,
            "lap_count": lap_count,
            "max_vx": round(max_vx, 4),
            "total_steps": total_steps,
        }
        logger.log_scalars(
            "episode",
            {k: float(v) for k, v in row.items() if k != "episode"},
            episode,
        )
        logger.log_episode_csv(row)
        print(
            f"ep {episode:5d}  steps {total_steps:7d}  R {episode_reward:10.2f}  "
            f"len {ep_len:4d}  laps {lap_count}  max_vx {max_vx:.2f}"
        )

        if eval_every > 0 and episode % eval_every == 0:
            er, es, emv, elaps = run_eval_episode(env, agent, seed=eval_seed)
            logger.log_scalars(
                "eval",
                {
                    "return": er,
                    "length": float(es),
                    "max_vx": emv,
                    "lap_count": float(elaps),
                },
                episode,
            )
            print(
                f"  eval  R {er:.2f}  len {es}  max_vx {emv:.2f}  laps {elaps}"
            )

        if save_every > 0 and episode % save_every == 0:
            agent.save(ckpt_root / f"sac_episode_{episode}.pt")
            agent.save(ckpt_root / "sac_latest.pt")

    logger.close()
    agent.save(ckpt_root / "sac_final.pt")
    print("Done.")


def main() -> int:
    here = Path(__file__).resolve()
    default_cfg = here.parent.parent / "configs" / "default.yaml"
    p = argparse.ArgumentParser(description="Train SAC on RacingEnv")
    p.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="YAML config path",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Override training.max_episodes (for shorter dev runs)",
    )
    args = p.parse_args()
    cfg = load_rl_config(args.config.expanduser().resolve())
    train(cfg, max_episodes_override=args.max_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
