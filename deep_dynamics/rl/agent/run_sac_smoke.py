#!/usr/bin/env python3
"""
Phase 5 sanity (plan steps 7–9): buffer sample + one SAC update, finite losses.

  python -m deep_dynamics.rl.agent.run_sac_smoke
"""

from __future__ import annotations

import numpy as np

from deep_dynamics.rl.agent.replay_buffer import ReplayBuffer
from deep_dynamics.rl.agent.sac import SAC


def main() -> int:
    obs_dim, action_dim = 20, 2
    low = np.array([-0.5, -0.05], dtype=np.float32)
    high = np.array([0.5, 0.05], dtype=np.float32)
    buf = ReplayBuffer(obs_dim, action_dim, capacity=10_000)
    agent = SAC(obs_dim, action_dim, low, high, device="cpu")

    for _ in range(512):
        o = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.uniform(low, high).astype(np.float32)
        r = float(np.random.randn())
        o2 = np.random.randn(obs_dim).astype(np.float32)
        buf.add(o, a, r, o2, done=False)

    batch = buf.sample(256)
    m = agent.update(batch)
    assert all(np.isfinite(m[k]) for k in m), m

    act = agent.select_action(np.zeros(obs_dim, dtype=np.float32))
    assert act.shape == (action_dim,)

    print("SAC smoke OK:", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
