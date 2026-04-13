#!/usr/bin/env python3
"""
Smoke test: ``reset`` + N random actions (implementation plan Step 6).

Replace the three file arguments with **real paths** on your machine (the
unicode ellipsis ``…`` or the words ``path/to`` are not valid files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deep_dynamics.rl.environment.racing_env import RacingEnv

_PLACEHOLDER_HINTS = frozenset(
    {"…", "...", "path/to", "path/to/model.yaml", "path/to/weights.pth", "path/to/scaler.pkl"}
)


def _default_tracks_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "visualize" / "tracks"


def _require_file(label: str, path: Path) -> Path:
    s = str(path).strip()
    if s in _PLACEHOLDER_HINTS or "\u2026" in s:
        print(
            f"Invalid {label!r}: {path!s}\n"
            "  That looks like a documentation placeholder, not a real file.\n"
            "  Pass your actual model YAML, checkpoint .pth, and scaler.pkl paths.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    p = path.expanduser()
    if not p.is_file():
        print(
            f"{label}: not a file or missing: {p.resolve()}\n"
            "  Use absolute paths or paths relative to your current working directory.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return p


def main() -> int:
    epilog = r"""
Example (run from `EDGAR/` — use the same run directory that contains
`scaler.pkl` next to your `epoch_*.pth`):

  python -m deep_dynamics.rl.environment.run_racing_env_smoke \
    --model-config deep_dynamics/cfgs/model/deep_dynamics_iac.yaml \
    --checkpoint deep_dynamics/output/deep_dynamics_iac/my_experiment_name/epoch_82.pth \
    --scaler deep_dynamics/output/deep_dynamics_iac/my_experiment_name/scaler.pkl
"""
    p = argparse.ArgumentParser(
        description="RacingEnv smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--track-name", default="lvms")
    p.add_argument(
        "--tracks-dir",
        type=Path,
        default=None,
        help="Directory with *_inner_bound.csv (default: visualize/tracks)",
    )
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model_config = _require_file("--model-config", args.model_config)
    checkpoint = _require_file("--checkpoint", args.checkpoint)
    scaler = _require_file("--scaler", args.scaler)

    tracks_dir = args.tracks_dir or _default_tracks_dir()
    tracks_dir = tracks_dir.expanduser()
    if not tracks_dir.is_dir():
        print(
            f"--tracks-dir is not a directory: {tracks_dir.resolve()}\n"
            f"  Default would be: {_default_tracks_dir()}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cfg = {
        "track_dir": str(tracks_dir),
        "track_name": args.track_name,
        "model_config_path": str(model_config),
        "checkpoint_path": str(checkpoint),
        "scaler_path": str(scaler),
        "device": args.device,
    }

    env = RacingEnv(cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)

    for t in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            print(f"ended at t={t} term={term} trunc={trunc} info_keys={list(info)}")
            obs, _ = env.reset(seed=t + 1)

    print(
        f"RacingEnv smoke OK ({args.steps} steps): "
        f"obs_dim={obs.shape[0]} last_reward={r:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
