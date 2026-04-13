"""
Phase 7: load central YAML → resolved paths and component-ready dicts.

Used by ``train.py`` and any script that needs the same wiring (eval, viz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Union

import yaml


def _resolve_path(config_path: Path, p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path.resolve())
    return str((config_path.parent / path).resolve())


def _build_env_config(raw: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    d = raw["dynamics"]
    t = raw["track"]
    e = raw.get("env", {})
    a = raw.get("action", {})
    r = raw.get("reward", {})
    tr = a.get("throttle_cmd_range", [-0.5, 0.5])
    sr = a.get("steering_cmd_range", [-0.05, 0.05])
    cfg: Dict[str, Any] = {
        "model_config_path": _resolve_path(config_path, d["model_config_path"]),
        "checkpoint_path": _resolve_path(config_path, d["checkpoint_path"]),
        "scaler_path": _resolve_path(config_path, d["scaler_path"]),
        "device": d.get("device", "cpu"),
        "max_steps": e.get("max_steps", 5000),
        "initial_speed": e.get("initial_speed", 15.0),
        "n_curvature_points": e.get("n_curvature_points", 10),
        "curvature_lookahead_spacing": float(e.get("curvature_lookahead_spacing", 5.0)),
        "boundary_margin": e.get("boundary_margin", 0.0),
        "action_low": [float(tr[0]), float(sr[0])],
        "action_high": [float(tr[1]), float(sr[1])],
        "alpha_d": r.get("alpha_d", 0.1),
        "alpha_h": r.get("alpha_h", 0.5),
        "alpha_a": r.get("alpha_a", 0.01),
        "alpha_v": r.get("alpha_v", 0.0),
        "vx_max": float(r.get("vx_max", 50.0)),
        "alpha_overspeed": float(r.get("alpha_overspeed", 1.0)),
        "crash_penalty": r.get("crash_penalty", 100.0),
        "stuck_penalty": r.get("stuck_penalty", 50.0),
        "lap_bonus": r.get("lap_bonus", 0.0),
    }
    if "inner_csv" in t and "outer_csv" in t:
        cfg["inner_csv"] = _resolve_path(config_path, t["inner_csv"])
        cfg["outer_csv"] = _resolve_path(config_path, t["outer_csv"])
        if t.get("track_name") is not None:
            cfg["track_name"] = t["track_name"]
    elif "track_dir" in t and "track_name" in t:
        cfg["track_dir"] = _resolve_path(config_path, t["track_dir"])
        cfg["track_name"] = t["track_name"]
    else:
        raise ValueError(
            "config['track']: provide either (track_dir, track_name) or "
            "(inner_csv, outer_csv)"
        )
    for opt in ("stuck_vx_threshold", "stuck_steps_threshold"):
        if opt in e:
            cfg[opt] = e[opt]
    return cfg


def _build_sac_agent_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    s = raw.get("sac", {})
    out: Dict[str, Any] = {
        "lr_actor": float(s.get("lr_actor", 3e-4)),
        "lr_critic": float(s.get("lr_critic", 3e-4)),
        "lr_alpha": float(s.get("lr_alpha", 3e-4)),
        "gamma": float(s.get("gamma", 0.99)),
        "tau": float(s.get("tau", 0.005)),
        "hidden_dim": int(s.get("hidden_dim", 256)),
    }
    if s.get("target_entropy") is not None:
        out["target_entropy"] = float(s["target_entropy"])
    return out


def _build_training_runtime(
    raw: Mapping[str, Any], config_path: Path
) -> Dict[str, Any]:
    tr = raw.get("training", {})
    s = raw.get("sac", {})
    out: Dict[str, Any] = {
        "max_episodes": int(tr.get("max_episodes", 10_000)),
        "eval_every": int(tr.get("eval_every", 50)),
        "save_every": int(tr.get("save_every", 100)),
        "log_every": int(tr.get("log_every", 10)),
        "eval_seed": int(tr.get("eval_seed", 42)),
        "train_seed": tr.get("train_seed", None),
        "batch_size": int(s.get("batch_size", 256)),
        "buffer_size": int(s.get("buffer_size", 1_000_000)),
        "initial_random_steps": int(s.get("initial_random_steps", 10_000)),
        "checkpoint_dir": _resolve_path(
            config_path, str(tr.get("checkpoint_dir", "checkpoints"))
        ),
        "log_dir": _resolve_path(config_path, str(tr.get("log_dir", "logs"))),
    }
    return out


@dataclass(frozen=True)
class RLConfig:
    """Parsed ``default.yaml`` (or equivalent) with absolute paths where needed."""

    path: Path
    raw: Dict[str, Any]
    env: Dict[str, Any]
    sac_agent: Dict[str, Any]
    training: Dict[str, Any]

    @property
    def device(self) -> str:
        return str(self.env.get("device", "cpu"))


def load_rl_config(path: Union[str, Path]) -> RLConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, MutableMapping):
        raise ValueError(f"Invalid YAML root in {config_path}")

    return RLConfig(
        path=config_path,
        raw=dict(raw),
        env=_build_env_config(raw, config_path),
        sac_agent=_build_sac_agent_config(raw),
        training=_build_training_runtime(raw, config_path),
    )
