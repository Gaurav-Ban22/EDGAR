"""
Phase 7: load central YAML → resolved paths and component-ready dicts.

Used by ``train.py`` and any script that needs the same wiring (eval, viz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

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
    rl = raw.get("raceline", {})
    tr = a.get("throttle_cmd_range", [-0.5, 0.5])
    sr = a.get("steering_cmd_range", [-0.05, 0.05])
    cfg: Dict[str, Any] = {
        "model_config_path": _resolve_path(config_path, d["model_config_path"]),
        "checkpoint_path": _resolve_path(config_path, d["checkpoint_path"]),
        "scaler_path": _resolve_path(config_path, d["scaler_path"]),
        "device": d.get("device", "cpu"),
        "max_steps": e.get("max_steps", 5000),
        "initial_speed": e.get("initial_speed", 15.0),
        "initial_speed_from_reference": bool(
            e.get("initial_speed_from_reference", False)
        ),
        "initial_speed_scale": float(e.get("initial_speed_scale", 1.0)),
        "initial_speed_min": float(e.get("initial_speed_min", 5.0)),
        "initial_speed_max": float(e.get("initial_speed_max", r.get("vx_max", 75.0))),
        "n_curvature_points": e.get("n_curvature_points", 10),
        "curvature_lookahead_spacing": float(e.get("curvature_lookahead_spacing", 5.0)),
        "boundary_margin": e.get("boundary_margin", 0.0),
        "normalize_observations": bool(e.get("normalize_observations", True)),
        "obs_clip": float(e.get("obs_clip", 10.0)),
        "action_low": [float(tr[0]), float(sr[0])],
        "action_high": [float(tr[1]), float(sr[1])],
        "throttle_fb_range": [
            float(a.get("throttle_fb_range", [-0.1, 0.5])[0]),
            float(a.get("throttle_fb_range", [-0.1, 0.5])[1]),
        ],
        "steering_fb_range": [
            float(a.get("steering_fb_range", [-0.02, 0.025])[0]),
            float(a.get("steering_fb_range", [-0.02, 0.025])[1]),
        ],
        "reward_mode": str(r.get("mode", "balanced")),
        "alpha_progress": float(r.get("alpha_progress", 1.0)),
        "alpha_d": float(r.get("alpha_d", 0.5)),
        "d_cap": float(r.get("d_cap", 2.0)),
        "d_ref": float(r.get("d_ref", 1.0)),
        "alpha_h": float(r.get("alpha_h", 0.5)),
        "alpha_a": float(r.get("alpha_a", 0.01)),
        "alpha_v": float(r.get("alpha_v", 0.0)),
        "alpha_slip": float(r.get("alpha_slip", 2.0)),
        "alpha_steerspeed": float(r.get("alpha_steerspeed", 0.0)),
        "alpha_underspeed": float(r.get("alpha_underspeed", 0.0)),
        "underspeed_tolerance": float(r.get("underspeed_tolerance", 2.0)),
        "vx_max": float(r.get("vx_max", 75.0)),
        "alpha_overspeed": float(r.get("alpha_overspeed", 1.0)),
        "use_curvature_vxmax": bool(r.get("use_curvature_vxmax", True)),
        "use_curvature_vx_ref": bool(r.get("use_curvature_vx_ref", True)),
        "use_lookahead_overspeed": bool(r.get("use_lookahead_overspeed", True)),
        "overspeed_lookahead_max_distance": float(
            r.get("overspeed_lookahead_max_distance", 160.0)
        ),
        "braking_decel": float(r.get("braking_decel", 4.0)),
        "a_lat_max": float(r.get("a_lat_max", 15.0)),
        "kappa_min": float(r.get("kappa_min", 5e-3)),
        "crash_penalty": float(r.get("crash_penalty", 200.0)),
        "stuck_penalty": float(r.get("stuck_penalty", 50.0)),
        "lap_bonus": float(r.get("lap_bonus", 50.0)),
    }
    # Explicit curvature look-ahead distances (metres). When present, these
    # override the legacy (n_curvature_points × spacing) grid. Paper-style
    # non-uniform look-aheads (20/40/60/80 m) give the agent longer horizon
    # perception of upcoming corners, which matters more than dense short-
    # range sampling.
    cl_dist = e.get("curvature_lookahead_distances")
    if cl_dist is not None:
        cfg["curvature_lookahead_distances"] = [float(x) for x in cl_dist]

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

    # Optional expert raceline from telemetry. When enabled, the env
    # computes Frenet frame + vx_ref off this curve instead of the
    # geometric centerline.
    if rl.get("csv"):
        cfg["raceline_csv"] = _resolve_path(config_path, str(rl["csv"]))
        cfg["use_raceline"] = bool(rl.get("enabled", True))
        cfg["use_raceline_vx_ref"] = bool(rl.get("use_vx_profile", True))
        cfg["raceline_smooth_xy"] = int(rl.get("smooth_xy", 15))
        cfg["raceline_smooth_vx"] = int(rl.get("smooth_vx", 25))
        cfg["raceline_resample_ds"] = float(rl.get("resample_ds", 2.0))
        cfg["raceline_min_lap_coverage"] = float(
            rl.get("min_lap_coverage", 0.90)
        )

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
        "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
        "log_alpha_min": float(s.get("log_alpha_min", -10.0)),
        "log_alpha_max": float(s.get("log_alpha_max", 5.0)),
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
    if tr.get("sac_policy_checkpoint"):
        out["sac_policy_checkpoint"] = _resolve_path(
            config_path, str(tr["sac_policy_checkpoint"])
        )
    out["eval_randomize_start"] = bool(tr.get("eval_randomize_start", False))
    out["train_randomize_start"] = bool(tr.get("train_randomize_start", False))
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


def load_rl_config(
    path: Union[str, Path],
    *,
    env_overrides: Optional[Mapping[str, Any]] = None,
) -> RLConfig:
    """Load YAML and return :class:`RLConfig`.

    *env_overrides* is merged into the built env dict (last key wins). Used by
    eval/visualize to e.g. ``{"use_raceline": False}`` so old SAC checkpoints
    (20-dim obs) load while the YAML still enables raceline for new training.
    """
    config_path = Path(path).expanduser().resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, MutableMapping):
        raise ValueError(f"Invalid YAML root in {config_path}")

    env_cfg = _build_env_config(raw, config_path)
    if env_overrides:
        env_cfg = {**env_cfg, **dict(env_overrides)}

    return RLConfig(
        path=config_path,
        raw=dict(raw),
        env=env_cfg,
        sac_agent=_build_sac_agent_config(raw),
        training=_build_training_runtime(raw, config_path),
    )
