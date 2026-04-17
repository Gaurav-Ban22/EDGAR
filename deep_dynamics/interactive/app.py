"""
FastAPI server for the Putnam interactive drawing predictor.

Endpoints:
    GET  /            -> static frontend (index.html)
    GET  /track       -> track bounds + centerline in pixel coordinates for a
                         requested canvas size, plus the pixel<->world transform.
    GET  /models      -> list of selectable trained checkpoints.
    POST /predict     -> given drawn pixel samples and a model id, return
                         reconstructed drawn trajectory and predicted rollout
                         trajectory in both pixel and world coordinates.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deep_dynamics.interactive.pipeline import (
    CONTROLLERS,
    DEFAULT_TRACK_ID,
    MAX_ROLLOUT_STEPS,
    TRACK_REGISTRY,
    LoadedModel,
    Track,
    build_transform,
    load_model,
    load_track,
    predict_from_pixels,
    predict_from_point,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(MODULE_DIR, "static")
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))

# Each entry is a hand-curated selectable checkpoint. Paths are resolved
# relative to REPO_ROOT at request time. We pick the highest-epoch weights
# available in each run folder.
MODEL_REGISTRY: List[Dict[str, str]] = [
    {
        "id": "iac_dumbbounds_291",
        "label": "Deep Dynamics IAC — dumbboundsiac (epoch 291)",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_iac.yaml",
        "weights": "deep_dynamics/output/deep_dynamics_iac/dumbboundsiac/epoch_291.pth",
        "scaler": "deep_dynamics/output/deep_dynamics_iac/dumbboundsiac/scaler.pkl",
        "notes": "Best-trained IAC checkpoint. Recommended default.",
    },
    {
        "id": "iac_run2_102",
        "label": "Deep Dynamics IAC — iac_run2 (epoch 102)",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_iac.yaml",
        "weights": "deep_dynamics/output/deep_dynamics_iac/iac_run2/epoch_102.pth",
        "scaler": "deep_dynamics/output/deep_dynamics_iac/iac_run2/scaler.pkl",
        "notes": "Earlier IAC checkpoint, less trained.",
    },
    # NOTE: The `deep_dynamics_pinn` checkpoints in this repo were trained on
    # Bayesrace/ETHZ data (RC-scale vehicle, mass ~0.04 kg, wheelbase ~0.06 m)
    # and are not compatible with the real-scale Putnam track / IAC car, so
    # they're intentionally omitted from this registry. If you train a PINN
    # model on IAC data with HORIZON that matches, add it here.
]

DEFAULT_MODEL_ID = MODEL_REGISTRY[0]["id"]


def _available_models() -> List[Dict[str, object]]:
    """Filter the registry down to entries whose files actually exist."""
    out: List[Dict[str, object]] = []
    for m in MODEL_REGISTRY:
        weights_abs = os.path.join(REPO_ROOT, m["weights"])
        scaler_abs = os.path.join(REPO_ROOT, m["scaler"])
        cfg_abs = os.path.join(REPO_ROOT, m["cfg"])
        available = (
            os.path.exists(weights_abs)
            and os.path.exists(scaler_abs)
            and os.path.exists(cfg_abs)
        )
        out.append(
            {
                "id": m["id"],
                "label": m["label"],
                "notes": m["notes"],
                "available": available,
            }
        )
    return out


# ---------------------------------------------------------------------------
# App + one-time state
# ---------------------------------------------------------------------------

app = FastAPI(title="Deep Dynamics Drawing Predictor")

_model_cache: Dict[str, LoadedModel] = {}


def _get_loaded_model(model_id: str) -> LoadedModel:
    """Load (and cache) the requested model by registry id."""
    if model_id not in {m["id"] for m in MODEL_REGISTRY}:
        raise HTTPException(status_code=400, detail=f"Unknown model id: {model_id}")
    if model_id in _model_cache:
        return _model_cache[model_id]
    entry = next(m for m in MODEL_REGISTRY if m["id"] == model_id)
    try:
        loaded = load_model(
            cfg_path=os.path.join(REPO_ROOT, entry["cfg"]),
            weights_path=os.path.join(REPO_ROOT, entry["weights"]),
            scaler_path=os.path.join(REPO_ROOT, entry["scaler"]),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' files not available: {e}",
        )
    _model_cache[model_id] = loaded
    return loaded


def _get_track(track_id: str) -> Track:
    if track_id not in TRACK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown track id: {track_id}")
    return load_track(track_id)


@app.on_event("startup")
def _startup() -> None:
    # Warm-load the default track and model so the first prediction isn't slow.
    try:
        _get_track(DEFAULT_TRACK_ID)
    except HTTPException:
        pass
    try:
        _get_loaded_model(DEFAULT_MODEL_ID)
    except HTTPException:
        pass


# Static files + root page
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------------------------------------------------------------------------
# /track
# ---------------------------------------------------------------------------

@app.get("/tracks")
def get_tracks():
    return {
        "default": DEFAULT_TRACK_ID,
        "tracks": [
            {
                "id": tid,
                "label": entry["label"],
                "notes": entry["notes"],
                "default_zoom": float(entry["default_zoom"]),
            }
            for tid, entry in TRACK_REGISTRY.items()
        ],
    }


@app.get("/track")
def get_track(
    canvas_w: int = 1200,
    canvas_h: int = 800,
    zoom: float = 1.0,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    track_id: str = DEFAULT_TRACK_ID,
):
    track = _get_track(track_id)
    tf = build_transform(
        track, canvas_w, canvas_h, zoom=zoom, pan_px_x=pan_x, pan_px_y=pan_y
    )
    inner_px = tf.world_to_pixel_array(track.inner).tolist()
    outer_px = tf.world_to_pixel_array(track.outer).tolist()
    center_px = tf.world_to_pixel_array(track.centerline).tolist()
    return {
        "track_id": track_id,
        "canvas": {"w": canvas_w, "h": canvas_h},
        "transform": {"scale": tf.scale, "x0": tf.x0, "y0": tf.y0},
        "view": {"zoom": zoom, "pan_x": pan_x, "pan_y": pan_y},
        "inner_px": inner_px,
        "outer_px": outer_px,
        "centerline_px": center_px,
        "world_bbox": track.bbox,
    }


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class Sample(BaseModel):
    x: float
    y: float
    t: float = Field(..., description="Seconds since the user started drawing.")


class StartPoint(BaseModel):
    x: float = Field(..., description="Pixel x of the start location on the canvas.")
    y: float = Field(..., description="Pixel y of the start location on the canvas.")
    heading: Optional[float] = Field(
        default=None,
        description="Optional world-frame heading in radians. Auto-derived from "
                    "the centerline tangent if omitted.",
    )
    flip_heading: bool = Field(
        default=False,
        description="If true, add 180° to the (auto or provided) heading. Lets the "
                    "user go the other way around the track from a single click.",
    )
    speed: Optional[float] = Field(
        default=None, description="Optional initial speed in m/s (defaults to v_target)."
    )


class PredictRequest(BaseModel):
    canvas_w: int = 1200
    canvas_h: int = 800
    samples: List[Sample] = Field(default_factory=list)
    start: Optional[StartPoint] = Field(
        default=None,
        description="Seed a rollout from a single clicked point. If provided, "
                    "the request is run in point-mode and `samples` is ignored.",
    )
    mode: Optional[str] = Field(
        default=None,
        description="Explicit mode override: 'drawn' | 'point'. If omitted, the "
                    "server picks based on whether `start` is set.",
    )
    controller: str = Field(
        default="pursuit",
        description="Rollout controller: 'pursuit' (kinematic pure-pursuit) or "
                    "'mpc' (random-shooting MPC over the dynamics model).",
    )
    v_target: Optional[float] = None
    max_steps: int = 300
    model_id: Optional[str] = Field(
        default=None,
        description="Which trained checkpoint to use. See GET /models.",
    )
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    track_id: Optional[str] = Field(
        default=None,
        description="Which track geometry to use. See GET /tracks.",
    )


@app.get("/models")
def get_models():
    return {
        "default": DEFAULT_MODEL_ID,
        "models": _available_models(),
    }


@app.get("/controllers")
def get_controllers():
    """List selectable rollout controllers."""
    return {
        "default": "pursuit",
        "controllers": [
            {
                "id": "pursuit",
                "label": "Pure-pursuit",
                "notes": "Kinematic centerline follower. Fast, model-agnostic.",
            },
            {
                "id": "mpc",
                "label": "Model-Predictive (MPC)",
                "notes": (
                    "Samples candidate actions around pure-pursuit at every step, "
                    "simulates them through the dynamics model, picks the one whose "
                    "predicted 2-step pose best tracks the centerline. Slower "
                    "(~5\u00d7) but explicitly uses the trained dynamics model as its "
                    "planner."
                ),
            },
        ],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if req.controller not in CONTROLLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown controller '{req.controller}'. Expected one of {list(CONTROLLERS)}.",
        )

    mode = (req.mode or "").lower().strip()
    if not mode:
        mode = "point" if req.start is not None else "drawn"
    if mode not in ("drawn", "point"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}' (expected 'drawn' or 'point').",
        )

    if mode == "drawn" and len(req.samples) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 drawn samples.")
    if mode == "point" and req.start is None:
        raise HTTPException(
            status_code=400, detail="point-mode requires a `start` pixel location."
        )

    model_id = req.model_id or DEFAULT_MODEL_ID
    track_id = req.track_id or DEFAULT_TRACK_ID
    loaded = _get_loaded_model(model_id)
    track = _get_track(track_id)

    tf = build_transform(
        track,
        req.canvas_w,
        req.canvas_h,
        zoom=req.zoom,
        pan_px_x=req.pan_x,
        pan_px_y=req.pan_y,
    )

    max_steps = max(1, min(int(req.max_steps), MAX_ROLLOUT_STEPS))

    try:
        if mode == "drawn":
            pixel_samples = [(s.x, s.y, s.t) for s in req.samples]
            result = predict_from_pixels(
                pixel_samples=pixel_samples,
                transform=tf,
                track=track,
                loaded=loaded,
                v_target=req.v_target,
                max_steps=max_steps,
                controller_type=req.controller,
            )
        else:
            assert req.start is not None
            result = predict_from_point(
                pixel_x=req.start.x,
                pixel_y=req.start.y,
                transform=tf,
                track=track,
                loaded=loaded,
                v_target=req.v_target,
                max_steps=max_steps,
                controller_type=req.controller,
                heading=req.start.heading,
                flip_heading=req.start.flip_heading,
                start_speed=req.start.speed,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["model_id"] = model_id
    result["track_id"] = track_id
    return result
