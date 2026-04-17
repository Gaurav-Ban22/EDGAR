"""
Pipeline for the interactive Putnam drawing predictor.

Given a hand-drawn line on the Putnam track, reconstructs a best-effort vehicle
state history, then rolls out the trained DeepDynamicsIAC model forward using
a pure-pursuit controller on the track centerline to synthesize future commands.
"""

from __future__ import annotations

import csv
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from deep_dynamics.model.models import string_to_model


# ---------------------------------------------------------------------------
# Configuration / defaults
# ---------------------------------------------------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))

DEFAULT_CFG = os.path.join(REPO_ROOT, "deep_dynamics", "cfgs", "model", "deep_dynamics_iac.yaml")
DEFAULT_WEIGHTS = os.path.join(
    REPO_ROOT, "deep_dynamics", "output", "deep_dynamics_iac", "dumbboundsiac", "epoch_291.pth"
)
DEFAULT_SCALER = os.path.join(
    REPO_ROOT, "deep_dynamics", "output", "deep_dynamics_iac", "dumbboundsiac", "scaler.pkl"
)
DEFAULT_INNER = os.path.join(REPO_ROOT, "deep_dynamics", "visualize", "tracks", "putnam_inner_bound.csv")
DEFAULT_OUTER = os.path.join(REPO_ROOT, "deep_dynamics", "visualize", "tracks", "putnam_outer_bound.csv")

# F1TENTH-format centerlines (x_m, y_m, w_tr_right_m, w_tr_left_m) live here.
# Any *_centerline.csv file in this directory is auto-registered as a track.
F1TENTH_TRACKS_DIR = os.path.join(MODULE_DIR, "tracks")

TS = 0.04
DEFAULT_HORIZON = 15   # Fallback when a model does not advertise its own.
MIN_VX = 5.0            # training data was filtered to vx >= 5
MAX_VX = 50.0
MAX_ROLLOUT_STEPS = 1500  # 60 s at Ts=0.04

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Track geometry helpers
# ---------------------------------------------------------------------------

def _load_xy_csv(path: str) -> np.ndarray:
    pts = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                pts.append([float(row[0]), float(row[1])])
            except ValueError:
                continue
    return np.asarray(pts, dtype=np.float64)


def _resample_polyline_by_arclength(points: np.ndarray, n: int) -> np.ndarray:
    """Return `n` equally-arc-length-spaced points along an open polyline."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.tile(points[:1], (n, 1))
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=points.dtype)
    out[:, 0] = np.interp(targets, s, points[:, 0])
    out[:, 1] = np.interp(targets, s, points[:, 1])
    return out


def _polygon_contains(polygon: np.ndarray, x: float, y: float) -> bool:
    """Ray-cast point-in-polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class Track:
    """Geometry for a closed-loop racetrack.

    `inner` is the smaller loop (inside of the oval, e.g. infield grass edge)
    and `outer` is the larger enclosing loop. Points are "on track" when
    inside `outer` and outside `inner`.
    """
    inner: np.ndarray      # world (m), smaller loop
    outer: np.ndarray      # world (m), larger enclosing loop
    centerline: np.ndarray  # world (m), resampled along the drivable corridor

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        pts = np.vstack([self.inner, self.outer])
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())

    def on_track(self, x: float, y: float) -> bool:
        return _polygon_contains(self.outer, x, y) and not _polygon_contains(self.inner, x, y)


def build_centerline(inner: np.ndarray, outer: np.ndarray, n: int = 1200) -> np.ndarray:
    """Build a centerline by pairing inner/outer by arc-length and averaging."""
    inner_rs = _resample_polyline_by_arclength(inner, n)
    outer_rs = _resample_polyline_by_arclength(outer, n)

    # The two bounds may be traversed in opposite directions. Pick the alignment
    # (as-is vs. reversed outer) that minimizes total pairwise distance.
    d_forward = np.linalg.norm(inner_rs - outer_rs, axis=1).sum()
    d_reverse = np.linalg.norm(inner_rs - outer_rs[::-1], axis=1).sum()
    if d_reverse < d_forward:
        outer_rs = outer_rs[::-1]

    center = 0.5 * (inner_rs + outer_rs)
    return center


def _polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def load_putnam_track(inner_path: str = DEFAULT_INNER, outer_path: str = DEFAULT_OUTER) -> Track:
    """Load the two Putnam boundary CSVs.

    The file names in this repo are slightly counterintuitive:
    `putnam_inner_bound.csv` traces the LARGER enclosing loop while
    `putnam_outer_bound.csv` traces the smaller infield loop. We normalize
    here so the returned `Track.outer` is always the larger loop and
    `Track.inner` the smaller one.
    """
    a = _load_xy_csv(inner_path)
    b = _load_xy_csv(outer_path)
    area_a = abs(_polygon_area(a))
    area_b = abs(_polygon_area(b))
    if area_a >= area_b:
        outer, inner = a, b
    else:
        outer, inner = b, a
    centerline = build_centerline(inner, outer)
    return Track(inner=inner, outer=outer, centerline=centerline)


def make_simple_oval_track(
    a_center: float = 70.0,
    b_center: float = 35.0,
    width: float = 14.0,
    n: int = 360,
) -> Track:
    """Build a synthetic elliptical oval track.

    `a_center` and `b_center` are the semi-major / semi-minor axes of the
    centerline ellipse (in meters). `width` is the total lateral track
    width; the inner/outer boundaries sit ± width/2 along the ellipse's
    outward normal at each sample. `n` is the number of boundary samples
    per loop.

    The result is compact (default ~140 m × 70 m) and easy to draw on.
    """
    thetas = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    cx = a_center * np.cos(thetas)
    cy = b_center * np.sin(thetas)

    # Outward normal to the ellipse at angle theta is the gradient of the
    # implicit function (x/a)^2 + (y/b)^2 = 1, normalized.
    nx = np.cos(thetas) / a_center
    ny = np.sin(thetas) / b_center
    norm = np.sqrt(nx**2 + ny**2)
    nx = nx / norm
    ny = ny / norm

    half_w = 0.5 * width
    inner_xy = np.column_stack([cx - half_w * nx, cy - half_w * ny])
    outer_xy = np.column_stack([cx + half_w * nx, cy + half_w * ny])
    centerline = np.column_stack([cx, cy])

    # Close each loop so downstream polygon-contains / arc-length code does
    # not need to special-case the wrap-around edge.
    inner_xy = np.vstack([inner_xy, inner_xy[:1]])
    outer_xy = np.vstack([outer_xy, outer_xy[:1]])
    centerline = np.vstack([centerline, centerline[:1]])

    return Track(inner=inner_xy, outer=outer_xy, centerline=centerline)


def load_f1tenth_centerline_csv(path: str) -> np.ndarray:
    """Read an F1TENTH centerline CSV: returns an (N, 4) array of [x, y, w_right, w_left] in meters."""
    rows: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 4:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if len(rows) < 4:
        raise ValueError(f"{path}: too few centerline points ({len(rows)}).")
    return np.asarray(rows, dtype=np.float64)


def load_f1tenth_track(csv_path: str, scale: float = 10.0) -> Track:
    """Build a `Track` from an F1TENTH centerline CSV.

    The F1TENTH dataset ships real F1/DTM circuits downscaled 1:10 with a
    fixed nominal track width of 2.2 m. We apply `scale` (default 10×) to
    recover something close to real-world dimensions — so a track that was
    ~380 m long in the file becomes ~3.8 km on canvas, and the drivable
    corridor becomes ~22 m wide.

    The two track edges are constructed by offsetting each centerline point
    along its (left-hand) normal by `w_tr_left` on one side and `-w_tr_right`
    on the other. The enclosing-vs-enclosed role (outer vs inner) is then
    resolved by polygon-area comparison, so the downstream `on_track` check
    works regardless of which way around the loop the CSV was generated.
    """
    data = load_f1tenth_centerline_csv(csv_path)
    xy = data[:, :2] * scale
    w_right = data[:, 2] * scale
    w_left = data[:, 3] * scale

    # Drop a trailing duplicate if the file happens to include one, then
    # work with an open polyline of N unique points around the loop.
    if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
        xy = xy[:-1]
        w_right = w_right[:-1]
        w_left = w_left[:-1]
    n = len(xy)

    # Central-difference tangents with wrap-around (closed loop).
    nxt = np.roll(xy, -1, axis=0)
    prv = np.roll(xy, 1, axis=0)
    tangent = nxt - prv
    mag = np.linalg.norm(tangent, axis=1, keepdims=True)
    mag = np.where(mag < 1e-9, 1.0, mag)
    tangent = tangent / mag
    # Left-hand normal = tangent rotated +90° CCW = (-ty, tx).
    normal_left = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    edge_a = xy + w_left[:, None] * normal_left       # offset to the "left"
    edge_b = xy - w_right[:, None] * normal_left      # offset to the "right"

    # Close both loops + the centerline so the downstream polygon helpers
    # don't need to special-case wrap-around.
    edge_a_closed = np.vstack([edge_a, edge_a[:1]])
    edge_b_closed = np.vstack([edge_b, edge_b[:1]])
    centerline = np.vstack([xy, xy[:1]])

    if abs(_polygon_area(edge_a_closed)) >= abs(_polygon_area(edge_b_closed)):
        outer, inner = edge_a_closed, edge_b_closed
    else:
        outer, inner = edge_b_closed, edge_a_closed

    return Track(inner=inner, outer=outer, centerline=centerline)


# Track registry ----------------------------------------------------------

# Display metadata for known F1TENTH tracks we ship in this repo. Any CSV
# dropped into F1TENTH_TRACKS_DIR that isn't listed here will still get
# auto-registered with sensible fallback metadata.
_F1TENTH_OVERRIDES: Dict[str, Dict[str, object]] = {
    "monza": {
        "label": "Monza (F1)",
        "notes": "Autodromo Nazionale Monza — fast, flowing, famous. Real F1 layout, 10× upscaled from F1TENTH data.",
        "default_zoom": 3.0,
    },
    "silverstone": {
        "label": "Silverstone (F1)",
        "notes": "Silverstone Circuit — technical, medium-speed. Real F1 layout, 10× upscaled from F1TENTH data.",
        "default_zoom": 3.0,
    },
    "ims": {
        "label": "Indianapolis Motor Speedway (IndyCar)",
        "notes": "IMS — a simple rectangular superspeedway oval. Directly relevant to the IAC model; easy to draw on.",
        "default_zoom": 1.5,
    },
    "austin": {
        "label": "Austin / COTA (F1)",
        "notes": "Circuit of the Americas — mixed rhythm sections, big elevation changes IRL (here 2D). 10× upscaled.",
        "default_zoom": 3.0,
    },
}


def _make_f1tenth_loader(path: str):
    def _loader() -> Track:
        return load_f1tenth_track(path)
    return _loader


def _discover_f1tenth_tracks() -> Dict[str, Dict[str, object]]:
    """Find any `*_centerline.csv` files in F1TENTH_TRACKS_DIR and register them."""
    out: Dict[str, Dict[str, object]] = {}
    if not os.path.isdir(F1TENTH_TRACKS_DIR):
        return out
    for fname in sorted(os.listdir(F1TENTH_TRACKS_DIR)):
        if not fname.lower().endswith("_centerline.csv"):
            continue
        stem = fname[: -len("_centerline.csv")]
        track_id = stem.lower()
        path = os.path.join(F1TENTH_TRACKS_DIR, fname)
        meta = _F1TENTH_OVERRIDES.get(
            track_id,
            {
                "label": f"{stem} (F1TENTH)",
                "notes": "Loaded from an F1TENTH centerline CSV, 10× upscaled to real-world dimensions.",
                "default_zoom": 3.0,
            },
        )
        out[track_id] = {
            "label": meta["label"],
            "notes": meta["notes"],
            "default_zoom": float(meta["default_zoom"]),
            "loader": _make_f1tenth_loader(path),
        }
    return out


TRACK_REGISTRY: Dict[str, Dict[str, object]] = {
    "simple_oval": {
        "label": "Simple Oval",
        "notes": "Synthetic ~140×70 m elliptical oval — small and easy to draw a racing line on.",
        "default_zoom": 1.0,
        "loader": make_simple_oval_track,
    },
    "putnam": {
        "label": "Putnam Park Road Course",
        "notes": "Real Putnam Park bounds loaded from CSV. The dataset used to train the IAC model.",
        "default_zoom": 2.0,
        "loader": load_putnam_track,
    },
}
# Merge in any F1TENTH tracks found on disk.
TRACK_REGISTRY.update(_discover_f1tenth_tracks())

DEFAULT_TRACK_ID = "simple_oval"

_track_cache: Dict[str, Track] = {}


def load_track(track_id: str = DEFAULT_TRACK_ID) -> Track:
    """Return the `Track` for the given registry id, caching the result."""
    if track_id not in TRACK_REGISTRY:
        raise KeyError(f"Unknown track id: {track_id!r}")
    if track_id in _track_cache:
        return _track_cache[track_id]
    track = TRACK_REGISTRY[track_id]["loader"]()  # type: ignore[operator]
    _track_cache[track_id] = track
    return track


# ---------------------------------------------------------------------------
# Pixel <-> world transform
# ---------------------------------------------------------------------------

@dataclass
class PixelWorldTransform:
    """Axis-aligned, uniform-scale mapping between canvas pixels and world meters.

    World y grows upward; canvas y grows downward, so world_y = y0 - scale_y * px_y.
    """
    scale: float      # meters per pixel (uniform, preserving aspect ratio)
    x0: float         # world x at pixel x = 0
    y0: float         # world y at pixel y = 0
    canvas_w: int
    canvas_h: int

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        x = self.x0 + self.scale * px
        y = self.y0 - self.scale * py
        return x, y

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        px = (x - self.x0) / self.scale
        py = (self.y0 - y) / self.scale
        return px, py

    def world_to_pixel_array(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = (pts[:, 0] - self.x0) / self.scale
        out[:, 1] = (self.y0 - pts[:, 1]) / self.scale
        return out

    def pixel_to_world_array(self, px: np.ndarray) -> np.ndarray:
        out = np.empty_like(px, dtype=np.float64)
        out[:, 0] = self.x0 + self.scale * px[:, 0]
        out[:, 1] = self.y0 - self.scale * px[:, 1]
        return out


def build_transform(
    track: Track,
    canvas_w: int,
    canvas_h: int,
    margin: float = 0.05,
    zoom: float = 1.0,
    pan_px_x: float = 0.0,
    pan_px_y: float = 0.0,
) -> PixelWorldTransform:
    """Fit track bbox into canvas with aspect-preserving scale and `margin` border.

    `zoom` > 1 makes the track appear larger on screen (fewer meters per pixel).
    `pan_px_x`, `pan_px_y` shift the rendered content in canvas-pixel units
    relative to the centered, zoomed placement (dragging the view right by
    N pixels corresponds to pan_px_x = +N).
    """
    x_min, y_min, x_max, y_max = track.bbox
    world_w = x_max - x_min
    world_h = y_max - y_min
    usable_w = canvas_w * (1 - 2 * margin)
    usable_h = canvas_h * (1 - 2 * margin)
    base_scale = max(world_w / usable_w, world_h / usable_h)
    zoom = max(float(zoom), 1e-3)
    scale = base_scale / zoom  # meters per pixel

    # Center the (possibly zoomed) track in the canvas, then apply pan.
    pixel_w = world_w / scale
    pixel_h = world_h / scale
    pad_px_x = (canvas_w - pixel_w) / 2
    pad_px_y = (canvas_h - pixel_h) / 2
    x0 = x_min - scale * (pad_px_x + pan_px_x)
    y0 = y_max + scale * (pad_px_y + pan_px_y)
    return PixelWorldTransform(scale=scale, x0=x0, y0=y0, canvas_w=canvas_w, canvas_h=canvas_h)


# ---------------------------------------------------------------------------
# Nominal vehicle / parameter dictionary
# ---------------------------------------------------------------------------

def _nominal_params(param_dict: dict) -> Dict[str, float]:
    """Midpoint of each parameter range, used for inverse longitudinal model."""
    out: Dict[str, float] = {}
    for p in param_dict["PARAMETERS"]:
        name = next(iter(p.keys()))
        lo = p["Min"]
        hi = p["Max"]
        out[name] = 0.5 * (lo + hi)
    return out


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: torch.nn.Module
    scaler: "object"  # sklearn StandardScaler
    param_dict: dict
    nominal: Dict[str, float]
    wheelbase: float
    mass: float
    horizon: int


def load_model(
    cfg_path: str = DEFAULT_CFG,
    weights_path: str = DEFAULT_WEIGHTS,
    scaler_path: str = DEFAULT_SCALER,
) -> LoadedModel:
    with open(cfg_path, "rb") as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
    model.to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return LoadedModel(
        model=model,
        scaler=scaler,
        param_dict=param_dict,
        nominal=_nominal_params(param_dict),
        wheelbase=float(param_dict["VEHICLE_SPECS"]["L"]),
        mass=float(param_dict["VEHICLE_SPECS"]["mass"]),
        horizon=int(param_dict["MODEL"].get("HORIZON", DEFAULT_HORIZON)),
    )


# ---------------------------------------------------------------------------
# Longitudinal inverse model
# ---------------------------------------------------------------------------

def invert_longitudinal(ax: float, vx: float, nominal: Dict[str, float], mass: float) -> float:
    """Invert F_rx_desired = (Cm1 - Cm2*vx)*throttle - Cr0 - Cr2*vx^2 ; F_rx ~= m*ax."""
    Cm1 = nominal["Cm1"]
    Cm2 = nominal["Cm2"]
    Cr0 = nominal["Cr0"]
    Cr2 = nominal["Cr2"]
    numerator = mass * ax + Cr0 + Cr2 * vx * vx
    denom = Cm1 - Cm2 * vx
    if abs(denom) < 1e-6:
        return 0.0
    return float(np.clip(numerator / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# State estimation from a drawn line
# ---------------------------------------------------------------------------

@dataclass
class DrawnTrajectory:
    world_xy: np.ndarray        # (N, 2) smoothed, resampled at Ts
    vx: np.ndarray              # (N,)
    vy: np.ndarray              # (N,) assumed ~0 but kept as array for completeness
    phi: np.ndarray             # (N,) heading
    omega: np.ndarray           # (N,) yaw rate
    throttle_fb: np.ndarray     # (N,)
    steering_fb: np.ndarray     # (N,)
    throttle_cmd: np.ndarray    # (N,) delta throttle
    steering_cmd: np.ndarray    # (N,) delta steering
    clamped_vx: bool
    warnings: List[str]


def _moving_average(a: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with edge-replication padding (no zero-padding)."""
    if w <= 1 or len(a) < w:
        return a.copy()
    pad = w // 2
    padded = np.pad(a, pad, mode="edge")
    kernel = np.ones(w) / w
    smoothed = np.convolve(padded, kernel, mode="valid")
    # When w is even, valid convolution returns len(padded) - w + 1 = len(a) + 1
    return smoothed[:len(a)]


def _resample_time_series(
    points: np.ndarray, times: np.ndarray, ts: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (x, y, t) samples to a uniform dt = ts time grid."""
    if len(points) < 2:
        return points.copy(), times.copy()

    t0 = float(times[0])
    t1 = float(times[-1])
    if t1 - t0 < ts:
        return points.copy(), times.copy()

    n = max(int(np.floor((t1 - t0) / ts)) + 1, 2)
    new_t = t0 + np.arange(n) * ts
    new_xy = np.empty((n, 2), dtype=np.float64)
    new_xy[:, 0] = np.interp(new_t, times, points[:, 0])
    new_xy[:, 1] = np.interp(new_t, times, points[:, 1])
    return new_xy, new_t


def estimate_states_from_drawing(
    world_pts: np.ndarray,
    times_s: np.ndarray,
    nominal: Dict[str, float],
    mass: float,
    wheelbase: float,
    horizon: int = DEFAULT_HORIZON,
    smoothing_window: int = 5,
) -> DrawnTrajectory:
    """Convert a drawn world path (x, y, t) into a uniformly sampled state history."""
    warnings: List[str] = []

    if len(world_pts) < 2:
        raise ValueError("Need at least 2 drawn samples.")

    # De-duplicate consecutive points that share the same timestamp (can happen
    # if the browser delivers duplicates), preserving order.
    keep = np.concatenate([[True], np.diff(times_s) > 1e-4])
    world_pts = world_pts[keep]
    times_s = times_s[keep]
    if len(world_pts) < 2:
        raise ValueError("Drawn samples collapse after de-duplication.")

    # Resample to uniform Ts.
    xy, _t = _resample_time_series(world_pts, times_s, TS)
    if len(xy) < horizon + 2:
        raise ValueError(
            f"Drawn line too short: need at least {horizon + 2} samples at Ts={TS}s "
            f"(got {len(xy)}). Draw a longer path or draw more slowly."
        )

    # Smooth x, y a bit to damp noisy mouse input.
    x_s = _moving_average(xy[:, 0], smoothing_window)
    y_s = _moving_average(xy[:, 1], smoothing_window)

    # Finite differences (central where possible).
    dx = np.gradient(x_s, TS)
    dy = np.gradient(y_s, TS)

    speed = np.sqrt(dx * dx + dy * dy)
    clamped = np.any(speed < MIN_VX)
    if clamped:
        warnings.append(
            f"Minimum speed along drawn line ({speed.min():.2f} m/s) was below the "
            f"training regime (>= {MIN_VX} m/s). Velocity was clamped; predictions "
            f"may drift."
        )
    vx_world = np.clip(speed, MIN_VX, MAX_VX)

    # Tangent angle.
    phi = np.arctan2(dy, dx)
    phi_unwrapped = np.unwrap(phi)

    # In the bicycle model we assume no side-slip, so vy (body) ~= 0. vx (body)
    # equals the scalar speed along the tangent.
    vx = vx_world
    vy = np.zeros_like(vx)

    # Yaw rate: derivative of heading.
    omega = np.gradient(phi_unwrapped, TS)
    omega = _moving_average(omega, smoothing_window)

    # Longitudinal acceleration.
    ax = np.gradient(vx, TS)
    ax = _moving_average(ax, smoothing_window)

    # Inverse longitudinal model -> throttle feedback.
    throttle_fb = np.array([invert_longitudinal(ax_i, vx_i, nominal, mass)
                            for ax_i, vx_i in zip(ax, vx)])

    # Kinematic bicycle inverse -> steering feedback.
    with np.errstate(divide="ignore", invalid="ignore"):
        steering_fb = np.arctan2(wheelbase * omega, np.maximum(vx, 1e-3))
    steering_fb = np.clip(steering_fb, -0.2, 0.2)

    # Command signals: deltas matching the training-data convention used in
    # tools/csv_parser.py, where cmd[k] is the increment APPLIED BETWEEN step k
    # and step k+1, i.e. fb[k+1] - fb[k]. Final sample has no successor, so 0.
    throttle_cmd = np.concatenate([np.diff(throttle_fb), [0.0]])
    steering_cmd = np.concatenate([np.diff(steering_fb), [0.0]])

    return DrawnTrajectory(
        world_xy=np.stack([x_s, y_s], axis=1),
        vx=vx,
        vy=vy,
        phi=phi_unwrapped,
        omega=omega,
        throttle_fb=throttle_fb,
        steering_fb=steering_fb,
        throttle_cmd=throttle_cmd,
        steering_cmd=steering_cmd,
        clamped_vx=bool(clamped),
        warnings=warnings,
    )


def build_history_window(
    traj: DrawnTrajectory, horizon: int = DEFAULT_HORIZON
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Take the last `horizon` samples of a drawn trajectory and package them as
    a (horizon, 7) float array matching the training feature layout.

    Also returns the final pose + feedback values to seed the rollout.
    """
    if len(traj.vx) < horizon:
        raise ValueError(f"Need {horizon} samples, have {len(traj.vx)}.")

    sl = slice(len(traj.vx) - horizon, len(traj.vx))
    window = np.stack([
        traj.vx[sl],
        traj.vy[sl],
        traj.omega[sl],
        traj.throttle_fb[sl],
        traj.steering_fb[sl],
        traj.throttle_cmd[sl],
        traj.steering_cmd[sl],
    ], axis=1).astype(np.float64)

    seed = {
        "x": float(traj.world_xy[-1, 0]),
        "y": float(traj.world_xy[-1, 1]),
        "phi": float(traj.phi[-1]),
        "vx": float(traj.vx[-1]),
        "vy": float(traj.vy[-1]),
        "omega": float(traj.omega[-1]),
        "throttle_fb": float(traj.throttle_fb[-1]),
        "steering_fb": float(traj.steering_fb[-1]),
    }
    return window, seed


# ---------------------------------------------------------------------------
# Pure-pursuit centerline controller
# ---------------------------------------------------------------------------

@dataclass
class PursuitConfig:
    v_target: float = 25.0
    k_v: float = 0.3        # lookahead grows with speed
    L0: float = 6.0         # base lookahead [m]
    kp_throttle: float = 2.0  # throttle gain on speed error -> accel


class PurePursuit:
    """Centerline-following pure-pursuit controller returning target (throttle_fb, steering_fb)."""

    def __init__(
        self,
        track: Track,
        cfg: PursuitConfig,
        nominal: Dict[str, float],
        mass: float,
        wheelbase: float,
        reverse: bool = False,
    ):
        self.track = track
        self.cfg = cfg
        self.nominal = nominal
        self.mass = mass
        self.wheelbase = wheelbase
        # When `reverse` is True, treat the centerline as if traversed backward.
        # This is how "go the other way around the track" is supported from a
        # single start-point click.
        self.center = track.centerline[::-1].copy() if reverse else track.centerline
        self._seg = np.linalg.norm(np.diff(self.center, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._seg)])
        self._total = float(self._cum[-1])

    def _closest_index(self, x: float, y: float) -> int:
        d = (self.center[:, 0] - x) ** 2 + (self.center[:, 1] - y) ** 2
        return int(np.argmin(d))

    def _point_at_arclength(self, s: float) -> np.ndarray:
        s = s % self._total if self._total > 0 else 0.0
        i = int(np.searchsorted(self._cum, s) - 1)
        i = max(0, min(i, len(self._seg) - 1))
        seg_len = self._seg[i]
        if seg_len <= 0:
            return self.center[i].copy()
        alpha = (s - self._cum[i]) / seg_len
        return (1 - alpha) * self.center[i] + alpha * self.center[i + 1]

    def target(self, x: float, y: float, phi: float, vx: float) -> Tuple[float, float]:
        """Return (target_throttle_fb, target_steering_fb) for the given pose."""
        i = self._closest_index(x, y)
        s_here = float(self._cum[i])
        Ld = self.cfg.L0 + self.cfg.k_v * max(vx, 0.0)
        target_pt = self._point_at_arclength(s_here + Ld)

        dx = target_pt[0] - x
        dy = target_pt[1] - y
        alpha = math.atan2(dy, dx) - phi
        # Wrap to [-pi, pi].
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), max(Ld, 1e-3))
        steering = float(np.clip(steering, -0.35, 0.35))

        # Speed control: P on speed error -> desired ax -> invert to throttle.
        ax_des = self.cfg.kp_throttle * (self.cfg.v_target - vx)
        ax_des = float(np.clip(ax_des, -6.0, 6.0))
        throttle = invert_longitudinal(ax_des, vx, self.nominal, self.mass)
        return throttle, steering


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _normalize_window(window: np.ndarray, scaler) -> np.ndarray:
    shape = window.shape
    return scaler.transform(window.reshape(-1, shape[-1])).reshape(shape)


def rollout(
    loaded: LoadedModel,
    track: Track,
    history: np.ndarray,
    seed: Dict[str, float],
    controller: PurePursuit,
    max_steps: int = MAX_ROLLOUT_STEPS,
) -> Dict[str, np.ndarray]:
    """
    Roll the trained model forward step by step. Returns dict with arrays:
        xy (K+1, 2), vx (K+1,), vy (K+1,), omega (K+1,), phi (K+1,),
        throttle (K+1,), steering (K+1,), stop_reason (str)

    The `cmd` slot in each feature row stores the INCREMENT applied between
    that row and the next (fb[k+1] - fb[k]), matching the training data layout
    in `deep_dynamics/tools/csv_parser.py`.
    """
    model = loaded.model
    scaler = loaded.scaler

    window = history.copy()  # (horizon, 7)

    x = seed["x"]
    y = seed["y"]
    phi = seed["phi"]
    throttle_fb = float(window[-1, 3])
    steering_fb = float(window[-1, 4])

    # Fill the forward-looking cmd for the LAST window row using the pure-pursuit
    # controller's target at the current pose. This is the cmd that will be
    # applied during the upcoming T -> T+1 transition.
    target_throttle, target_steering = controller.target(x, y, phi, seed["vx"])
    window[-1, 5] = target_throttle - throttle_fb
    window[-1, 6] = target_steering - steering_fb

    xs = [x]
    ys = [y]
    vxs = [float(window[-1, 0])]
    vys = [float(window[-1, 1])]
    omegas = [float(window[-1, 2])]
    phis = [phi]
    thr_track = [throttle_fb]
    steer_track = [steering_fb]

    stop_reason = "max_steps"

    for step in range(max_steps):
        # Forward pass with the current window.
        norm_window = _normalize_window(window, scaler)
        x_t = torch.from_numpy(window[None, ...]).float().to(DEVICE)
        xn_t = torch.from_numpy(norm_window[None, ...]).float().to(DEVICE)

        with torch.no_grad():
            if model.is_rnn:
                h = model.init_hidden(1)
                next_state, _, _ = model(x_t, xn_t, h)
            else:
                next_state, _, _ = model(x_t, xn_t)
        next_state = next_state.cpu().numpy().reshape(-1)
        next_vx = float(np.clip(next_state[0], 0.5, MAX_VX))
        next_vy = float(next_state[1])
        next_omega = float(next_state[2])

        # Integrate pose in the world frame using the state at time T
        # (currently at window[-1]), matching the equations in
        # deep_dynamics/visualize/plot_predictions_iac.py.
        cur_vx = float(window[-1, 0])
        cur_vy = float(window[-1, 1])
        cur_omega = float(window[-1, 2])
        x = x + (cur_vx * math.cos(phi) - cur_vy * math.sin(phi)) * TS
        y = y + (cur_vx * math.sin(phi) + cur_vy * math.cos(phi)) * TS
        phi = phi + cur_omega * TS

        # fb at T+1 is the target the controller had computed at pose T (that
        # is the command that just drove us through T -> T+1).
        new_throttle_fb = target_throttle
        new_steering_fb = target_steering

        # Compute the next target at pose T+1, which becomes the forward cmd for
        # the newly appended row (T+1 -> T+2).
        target_throttle, target_steering = controller.target(x, y, phi, next_vx)
        new_throttle_cmd = target_throttle - new_throttle_fb
        new_steering_cmd = target_steering - new_steering_fb

        new_row = np.array([next_vx, next_vy, next_omega,
                            new_throttle_fb, new_steering_fb,
                            new_throttle_cmd, new_steering_cmd], dtype=np.float64)
        window = np.vstack([window[1:], new_row])

        throttle_fb = new_throttle_fb
        steering_fb = new_steering_fb

        xs.append(x)
        ys.append(y)
        vxs.append(next_vx)
        vys.append(next_vy)
        omegas.append(next_omega)
        phis.append(phi)
        thr_track.append(throttle_fb)
        steer_track.append(steering_fb)

        # Stopping criteria.
        if not track.on_track(x, y):
            stop_reason = "left_track"
            break
        if abs(next_omega) > 3.0:
            stop_reason = "unstable_yaw"
            break
        if next_vx < MIN_VX * 0.5:
            stop_reason = "low_speed"
            break

    return {
        "xy": np.array(list(zip(xs, ys))),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "omega": np.array(omegas),
        "phi": np.array(phis),
        "throttle": np.array(thr_track),
        "steering": np.array(steer_track),
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Model-predictive control (random-shooting MPC with the dynamics model)
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    """Random-shooting MPC over the learned dynamics model.

    At every control step we sample K candidate (throttle, steering) actions
    around pure-pursuit's suggestion, simulate ONE step forward through the
    dynamics model for each candidate in a single batched forward pass, then
    propagate each resulting state one more kinematic step (with the candidate
    velocities held constant) so the candidate poses actually diverge. The
    candidate that minimizes distance to the centerline — plus a speed-tracking
    bonus and a large off-track penalty — is applied. Its next-state prediction
    is re-used so we don't need a second forward pass per step.
    """
    K: int = 8                   # number of sampled actions per step
    steering_sigma: float = 0.06 # stddev of steering perturbations [rad]
    throttle_sigma: float = 0.0  # 0 disables throttle search (keeps pp's value)
    speed_weight: float = 0.02   # score weight on (vx - v_target)^2 / v_target^2
    offtrack_penalty: float = 1e6


def rollout_mpc(
    loaded: LoadedModel,
    track: Track,
    history: np.ndarray,
    seed: Dict[str, float],
    controller: PurePursuit,
    max_steps: int = MAX_ROLLOUT_STEPS,
    cfg: Optional[MPCConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Roll the model forward using sampling-based MPC as the controller.

    Structure mirrors `rollout()` closely so the returned dict is identical
    in shape. The key difference is what populates `window[-1, 5:7]` (the
    command that will drive the T -> T+1 transition): instead of taking
    pure-pursuit's suggestion verbatim, we sample K perturbations around it,
    batch-simulate one step through the dynamics model for each, and pick
    the perturbation whose PREDICTED pose at T+2 is best.
    """
    cfg = cfg or MPCConfig()
    rng = rng or np.random.default_rng(0)

    model = loaded.model
    scaler = loaded.scaler

    window = history.copy()  # (horizon, 7)

    x = seed["x"]
    y = seed["y"]
    phi = seed["phi"]
    throttle_fb = float(window[-1, 3])
    steering_fb = float(window[-1, 4])

    xs = [x]
    ys = [y]
    vxs = [float(window[-1, 0])]
    vys = [float(window[-1, 1])]
    omegas = [float(window[-1, 2])]
    phis = [phi]
    thr_track = [throttle_fb]
    steer_track = [steering_fb]

    center = track.centerline
    # Tangent along the centerline, used for the pose-at-T+2 extrapolation.
    K = int(cfg.K)
    v_target = float(controller.cfg.v_target)
    v_target_sq = max(v_target * v_target, 1.0)

    stop_reason = "max_steps"

    for step in range(max_steps):
        # Pure-pursuit baseline (also a safe fallback if scoring all-ties).
        pp_throttle, pp_steering = controller.target(
            x, y, phi, float(window[-1, 0])
        )

        # Draw K candidate actions (steering spread around pp, throttle fixed
        # unless throttle_sigma is set). Always include pp itself at index 0.
        steerings = pp_steering + rng.normal(0.0, cfg.steering_sigma, size=K)
        if cfg.throttle_sigma > 0.0:
            throttles = pp_throttle + rng.normal(0.0, cfg.throttle_sigma, size=K)
        else:
            throttles = np.full(K, pp_throttle, dtype=np.float64)
        steerings[0] = pp_steering
        throttles[0] = pp_throttle
        steerings = np.clip(steerings, -0.35, 0.35)

        # Build K copies of the current window, each with a different cmd
        # on the last row.
        batch = np.repeat(window[None, :, :], K, axis=0).astype(np.float64)
        batch[:, -1, 5] = throttles - throttle_fb
        batch[:, -1, 6] = steerings - steering_fb

        # Batched forward pass.
        shape = batch.shape
        batch_norm = scaler.transform(batch.reshape(-1, shape[-1])).reshape(shape)
        bt = torch.from_numpy(batch).float().to(DEVICE)
        bnt = torch.from_numpy(batch_norm).float().to(DEVICE)
        with torch.no_grad():
            if model.is_rnn:
                h = model.init_hidden(K)
                next_states, _, _ = model(bt, bnt, h)
            else:
                next_states, _, _ = model(bt, bnt)
        next_states_np = next_states.cpu().numpy()  # (K, 3)

        # Pose at T+1 (identical across K since it depends only on state at T).
        cur_vx = float(window[-1, 0])
        cur_vy = float(window[-1, 1])
        cur_omega = float(window[-1, 2])
        x_T1 = x + (cur_vx * math.cos(phi) - cur_vy * math.sin(phi)) * TS
        y_T1 = y + (cur_vx * math.sin(phi) + cur_vy * math.cos(phi)) * TS
        phi_T1 = phi + cur_omega * TS

        # Pose at T+2, integrated with each candidate's predicted T+1 state.
        nvx = next_states_np[:, 0]
        nvy = next_states_np[:, 1]
        nomega = next_states_np[:, 2]
        cos1 = math.cos(phi_T1)
        sin1 = math.sin(phi_T1)
        x_T2 = x_T1 + (nvx * cos1 - nvy * sin1) * TS
        y_T2 = y_T1 + (nvx * sin1 + nvy * cos1) * TS

        # Vectorized distance from each candidate's T+2 pose to the centerline.
        dx_c = center[:, 0][None, :] - x_T2[:, None]
        dy_c = center[:, 1][None, :] - y_T2[:, None]
        min_d2 = (dx_c * dx_c + dy_c * dy_c).min(axis=1)  # (K,)

        speed_err_sq = (nvx - v_target) ** 2 / v_target_sq
        scores = -min_d2 - cfg.speed_weight * speed_err_sq * v_target_sq

        # Off-track predictions are heavily penalized so the planner is
        # strongly biased to stay inside the corridor.
        for k in range(K):
            if not track.on_track(float(x_T2[k]), float(y_T2[k])):
                scores[k] -= cfg.offtrack_penalty

        best = int(np.argmax(scores))
        chosen_throttle = float(throttles[best])
        chosen_steering = float(steerings[best])

        # Commit the chosen action: overwrite window's last-row cmd so the
        # recorded history reflects what was actually applied, then take the
        # corresponding forward-pass result from the batch (no extra forward
        # pass needed).
        window[-1, 5] = chosen_throttle - throttle_fb
        window[-1, 6] = chosen_steering - steering_fb

        next_vx = float(np.clip(nvx[best], 0.5, MAX_VX))
        next_vy = float(nvy[best])
        next_omega = float(nomega[best])

        x = x_T1
        y = y_T1
        phi = phi_T1

        new_throttle_fb = chosen_throttle
        new_steering_fb = chosen_steering

        # Seed the next iteration's "forward-looking cmd" with pure-pursuit's
        # baseline; next iteration's MPC call will overwrite it before using it.
        seed_throttle, seed_steering = controller.target(x, y, phi, next_vx)
        new_throttle_cmd = seed_throttle - new_throttle_fb
        new_steering_cmd = seed_steering - new_steering_fb

        new_row = np.array(
            [next_vx, next_vy, next_omega,
             new_throttle_fb, new_steering_fb,
             new_throttle_cmd, new_steering_cmd],
            dtype=np.float64,
        )
        window = np.vstack([window[1:], new_row])
        throttle_fb = new_throttle_fb
        steering_fb = new_steering_fb

        xs.append(x)
        ys.append(y)
        vxs.append(next_vx)
        vys.append(next_vy)
        omegas.append(next_omega)
        phis.append(phi)
        thr_track.append(throttle_fb)
        steer_track.append(steering_fb)

        if not track.on_track(x, y):
            stop_reason = "left_track"
            break
        if abs(next_omega) > 3.0:
            stop_reason = "unstable_yaw"
            break
        if next_vx < MIN_VX * 0.5:
            stop_reason = "low_speed"
            break

    return {
        "xy": np.array(list(zip(xs, ys))),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "omega": np.array(omegas),
        "phi": np.array(phis),
        "throttle": np.array(thr_track),
        "steering": np.array(steer_track),
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Start-from-point: synthesize a history window from a single clicked point
# ---------------------------------------------------------------------------

def centerline_heading_at(track: Track, x: float, y: float) -> Tuple[int, float]:
    """Return (index_of_nearest_centerline_point, tangent_angle_rad)."""
    c = track.centerline
    d2 = (c[:, 0] - x) ** 2 + (c[:, 1] - y) ** 2
    i = int(np.argmin(d2))
    n = len(c) - 1  # centerline closes on itself, last row == first
    i_next = (i + 1) % n
    i_prev = (i - 1) % n
    dx = float(c[i_next, 0] - c[i_prev, 0])
    dy = float(c[i_next, 1] - c[i_prev, 1])
    return i, math.atan2(dy, dx)


def build_history_from_start_point(
    *,
    world_x: float,
    world_y: float,
    heading: float,
    speed: float,
    horizon: int,
    nominal: Dict[str, float],
    mass: float,
) -> Tuple[np.ndarray, Dict[str, float], DrawnTrajectory]:
    """Create a synthetic `horizon`-long history seeded at (world_x, world_y).

    The synthesized history is a steady-state cruise at `speed` with zero
    yaw rate and zero steering. Throttle feedback is the value that holds the
    requested speed against aero/rolling resistance at ax = 0. All cmds are
    zero so the last-row cmd becomes the first thing the rollout overwrites.

    Returns (history_window, seed_dict, drawn_like_trajectory). The trajectory
    is shaped identically to a drawn one (so callers can treat both paths
    uniformly) and its `world_xy` is just `horizon` repeats of the start point.
    """
    speed = float(max(MIN_VX, min(MAX_VX, speed)))
    steady_throttle = float(invert_longitudinal(0.0, speed, nominal, mass))

    vx = np.full(horizon, speed, dtype=np.float64)
    vy = np.zeros(horizon, dtype=np.float64)
    omega = np.zeros(horizon, dtype=np.float64)
    throttle_fb = np.full(horizon, steady_throttle, dtype=np.float64)
    steering_fb = np.zeros(horizon, dtype=np.float64)
    throttle_cmd = np.zeros(horizon, dtype=np.float64)
    steering_cmd = np.zeros(horizon, dtype=np.float64)
    xy = np.tile(np.array([[world_x, world_y]], dtype=np.float64), (horizon, 1))
    phi = np.full(horizon, heading, dtype=np.float64)

    window = np.stack(
        [vx, vy, omega, throttle_fb, steering_fb, throttle_cmd, steering_cmd],
        axis=1,
    )
    seed = {
        "x": float(world_x),
        "y": float(world_y),
        "phi": float(heading),
        "vx": speed,
        "vy": 0.0,
        "omega": 0.0,
        "throttle_fb": steady_throttle,
        "steering_fb": 0.0,
    }
    synthetic_traj = DrawnTrajectory(
        world_xy=xy,
        vx=vx,
        vy=vy,
        phi=phi,
        omega=omega,
        throttle_fb=throttle_fb,
        steering_fb=steering_fb,
        throttle_cmd=throttle_cmd,
        steering_cmd=steering_cmd,
        clamped_vx=False,
        warnings=[],
    )
    return window, seed, synthetic_traj


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

CONTROLLERS = ("pursuit", "mpc")


def _run_rollout_and_pack(
    *,
    loaded: LoadedModel,
    track: Track,
    history: np.ndarray,
    seed: Dict[str, float],
    traj: DrawnTrajectory,
    transform: PixelWorldTransform,
    v_target: float,
    max_steps: int,
    controller_type: str,
    mode: str,
    reverse_pursuit: bool = False,
) -> Dict[str, object]:
    """Shared post-processing: run the chosen controller's rollout and pack the
    response body identically regardless of whether we came from pixel samples
    or a single start point."""
    if controller_type not in CONTROLLERS:
        raise ValueError(
            f"Unknown controller '{controller_type}' (expected one of {CONTROLLERS})."
        )

    cfg = PursuitConfig(v_target=v_target)
    pp = PurePursuit(
        track, cfg, loaded.nominal, loaded.mass, loaded.wheelbase,
        reverse=reverse_pursuit,
    )

    if controller_type == "mpc":
        pred = rollout_mpc(
            loaded, track, history, seed, pp, max_steps=max_steps
        )
    else:
        pred = rollout(loaded, track, history, seed, pp, max_steps=max_steps)

    drawn_pixel = transform.world_to_pixel_array(traj.world_xy)
    pred_pixel = transform.world_to_pixel_array(pred["xy"])

    return {
        "drawn_world": traj.world_xy.tolist(),
        "drawn_pixel": drawn_pixel.tolist(),
        "drawn_vx": traj.vx.tolist(),
        "drawn_steering": traj.steering_fb.tolist(),
        "drawn_throttle": traj.throttle_fb.tolist(),
        "predicted_world": pred["xy"].tolist(),
        "predicted_pixel": pred_pixel.tolist(),
        "predicted_vx": pred["vx"].tolist(),
        "predicted_steering": pred["steering"].tolist(),
        "predicted_throttle": pred["throttle"].tolist(),
        "stop_reason": pred["stop_reason"],
        "warnings": list(traj.warnings),
        "v_target": float(v_target),
        "clamped_vx": bool(traj.clamped_vx),
        "num_prediction_steps": int(len(pred["vx"]) - 1),
        "controller": controller_type,
        "mode": mode,
    }


def predict_from_pixels(
    pixel_samples: List[Tuple[float, float, float]],
    transform: PixelWorldTransform,
    track: Track,
    loaded: LoadedModel,
    v_target: Optional[float] = None,
    max_steps: int = MAX_ROLLOUT_STEPS,
    controller_type: str = "pursuit",
) -> Dict[str, object]:
    """End-to-end: pixel samples -> world path -> state history -> rollout."""
    if len(pixel_samples) < 2:
        raise ValueError("Need at least 2 drawn points.")

    arr = np.asarray(pixel_samples, dtype=np.float64)
    world = transform.pixel_to_world_array(arr[:, :2])
    t = arr[:, 2] - arr[0, 2]

    traj = estimate_states_from_drawing(
        world_pts=world,
        times_s=t,
        nominal=loaded.nominal,
        mass=loaded.mass,
        wheelbase=loaded.wheelbase,
        horizon=loaded.horizon,
    )
    history, seed = build_history_window(traj, horizon=loaded.horizon)

    if v_target is None:
        v_target = float(np.clip(np.mean(traj.vx[-loaded.horizon:]), 10.0, 45.0))

    # Auto-detect traversal direction: dot the drawn tangent at the last sample
    # against the centerline tangent at the nearest centerline point. If they
    # point opposite ways the user drew the track backward, so pursuit should
    # also walk the centerline backward.
    reverse = _should_reverse_for_drawn(traj, track)

    return _run_rollout_and_pack(
        loaded=loaded,
        track=track,
        history=history,
        seed=seed,
        traj=traj,
        transform=transform,
        v_target=float(v_target),
        max_steps=max_steps,
        controller_type=controller_type,
        mode="drawn",
        reverse_pursuit=reverse,
    )


def _should_reverse_for_drawn(traj: DrawnTrajectory, track: Track) -> bool:
    """Decide whether pursuit should traverse the centerline backward based on
    the direction of the last ~0.5s of the drawn line vs. the local centerline
    tangent. Returns True iff the dot product is negative."""
    if len(traj.world_xy) < 3:
        return False
    end = traj.world_xy[-1]
    start_idx = max(0, len(traj.world_xy) - 12)  # ~0.48s at Ts=0.04
    tangent_draw = end - traj.world_xy[start_idx]
    norm = np.linalg.norm(tangent_draw)
    if norm < 1e-6:
        return False
    tangent_draw /= norm

    _i, theta = centerline_heading_at(track, float(end[0]), float(end[1]))
    tangent_center = np.array([math.cos(theta), math.sin(theta)])
    return float(np.dot(tangent_draw, tangent_center)) < 0.0


def predict_from_point(
    pixel_x: float,
    pixel_y: float,
    transform: PixelWorldTransform,
    track: Track,
    loaded: LoadedModel,
    v_target: Optional[float] = None,
    max_steps: int = MAX_ROLLOUT_STEPS,
    controller_type: str = "pursuit",
    heading: Optional[float] = None,
    flip_heading: bool = False,
    start_speed: Optional[float] = None,
) -> Dict[str, object]:
    """End-to-end: a single clicked pixel -> synthetic history -> rollout.

    The start heading defaults to the centerline tangent at the closest point
    (with optional 180° flip so the user can go the other way around the
    track). Start speed defaults to v_target (clamped to the training regime).
    """
    world_xy = transform.pixel_to_world_array(
        np.array([[pixel_x, pixel_y]], dtype=np.float64)
    )[0]
    world_x, world_y = float(world_xy[0]), float(world_xy[1])

    # Auto-derive heading from the centerline tangent if not provided.
    if heading is None:
        _, heading = centerline_heading_at(track, world_x, world_y)
    if flip_heading:
        heading = heading + math.pi

    if v_target is None:
        v_target = 25.0
    v_target = float(np.clip(v_target, 8.0, MAX_VX))
    speed = float(start_speed) if start_speed is not None else v_target
    speed = float(np.clip(speed, MIN_VX, MAX_VX))

    history, seed, traj = build_history_from_start_point(
        world_x=world_x,
        world_y=world_y,
        heading=heading,
        speed=speed,
        horizon=loaded.horizon,
        nominal=loaded.nominal,
        mass=loaded.mass,
    )

    result = _run_rollout_and_pack(
        loaded=loaded,
        track=track,
        history=history,
        seed=seed,
        traj=traj,
        transform=transform,
        v_target=v_target,
        max_steps=max_steps,
        controller_type=controller_type,
        mode="point",
        reverse_pursuit=bool(flip_heading),
    )
    # Expose the resolved heading / speed back to the UI so it can render
    # the start arrow exactly where the backend placed it.
    result["start_world"] = [world_x, world_y]
    result["start_heading"] = float(heading)
    result["start_speed"] = speed
    # Overwrite `drawn_*` fields with empty lists in point-mode — the "drawn"
    # trajectory is synthetic and meaningless to render as a polyline.
    result["drawn_world"] = []
    result["drawn_pixel"] = []
    return result
