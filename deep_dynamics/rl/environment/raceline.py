"""
Raceline module: build an expert-trajectory reference from real car telemetry.

The raceline is the curve the car actually drove during a fast lap of real
data. Using this as the RL reference (instead of the geometric centerline)
gives three benefits:

1. **Correct racing line** — apex-in-out, not geometric center.
2. **Correct speed profile** — :math:`v_{ref}(s)` comes from what a driver
   actually achieved at every point, not a quasi-static
   :math:`\\sqrt{a_{lat}/\\kappa}` estimate.
3. **Reduced model exploitation** — staying near this curve keeps the
   Deep Dynamics network in its training distribution, where its
   predictions are accurate.

The telemetry CSV (``deep_dynamics/data/LVMS_23_01_04_A.csv``) has the format
produced by the EDGAR stack::

    time(s), x(m), y(m), vx(m/s), vy(m/s), phi(rad), ...

We (a) segment it into laps by detecting arc-length wraps against a supplied
centerline track, (b) pick the fastest *full* lap by wall-clock duration,
(c) smooth and resample at uniform arc-length spacing, and (d) compute
heading / curvature / speed-profile arrays on the resampled curve.

Exposed API mirrors :class:`deep_dynamics.rl.environment.track.Track` so
``RacingEnv`` can use either interchangeably for Frenet projection and
lookahead curvature.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# CSV parsing + lap segmentation
# ---------------------------------------------------------------------------

# Columns of the EDGAR IAC telemetry CSVs, confirmed against
# deep_dynamics/data/LVMS_23_01_04_A.csv header.
_COL_TIME = 0
_COL_X = 1
_COL_Y = 2
_COL_VX = 3
_COL_VY = 4
_COL_PHI = 5


def _load_telemetry_csv(path: str) -> NDArray:
    """Return an (N, 6) array of ``[time, x, y, vx, vy, phi]`` floats.

    The header row (first line, starts with ``#`` or contains ``time``) is
    auto-skipped. Malformed rows are skipped silently rather than crashing.
    """
    rows: List[List[float]] = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip comment / header row.
            first = row[0].strip()
            if first.startswith("#") or first.startswith("time"):
                continue
            try:
                vals = [
                    float(row[_COL_TIME]),
                    float(row[_COL_X]),
                    float(row[_COL_Y]),
                    float(row[_COL_VX]),
                    float(row[_COL_VY]),
                    float(row[_COL_PHI]),
                ]
            except (ValueError, IndexError):
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No usable telemetry rows found in {path}")
    return np.array(rows, dtype=np.float64)


def _segment_laps_by_s(
    s_series: NDArray,
    total_length: float,
) -> List[Tuple[int, int]]:
    """Split a time-series of arc-length values into per-lap index ranges.

    A "lap boundary" is a large negative jump (``s`` wraps from ~L back to 0).
    Returns a list of ``(i_start, i_end_inclusive)`` tuples. Partial laps at
    the start and end of the series are dropped.
    """
    if len(s_series) < 2:
        return []
    diffs = np.diff(s_series)
    wrap_threshold = -0.5 * total_length
    wrap_indices = np.where(diffs < wrap_threshold)[0] + 1
    if len(wrap_indices) < 2:
        return []
    laps: List[Tuple[int, int]] = []
    for a, b in zip(wrap_indices[:-1], wrap_indices[1:]):
        laps.append((int(a), int(b) - 1))
    return laps


def _pick_fastest_full_lap(
    telem: NDArray,
    s_series: NDArray,
    total_length: float,
    *,
    min_length_frac: float = 0.90,
) -> Tuple[int, int, float]:
    """Return ``(i_start, i_end, duration_s)`` of the fastest complete lap.

    A lap is "full" if its ``s`` range spans at least ``min_length_frac`` of
    the track's total arc length — this filters out degenerate segments
    (e.g. the car stopped in pit lane mid-segment).
    """
    laps = _segment_laps_by_s(s_series, total_length)
    if not laps:
        raise ValueError(
            "Could not segment telemetry into laps — no arc-length wraps "
            "detected. Check that the telemetry and the track actually "
            "correspond to the same circuit."
        )
    best: Optional[Tuple[int, int, float]] = None
    for i_start, i_end in laps:
        seg = s_series[i_start : i_end + 1]
        if seg.size < 10:
            continue
        # arc-length coverage within this segment (handles start/end near wrap)
        s_min, s_max = float(seg.min()), float(seg.max())
        coverage = (s_max - s_min) / total_length
        if coverage < min_length_frac:
            continue
        t_start = float(telem[i_start, _COL_TIME])
        t_end = float(telem[i_end, _COL_TIME])
        duration = t_end - t_start
        if duration <= 0:
            continue
        if best is None or duration < best[2]:
            best = (i_start, i_end, duration)
    if best is None:
        raise ValueError(
            "No full lap found in telemetry "
            f"(min coverage {min_length_frac:.0%} of track length)."
        )
    return best


# ---------------------------------------------------------------------------
# Smoothing + resampling
# ---------------------------------------------------------------------------

def _moving_average_1d(x: NDArray, window: int) -> NDArray:
    """Centered moving average with reflect padding."""
    window = max(1, int(window))
    if window <= 1:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def _cumulative_arc_length(pts: NDArray) -> NDArray:
    diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_uniform(
    pts: NDArray,
    aux: NDArray,
    ds: float,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Resample ``pts`` (Nx2) and per-point aux array ``aux`` (N,) at uniform
    arc-length spacing ``ds``. Returns ``(pts_u, aux_u, s_u)``.
    """
    s_raw = _cumulative_arc_length(pts)
    total = float(s_raw[-1])
    n = max(8, int(np.round(total / ds)))
    s_u = np.linspace(0.0, total, n, endpoint=False)
    pts_u = np.column_stack([
        np.interp(s_u, s_raw, pts[:, 0]),
        np.interp(s_u, s_raw, pts[:, 1]),
    ])
    aux_u = np.interp(s_u, s_raw, aux)
    return pts_u, aux_u, s_u


def _finite_diff_heading(pts: NDArray) -> NDArray:
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    return np.arctan2(dy, dx)


def _finite_diff_curvature(pts: NDArray, headings: NDArray) -> NDArray:
    s = _cumulative_arc_length(pts)
    ds = np.gradient(s)
    ds = np.clip(ds, 1e-9, None)
    dtheta = np.gradient(np.unwrap(headings))
    return dtheta / ds


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class Raceline:
    """Expert reference trajectory parameterized by arc length.

    After construction these arrays are available (all shape ``(N,)`` unless
    noted):

    * ``xy``            – (N, 2) resampled raceline points
    * ``s``             – cumulative arc length at each vertex
    * ``heading``       – tangent heading (rad)
    * ``curvature``     – signed curvature (1/m)
    * ``vx_profile``    – longitudinal speed (m/s) at each vertex
    * ``total_length``  – total arc length of the lap (m)
    """

    name: str
    xy: NDArray
    s: NDArray
    heading: NDArray
    curvature: NDArray
    vx_profile: NDArray
    total_length: float
    _n: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._n = len(self.xy)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_telemetry(
        cls,
        telemetry_csv: str,
        track,  # deep_dynamics.rl.environment.track.Track (avoid circular import)
        *,
        smooth_window_xy: int = 15,
        smooth_window_vx: int = 25,
        resample_ds: float = 2.0,
        min_lap_coverage: float = 0.90,
        name: Optional[str] = None,
    ) -> "Raceline":
        """Extract the fastest lap from ``telemetry_csv`` and build a raceline.

        Parameters
        ----------
        telemetry_csv
            Path to an EDGAR-format telemetry CSV (``time, x, y, vx, vy, phi, ...``).
        track
            A :class:`Track` used for (a) arc-length projection when detecting
            lap boundaries and (b) deriving the default raceline name.
        smooth_window_xy
            Moving-average window (samples) applied to ``x`` and ``y`` before
            arc-length parameterization. 15 ≈ 0.6 s at 25 Hz — removes sensor
            noise without blurring corner geometry.
        smooth_window_vx
            Moving-average window applied to ``vx`` after resampling. Larger
            than the xy window because speed is noisier than position.
        resample_ds
            Target arc-length spacing (m) between raceline vertices.
            2 m → ~1000 points on an LVMS lap (~2 km).
        min_lap_coverage
            Minimum fraction of the track length a segment must span to
            count as a "full" lap.
        """
        telem = _load_telemetry_csv(telemetry_csv)

        # Project every telemetry sample onto the track centerline so we get
        # an arc-length time series — this is what lets us detect lap wraps.
        s_series = np.empty(len(telem), dtype=np.float64)
        for i in range(len(telem)):
            s_series[i], _, _ = track.cartesian_to_frenet(
                float(telem[i, _COL_X]), float(telem[i, _COL_Y])
            )

        i_start, i_end, duration = _pick_fastest_full_lap(
            telem,
            s_series,
            total_length=track.total_length,
            min_length_frac=min_lap_coverage,
        )

        lap = telem[i_start : i_end + 1]
        xy_raw = lap[:, [_COL_X, _COL_Y]]
        vx_raw = lap[:, _COL_VX]

        # Smooth position and speed before arc-length parameterization.
        x_s = _moving_average_1d(xy_raw[:, 0], smooth_window_xy)
        y_s = _moving_average_1d(xy_raw[:, 1], smooth_window_xy)
        xy_smooth = np.column_stack([x_s, y_s])

        # Close the lap: make first == last point (within the smoothed trace)
        # so resampling and curvature finite-diffs don't have a seam.
        xy_closed = np.vstack([xy_smooth, xy_smooth[:1]])
        vx_closed = np.concatenate([vx_raw, vx_raw[:1]])

        xy_u, vx_u, s_u = _resample_uniform(xy_closed, vx_closed, resample_ds)

        vx_u = _moving_average_1d(vx_u, smooth_window_vx)

        heading = _finite_diff_heading(xy_u)
        curvature = _finite_diff_curvature(xy_u, heading)

        total_length = (
            float(s_u[-1] + (s_u[1] - s_u[0])) if len(s_u) > 1 else float(s_u[-1])
        )

        if name is None:
            base = os.path.splitext(os.path.basename(telemetry_csv))[0]
            name = f"{base}_fastest_lap"

        rl = cls(
            name=name,
            xy=xy_u,
            s=s_u,
            heading=heading,
            curvature=curvature,
            vx_profile=vx_u,
            total_length=total_length,
        )
        # Expose the best-lap metadata on the instance for logging/debugging.
        rl.lap_duration = duration  # type: ignore[attr-defined]
        rl.source_row_range = (int(i_start), int(i_end))  # type: ignore[attr-defined]
        rl.mean_vx = float(np.mean(vx_u))  # type: ignore[attr-defined]
        rl.max_vx = float(np.max(vx_u))  # type: ignore[attr-defined]
        return rl

    # -- projection ---------------------------------------------------------

    def project(
        self, x: float, y: float
    ) -> Tuple[int, float, float, float]:
        """Project a world-frame point onto the raceline.

        Returns ``(idx, s, e_lat, heading)`` with the same sign convention as
        :meth:`Track.project` — positive ``e_lat`` means left of the tangent.
        """
        pt = np.array([x, y])
        diffs = self.xy - pt
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dists_sq))
        s = float(self.s[idx])
        heading = float(self.heading[idx])
        dx = x - self.xy[idx, 0]
        dy = y - self.xy[idx, 1]
        e_lat = -dx * np.sin(heading) + dy * np.cos(heading)
        return idx, s, float(e_lat), heading

    def cartesian_to_frenet(
        self, x: float, y: float
    ) -> Tuple[float, float, float]:
        """``(s, d, heading)`` relative to the raceline."""
        _, s, e_lat, heading = self.project(x, y)
        return s, e_lat, heading

    def update_progress(
        self, s_prev: float, s_new: float
    ) -> Tuple[float, bool]:
        """Shortest-path ``delta_s`` around the loop and whether it wrapped."""
        L = self.total_length
        if L <= 1e-9:
            return 0.0, False
        a = float(s_prev % L)
        b = float(s_new % L)
        raw = b - a
        if raw < -L / 2.0:
            return raw + L, True
        if raw > L / 2.0:
            return raw - L, False
        return raw, False

    # -- queries ------------------------------------------------------------

    def vx_at(self, s: float) -> float:
        """Interpolated reference speed (m/s) at arc length ``s``."""
        s_mod = float(s) % self.total_length
        return float(np.interp(s_mod, self.s, self.vx_profile))

    def curvature_at(self, s: float) -> float:
        s_mod = float(s) % self.total_length
        return float(np.interp(s_mod, self.s, self.curvature))

    def heading_at(self, s: float) -> float:
        s_mod = float(s) % self.total_length
        return float(np.interp(s_mod, self.s, np.unwrap(self.heading)))

    def lookahead_curvatures(
        self, s: float, distances: NDArray
    ) -> NDArray:
        return np.array([self.curvature_at(s + d) for d in distances])

    def lookahead_vx_refs(
        self, s: float, distances: NDArray
    ) -> NDArray:
        return np.array([self.vx_at(s + d) for d in distances])

    # -- convenience --------------------------------------------------------

    def start_pose(self) -> Tuple[float, float, float]:
        """``(x, y, heading)`` at ``s = 0``."""
        return (
            float(self.xy[0, 0]),
            float(self.xy[0, 1]),
            float(self.heading[0]),
        )

    def __repr__(self) -> str:
        return (
            f"Raceline(name={self.name!r}, vertices={self._n}, "
            f"length={self.total_length:.1f}m, "
            f"mean_vx={getattr(self, 'mean_vx', float('nan')):.1f}, "
            f"max_vx={getattr(self, 'max_vx', float('nan')):.1f})"
        )
