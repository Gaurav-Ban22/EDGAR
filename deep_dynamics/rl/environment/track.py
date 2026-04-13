"""
Track module for RL environment.

Loads inner/outer boundary CSVs, computes a centerline, arc-length
parameterization, local heading, curvature, and width. Provides fast
projection (nearest centerline point → progress, lateral offset) and
boundary-containment queries used by the Gym environment.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _load_boundary_csv(path: str) -> NDArray:
    """Load a boundary CSV (x,y[,z]) and return Nx2 array (z is dropped)."""
    pts = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            pts.append([float(row[0]), float(row[1])])
    return np.array(pts, dtype=np.float64)


def _cumulative_arc_length(pts: NDArray) -> NDArray:
    """Return 1-d array of cumulative arc lengths (first entry = 0)."""
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _finite_diff_heading(pts: NDArray) -> NDArray:
    """Heading angle (rad) at each point via central finite differences.

    Wraps around for closed tracks (first/last share the same heading).
    """
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    return np.arctan2(dy, dx)


def _finite_diff_curvature(pts: NDArray, headings: NDArray) -> NDArray:
    """Signed curvature (1/m) at each centerline vertex.

    Uses d(heading)/ds where s is arc length.
    """
    arc = _cumulative_arc_length(pts)
    ds = np.gradient(arc)
    ds = np.clip(ds, 1e-9, None)
    dtheta = np.gradient(np.unwrap(headings))
    return dtheta / ds


def _close_loop(pts: NDArray, tol: float = 0.5) -> NDArray:
    """If the boundary is an open polygon, close it by appending the first
    point.  If it is already closed (within *tol*), return as-is."""
    if np.linalg.norm(pts[-1] - pts[0]) > tol:
        return np.vstack([pts, pts[0:1]])
    return pts


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Racetrack geometry built from inner/outer boundary CSVs.

    After construction the following arrays are available (all Nx2 or N):

    * ``centerline``  – Nx2 centerline waypoints
    * ``center_s``    – cumulative arc-length at each centerline vertex
    * ``center_heading`` – heading (rad) at each vertex
    * ``center_curvature`` – signed curvature (1/m) at each vertex
    * ``half_widths``  – half-track-width at each centerline vertex
    * ``inner``, ``outer`` – raw boundary arrays (Mx2 each)
    * ``total_length`` – total centerline arc length (m)
    """

    name: str
    inner: NDArray
    outer: NDArray
    centerline: NDArray = field(init=False)
    center_s: NDArray = field(init=False)
    center_heading: NDArray = field(init=False)
    center_curvature: NDArray = field(init=False)
    half_widths: NDArray = field(init=False)
    total_length: float = field(init=False)
    _n: int = field(init=False, repr=False)

    # -- construction -------------------------------------------------------

    def __post_init__(self) -> None:
        self.inner = _close_loop(self.inner)
        self.outer = _close_loop(self.outer)
        self._build_centerline()

    @classmethod
    def from_csv(
        cls,
        inner_csv: str,
        outer_csv: str,
        name: Optional[str] = None,
    ) -> "Track":
        """Construct a Track from two boundary CSV files."""
        inner = _load_boundary_csv(inner_csv)
        outer = _load_boundary_csv(outer_csv)
        if name is None:
            name = os.path.splitext(os.path.basename(inner_csv))[0].replace("_inner_bound", "")
        return cls(name=name, inner=inner, outer=outer)

    @classmethod
    def from_track_dir(cls, track_dir: str, track_name: str) -> "Track":
        """Load ``<track_name>_inner_bound.csv`` and ``..._outer_bound.csv``
        from *track_dir*."""
        inner_csv = os.path.join(track_dir, f"{track_name}_inner_bound.csv")
        outer_csv = os.path.join(track_dir, f"{track_name}_outer_bound.csv")
        return cls.from_csv(inner_csv, outer_csv, name=track_name)

    # -- internal build -----------------------------------------------------

    def _build_centerline(self) -> None:
        """Match inner ↔ outer boundaries and average to get centerline."""
        n_center = min(len(self.inner), len(self.outer))

        inner_s = _cumulative_arc_length(self.inner)
        outer_s = _cumulative_arc_length(self.outer)

        inner_s_norm = inner_s / inner_s[-1]
        outer_s_norm = outer_s / outer_s[-1]

        # Resample both boundaries to the same number of equi-spaced
        # normalised arc-length stations so they can be averaged pointwise.
        t = np.linspace(0.0, 1.0, n_center, endpoint=False)
        inner_resampled = np.column_stack([
            np.interp(t, inner_s_norm, self.inner[:, 0]),
            np.interp(t, inner_s_norm, self.inner[:, 1]),
        ])
        outer_resampled = np.column_stack([
            np.interp(t, outer_s_norm, self.outer[:, 0]),
            np.interp(t, outer_s_norm, self.outer[:, 1]),
        ])

        self.centerline = 0.5 * (inner_resampled + outer_resampled)
        self._n = len(self.centerline)
        self.center_s = _cumulative_arc_length(self.centerline)
        self.total_length = float(self.center_s[-1])
        self.center_heading = _finite_diff_heading(self.centerline)
        self.center_curvature = _finite_diff_curvature(
            self.centerline, self.center_heading
        )

        # Half-width = distance from each centerline point to nearest inner
        # boundary point (approximation; symmetric tracks make this exact).
        self.half_widths = np.array([
            0.5 * np.linalg.norm(inner_resampled[i] - outer_resampled[i])
            for i in range(self._n)
        ])

    # -- projection ---------------------------------------------------------

    def project(
        self, x: float, y: float
    ) -> Tuple[int, float, float, float]:
        """Project a world-frame point onto the centerline.

        Returns
        -------
        idx : int
            Index of the closest centerline vertex.
        s : float
            Arc-length progress along the centerline (m).  Wraps at
            ``total_length``.
        e_lat : float
            Signed lateral offset from the centerline (m).
            Positive = left of the heading direction (right-hand rule).
        heading : float
            Centerline heading (rad) at the projected point.
        """
        pt = np.array([x, y])
        diffs = self.centerline - pt
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dists_sq))

        s = self.center_s[idx]
        heading = self.center_heading[idx]

        # signed lateral offset: positive = left of heading
        dx = x - self.centerline[idx, 0]
        dy = y - self.centerline[idx, 1]
        # normal is 90° CCW from heading
        e_lat = -dx * np.sin(heading) + dy * np.cos(heading)

        return idx, float(s), float(e_lat), float(heading)

    def cartesian_to_frenet(
        self, x: float, y: float
    ) -> Tuple[float, float, float]:
        """Alias for :meth:`project` in (s, d, heading) form (Phase 4 / plan API)."""
        _, s, e_lat, heading = self.project(x, y)
        return s, e_lat, heading

    def update_progress(self, s_prev: float, s_new: float) -> Tuple[float, bool]:
        """Arc-length step and whether the agent crossed the finish (forward wrap).

        Returns
        -------
        delta_s
            Shortest signed progress along the loop from *s_prev* to *s_new*.
        crossed_finish_line
            True when the shortest forward path wraps past the start line.
        """
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

    def frenet_to_cartesian(
        self, s: float, e_lat: float
    ) -> Tuple[float, float, float]:
        """Convert Frenet (s, e_lat) back to Cartesian (x, y, heading).

        Uses linear interpolation between centerline vertices.
        """
        s_mod = s % self.total_length
        idx = int(np.searchsorted(self.center_s, s_mod, side="right")) - 1
        idx = np.clip(idx, 0, self._n - 2)

        frac = (s_mod - self.center_s[idx]) / max(
            self.center_s[idx + 1] - self.center_s[idx], 1e-9
        )
        cx = self.centerline[idx, 0] + frac * (
            self.centerline[idx + 1, 0] - self.centerline[idx, 0]
        )
        cy = self.centerline[idx, 1] + frac * (
            self.centerline[idx + 1, 1] - self.centerline[idx, 1]
        )
        heading = self.center_heading[idx]

        nx = -np.sin(heading)
        ny = np.cos(heading)
        return float(cx + e_lat * nx), float(cy + e_lat * ny), float(heading)

    # -- queries ------------------------------------------------------------

    def is_inside(self, x: float, y: float) -> bool:
        """Return True if (x, y) lies between the inner and outer boundaries."""
        idx, _, e_lat, _ = self.project(x, y)
        return bool(abs(e_lat) <= self.half_widths[idx])

    def normalized_progress(self, s: float) -> float:
        """Return progress in [0, 1) around the track."""
        return (s % self.total_length) / self.total_length

    def heading_at(self, s: float) -> float:
        """Interpolated heading (rad) at arc-length *s*."""
        s_mod = s % self.total_length
        return float(np.interp(s_mod, self.center_s, np.unwrap(self.center_heading)) % (2 * np.pi))

    def curvature_at(self, s: float) -> float:
        """Interpolated signed curvature (1/m) at arc-length *s*."""
        s_mod = s % self.total_length
        return float(np.interp(s_mod, self.center_s, self.center_curvature))

    def width_at(self, s: float) -> float:
        """Full track width (m) at arc-length *s*."""
        s_mod = s % self.total_length
        return float(2.0 * np.interp(s_mod, self.center_s, self.half_widths))

    def lookahead_curvatures(
        self, s: float, distances: NDArray
    ) -> NDArray:
        """Return curvature values at ``s + distances[i]`` for each entry in
        *distances*.  Useful for giving the RL agent preview of upcoming
        track geometry."""
        return np.array([self.curvature_at(s + d) for d in distances])

    # -- visualisation ------------------------------------------------------

    def plot(
        self,
        ax=None,
        show_centerline: bool = True,
        show_boundaries: bool = True,
        car_xy: Optional[Tuple[float, float]] = None,
    ):
        """Quick matplotlib plot of the track.  Returns the Axes object."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        if show_boundaries:
            ax.plot(self.inner[:, 0], self.inner[:, 1], "k-", lw=1, label="inner")
            ax.plot(self.outer[:, 0], self.outer[:, 1], "k-", lw=1, label="outer")
        if show_centerline:
            ax.plot(
                self.centerline[:, 0],
                self.centerline[:, 1],
                "--",
                color="gray",
                lw=0.8,
                label="centerline",
            )
        if car_xy is not None:
            ax.plot(car_xy[0], car_xy[1], "ro", markersize=6, label="car")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Track: {self.name}")
        ax.legend(fontsize=8)
        return ax

    # -- convenience --------------------------------------------------------

    @property
    def cumulative_s(self) -> NDArray:
        """Same as ``center_s`` (plan / legacy name)."""
        return self.center_s

    @property
    def headings(self) -> NDArray:
        """Same as ``center_heading`` (plan / legacy name)."""
        return self.center_heading

    def is_within_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Plan API: inside track; *margin* shrinks the allowed half-width."""
        idx, _, e_lat, _ = self.project(x, y)
        hw = float(self.half_widths[idx]) - margin
        return bool(abs(e_lat) <= max(hw, 0.0))

    @property
    def start_xy(self) -> Tuple[float, float]:
        """(x, y) of the first centerline waypoint."""
        return float(self.centerline[0, 0]), float(self.centerline[0, 1])

    @property
    def start_heading(self) -> float:
        """Heading (rad) at the first centerline waypoint."""
        return float(self.center_heading[0])

    def __repr__(self) -> str:
        return (
            f"Track(name={self.name!r}, vertices={self._n}, "
            f"length={self.total_length:.1f}m)"
        )
