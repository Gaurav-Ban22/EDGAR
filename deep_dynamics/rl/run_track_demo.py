#!/usr/bin/env python3
"""

Loads track boundary CSVs from deep_dynamics/visualize/tracks/, prints
geometry stats, and writes PNG plots. Run from any directory:

    python -m deep_dynamics.rl.run_track_demo
    python deep_dynamics/rl/run_track_demo.py

Options: --track {lvms,putnam,all}  --output-dir DIR  --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from deep_dynamics.rl.track import Track


def _tracks_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "visualize" / "tracks"


def _demo_one(track: Track, output_dir: Path, plot: bool) -> None:
    print(track)
    print(f"  start_xy:        {track.start_xy}")
    print(f"  start_heading:   {track.start_heading:.6f} rad")
    print(f"  total_length:    {track.total_length:.2f} m")
    print(f"  centerline pts:  {len(track.centerline)}")
    print(
        f"  half_width:      [{track.half_widths.min():.2f}, "
        f"{track.half_widths.max():.2f}] m (per side)"
    )
    idx, s, e_lat, heading = track.project(*track.start_xy)
    print(f"  project(start):  idx={idx}, s={s:.4f}, e_lat={e_lat:.6f}, heading={heading:.6f}")
    print(f"  is_inside(start): {track.is_inside(*track.start_xy)}")
    print(f"  is_inside(far):   {track.is_inside(9999.0, 9999.0)}")
    print(f"  curvature_at(0): {track.curvature_at(0):.6f} 1/m")
    print(f"  width_at(0):     {track.width_at(0):.2f} m")
    print()

    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{track.name}_track_demo.png"
        ax = track.plot(car_xy=track.start_xy)
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"  saved plot: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Track module demo (load, print, plot).")
    parser.add_argument(
        "--track",
        choices=("lvms", "putnam", "all"),
        default="all",
        help="Which track to load (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: this rl/ folder)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib PNG export")
    args = parser.parse_args(argv)

    tdir = _tracks_dir()
    if not tdir.is_dir():
        print(f"error: track directory not found: {tdir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    names = ["lvms", "putnam"] if args.track == "all" else [args.track]
    plot = not args.no_plot

    print(f"tracks dir: {tdir}")
    print(f"output dir: {output_dir}")
    print()

    for name in names:
        inner = tdir / f"{name}_inner_bound.csv"
        outer = tdir / f"{name}_outer_bound.csv"
        if not inner.is_file() or not outer.is_file():
            print(f"error: missing CSV for {name!r}", file=sys.stderr)
            return 1
        track = Track.from_csv(str(inner), str(outer), name=name)
        _demo_one(track, output_dir, plot)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
