#!/usr/bin/env python3
"""
Browser-based localhost demo for the SAC racecar policy.

From EDGAR/:

    python -m deep_dynamics.rl.scripts.drive_server \
        --checkpoint deep_dynamics/rl_runs/final_models/final_model_v3.pt \
        --speed 2
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from deep_dynamics.rl.agent.sac import SAC
from deep_dynamics.rl.configs import load_rl_config
from deep_dynamics.rl.environment.racing_env import RacingEnv


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>EDGAR RL Racecar Demo</title>
  <style>
    body { margin: 0; background: #0b1020; color: #e7edf7; font-family: system-ui, sans-serif; }
    #bar { display: flex; align-items: center; gap: 14px; padding: 12px 16px; background: #121a2f; }
    button { font-size: 15px; padding: 7px 12px; border-radius: 8px; border: 0; cursor: pointer; }
    #status { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
    canvas { display: block; width: 100vw; height: calc(100vh - 58px); background: #07101d; }
  </style>
</head>
<body>
  <div id="bar">
    <button onclick="restart()">Restart Random Spawn</button>
    <div id="status">Loading...</div>
  </div>
  <canvas id="canvas"></canvas>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
let track = null;
let state = null;
let bounds = null;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener("resize", resize);

function includeBounds(points) {
  for (const p of points || []) {
    bounds.minX = Math.min(bounds.minX, p[0]);
    bounds.maxX = Math.max(bounds.maxX, p[0]);
    bounds.minY = Math.min(bounds.minY, p[1]);
    bounds.maxY = Math.max(bounds.maxY, p[1]);
  }
}

function computeBounds() {
  bounds = {minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity};
  includeBounds(track.inner);
  includeBounds(track.outer);
  includeBounds(track.raceline);
  const pad = 40;
  bounds.minX -= pad; bounds.maxX += pad; bounds.minY -= pad; bounds.maxY += pad;
}

function worldToCanvas(x, y) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const sx = w / (bounds.maxX - bounds.minX);
  const sy = h / (bounds.maxY - bounds.minY);
  const s = Math.min(sx, sy);
  const ox = (w - (bounds.maxX - bounds.minX) * s) / 2;
  const oy = (h - (bounds.maxY - bounds.minY) * s) / 2;
  return [ox + (x - bounds.minX) * s, h - (oy + (y - bounds.minY) * s)];
}

function drawPolyline(points, color, width, alpha=1) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  let [x0, y0] = worldToCanvas(points[0][0], points[0][1]);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < points.length; i++) {
    const [x, y] = worldToCanvas(points[i][0], points[i][1]);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function draw() {
  if (!track || !bounds) return;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawPolyline(track.outer, "#94a3b8", 2, 0.9);
  drawPolyline(track.inner, "#94a3b8", 2, 0.9);
  drawPolyline(track.centerline, "#334155", 1, 0.8);
  drawPolyline(track.raceline, "#f59e0b", 2, 0.85);
  if (state && state.traj) drawPolyline(state.traj, "#38bdf8", 2, 0.9);
  if (!state) return;

  const [cx, cy] = worldToCanvas(state.x, state.y);
  ctx.fillStyle = state.done ? "#ef4444" : "#22c55e";
  ctx.beginPath();
  ctx.arc(cx, cy, 7, 0, Math.PI * 2);
  ctx.fill();

  const arrowLen = 22;
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + arrowLen * Math.cos(-state.heading), cy + arrowLen * Math.sin(-state.heading));
  ctx.stroke();
}

async function restart() {
  await fetch("/reset", {method: "POST"});
  await tick();
}

async function tick() {
  try {
    const res = await fetch("/state");
    state = await res.json();
    statusEl.textContent =
      `${state.done ? "Ended: " + state.reason : "Driving"} | ` +
      `step ${state.step} | laps ${state.laps} | ` +
      `vx ${state.vx.toFixed(1)} m/s | d ${state.d.toFixed(2)} m | ` +
      `${state.speed.toFixed(1)}x`;
    draw();
  } catch (err) {
    statusEl.textContent = "Server connection lost";
  }
  setTimeout(tick, track ? track.frame_ms : 40);
}

async function init() {
  track = await (await fetch("/track")).json();
  computeBounds();
  resize();
  await restart();
}
init();
</script>
</body>
</html>
"""


def _downsample(points: np.ndarray, max_points: int = 1200) -> list[list[float]]:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) > max_points:
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        arr = arr[idx]
    return arr[:, :2].round(3).tolist()


class DemoState:
    def __init__(
        self,
        env: RacingEnv,
        agent: SAC,
        *,
        speed: float,
        frame_ms: int,
        max_traj: int = 5000,
    ) -> None:
        self.env = env
        self.agent = agent
        self.speed = float(speed)
        self.frame_ms = int(frame_ms)
        self.steps_per_tick = max(1, int(round(self.speed)))
        self.max_traj = int(max_traj)
        self.lock = threading.Lock()
        self.obs: Optional[np.ndarray] = None
        self.traj: list[list[float]] = []
        self.done = True
        self.reason = "not started"
        self.step_idx = 0
        self.reset()

    def reset(self) -> None:
        with self.lock:
            seed = int(np.random.randint(0, 2**31 - 1))
            self.obs, _ = self.env.reset(seed=seed, options={"randomize_start": True})
            self.traj = [[float(self.env.pose.x), float(self.env.pose.y)]]
            self.done = False
            self.reason = ""
            self.step_idx = 0

    def step(self) -> Dict[str, Any]:
        with self.lock:
            if not self.done and self.obs is not None:
                for _ in range(self.steps_per_tick):
                    action = self.agent.select_action(self.obs, deterministic=True)
                    self.obs, _r, term, trunc, info = self.env.step(action)
                    self.step_idx += 1
                    self.traj.append([float(info["x"]), float(info["y"])])
                    if len(self.traj) > self.max_traj:
                        self.traj = self.traj[-self.max_traj :]
                    if term or trunc:
                        self.done = True
                        self.reason = info.get("termination_reason") or (
                            "timeout" if trunc else "done"
                        )
                        break
            return {
                "x": float(self.env.pose.x),
                "y": float(self.env.pose.y),
                "heading": float(self.env.pose.heading),
                "vx": float(getattr(self.env, "_vx", 0.0)),
                "d": float(getattr(self.env, "_d", 0.0)),
                "s": float(getattr(self.env, "_s", 0.0)),
                "laps": int(getattr(self.env, "lap_count", 0)),
                "step": int(self.step_idx),
                "done": bool(self.done),
                "reason": self.reason,
                "speed": float(self.speed),
                "traj": self.traj,
            }

    def track_payload(self) -> Dict[str, Any]:
        raceline = []
        if getattr(self.env, "_use_raceline", False) and self.env.raceline is not None:
            raceline = _downsample(self.env.raceline.xy)
        return {
            "inner": _downsample(self.env.track.inner),
            "outer": _downsample(self.env.track.outer),
            "centerline": _downsample(self.env.track.centerline),
            "raceline": raceline,
            "frame_ms": self.frame_ms,
        }


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/track":
                self._send_json(state.track_payload())
            elif path == "/state":
                self._send_json(state.step())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/reset":
                state.reset()
                self._send_json({"ok": True})
            else:
                self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    here = Path(__file__).resolve()
    default_cfg = here.parent.parent / "configs" / "default.yaml"
    p = argparse.ArgumentParser(description="Localhost browser demo for SAC driving")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier")
    p.add_argument("--frame-ms", type=int, default=40)
    p.add_argument("--legacy-policy", action="store_true")
    p.add_argument("--open", action="store_true", help="Open browser automatically")
    args = p.parse_args()

    env_overrides = {"use_raceline": False} if args.legacy_policy else None
    cfg = load_rl_config(args.config.expanduser().resolve(), env_overrides=env_overrides)
    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        p.error(f"checkpoint not found: {ckpt}")

    env = RacingEnv(cfg.env)
    agent = SAC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        config=cfg.sac_agent,
        device=cfg.device,
    )
    agent.load(ckpt, load_optimizers=False)

    state = DemoState(env, agent, speed=args.speed, frame_ms=args.frame_ms)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"
    print(f"Serving RL demo at {url}")
    print(f"checkpoint: {ckpt}")
    print(f"playback: {args.speed:.1f}x  frame_ms={args.frame_ms}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
