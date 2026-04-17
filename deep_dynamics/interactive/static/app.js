"use strict";

// Three stacked canvases: static track bounds, model prediction, user-drawn
// live path + start marker (top layer captures pointer events).
const stage = document.getElementById("stage");
const trackCanvas = document.getElementById("track");
const drawCanvas = document.getElementById("draw");
const predCanvas = document.getElementById("pred");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const btnClear = document.getElementById("btn-clear");
const btnFit = document.getElementById("btn-fit");
const btnZoomIn = document.getElementById("btn-zoom-in");
const btnZoomOut = document.getElementById("btn-zoom-out");
const zoomValueEl = document.getElementById("zoom-value");
const vTargetInput = document.getElementById("v-target");
const maxStepsInput = document.getElementById("max-steps");
const trackSelect = document.getElementById("track-id");
const modelSelect = document.getElementById("model-id");
const controllerSelect = document.getElementById("controller-id");
const hintPill = document.getElementById("hint-pill");
const actionPill = document.getElementById("action-pill");
const actionSub = document.getElementById("action-sub");
const btnFlip = document.getElementById("btn-flip");
const btnSimulate = document.getElementById("btn-simulate");
const btnCancelStart = document.getElementById("btn-cancel-start");

const trackCtx = trackCanvas.getContext("2d");
const drawCtx = drawCanvas.getContext("2d");
const predCtx = predCanvas.getContext("2d");

const COLORS = {
  outer: "#1d1d1f",
  inner: "#48484a",
  centerline: "rgba(99,99,102,0.45)",
  corridor: "rgba(0,113,227,0.05)",
  live: "#0071e3",
  drawn: "#32ade6",
  prediction: "#ff3b30",
  predictionHead: "#d70015",
  start: "#0071e3",
  startHalo: "rgba(0,113,227,0.18)",
};

const ZOOM_MIN = 1.0;
const ZOOM_MAX = 10.0;
const ZOOM_STEP_FACTOR = 1.25;

// Distinguishing a click from a drag: if total travel since pointerdown
// stays under CLICK_PX_THRESHOLD and the gesture releases within
// CLICK_MS_THRESHOLD, we treat it as a click (= place-start).
const CLICK_PX_THRESHOLD = 6;
const CLICK_MS_THRESHOLD = 320;

let canvasW = 0;
let canvasH = 0;
let trackData = null;
let samples = [];
let drawing = false;
let drawStartMs = 0;
let lastResult = null;
let startPoint = null;  // { px, py, flip } — null when no start is placed
let tracksById = {};

const view = { zoom: 1, panX: 0, panY: 0 };

let panning = false;
let panStart = null;
let trackRefreshTimer = null;

// Pointer gesture bookkeeping.
let pointerDownInfo = null; // { x, y, t, button }

function setStatus(msg, kind = "info") {
  statusEl.textContent = msg;
  statusEl.classList.remove("error", "warn");
  if (kind === "error") statusEl.classList.add("error");
  else if (kind === "warn") statusEl.classList.add("warn");
}

function resizeStageToWindow() {
  const top = stage.getBoundingClientRect().top;
  const w = Math.max(window.innerWidth - 56, 520);
  const h = Math.max(window.innerHeight - top - 220, 460);
  canvasW = Math.floor(w);
  canvasH = Math.floor(h);
  stage.style.height = canvasH + "px";
  for (const c of [trackCanvas, drawCanvas, predCanvas]) {
    c.width = canvasW;
    c.height = canvasH;
    c.style.width = canvasW + "px";
    c.style.height = canvasH + "px";
  }
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function fetchTrack() {
  const q = new URLSearchParams({
    canvas_w: canvasW,
    canvas_h: canvasH,
    zoom: view.zoom,
    pan_x: view.panX,
    pan_y: view.panY,
    track_id: trackSelect.value || "",
  });
  const r = await fetch(`/track?${q.toString()}`);
  if (!r.ok) throw new Error(`track fetch failed: ${r.status}`);
  return r.json();
}
async function fetchTracks() { const r = await fetch("/tracks"); if (!r.ok) throw new Error(`tracks fetch failed: ${r.status}`); return r.json(); }
async function fetchModels() { const r = await fetch("/models"); if (!r.ok) throw new Error(`models fetch failed: ${r.status}`); return r.json(); }
async function fetchControllers() { const r = await fetch("/controllers"); if (!r.ok) throw new Error(`controllers fetch failed: ${r.status}`); return r.json(); }

async function refreshTrack() {
  try {
    trackData = await fetchTrack();
    renderTrack();
    // Keep any stored prediction + start marker in sync with the new transform.
    if (lastResult) renderPrediction(lastResult);
    renderDrawLayer();
  } catch (e) {
    setStatus(`Track reload failed: ${e.message}`, "error");
  }
}

function scheduleTrackRefresh() {
  clearTimeout(trackRefreshTimer);
  trackRefreshTimer = setTimeout(refreshTrack, 40);
}

// ---------------------------------------------------------------------------
// Dropdowns
// ---------------------------------------------------------------------------

function populateTrackSelect(info) {
  trackSelect.innerHTML = "";
  tracksById = {};
  for (const t of info.tracks) {
    tracksById[t.id] = t;
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = t.label;
    if (t.notes) opt.title = t.notes;
    if (t.id === info.default) opt.selected = true;
    trackSelect.appendChild(opt);
  }
  applyTrackDefaults(trackSelect.value, { updateSelect: false });
}

function populateModelSelect(info) {
  modelSelect.innerHTML = "";
  for (const m of info.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.available ? m.label : `${m.label} (unavailable)`;
    if (!m.available) opt.disabled = true;
    if (m.notes) opt.title = m.notes;
    if (m.id === info.default) opt.selected = true;
    modelSelect.appendChild(opt);
  }
}

function populateControllerSelect(info) {
  controllerSelect.innerHTML = "";
  for (const c of info.controllers) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label;
    if (c.notes) opt.title = c.notes;
    if (c.id === info.default) opt.selected = true;
    controllerSelect.appendChild(opt);
  }
}

function applyTrackDefaults(trackId, { updateSelect = true } = {}) {
  const meta = tracksById[trackId];
  if (!meta) return;
  view.zoom = clampZoom(Number(meta.default_zoom) || 1);
  view.panX = 0;
  view.panY = 0;
  updateZoomUI();
  if (updateSelect) trackSelect.value = trackId;
}

function clampZoom(z) { return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z)); }
function formatZoom(z) { return `${Math.round(z * 100)}%`; }

function updateZoomUI() {
  if (zoomValueEl) zoomValueEl.textContent = formatZoom(view.zoom);
  if (btnZoomIn)  btnZoomIn.disabled  = view.zoom >= ZOOM_MAX - 1e-6;
  if (btnZoomOut) btnZoomOut.disabled = view.zoom <= ZOOM_MIN + 1e-6;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function drawPolyline(ctx, pts, { stroke, width = 1, dash = [] } = {}) {
  if (!pts || pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
  ctx.restore();
}

function renderTrack() {
  if (!trackData) return;
  trackCtx.clearRect(0, 0, canvasW, canvasH);

  const outer = trackData.outer_px;
  const inner = trackData.inner_px;
  if (outer && outer.length > 2 && inner && inner.length > 2) {
    trackCtx.save();
    trackCtx.fillStyle = COLORS.corridor;
    trackCtx.beginPath();
    trackCtx.moveTo(outer[0][0], outer[0][1]);
    for (let i = 1; i < outer.length; i++) trackCtx.lineTo(outer[i][0], outer[i][1]);
    trackCtx.closePath();
    trackCtx.moveTo(inner[0][0], inner[0][1]);
    for (let i = inner.length - 1; i >= 0; i--) trackCtx.lineTo(inner[i][0], inner[i][1]);
    trackCtx.closePath();
    trackCtx.fill("evenodd");
    trackCtx.restore();
  }

  drawPolyline(trackCtx, outer, { stroke: COLORS.outer, width: 1.8 });
  drawPolyline(trackCtx, inner, { stroke: COLORS.inner, width: 1.8 });
  drawPolyline(trackCtx, trackData.centerline_px, {
    stroke: COLORS.centerline,
    width: 1,
    dash: [4, 7],
  });
}

function worldToPixel(pt) {
  if (!trackData || !trackData.transform) return [0, 0];
  const tf = trackData.transform;
  return [(pt[0] - tf.x0) / tf.scale, (tf.y0 - pt[1]) / tf.scale];
}

// Find the centerline tangent (in pixel space) at the nearest centerline
// vertex to `(px, py)` so we can show a heading arrow for a placed start
// point without a round-trip to the server. The backend authoritatively
// computes heading from world coords; this is just for the preview.
function centerlineTangentAtPx(px, py) {
  if (!trackData || !trackData.centerline_px) return 0;
  const c = trackData.centerline_px;
  let best = 0, bestD2 = Infinity;
  for (let i = 0; i < c.length; i++) {
    const dx = c[i][0] - px;
    const dy = c[i][1] - py;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) { bestD2 = d2; best = i; }
  }
  const n = c.length;
  const prev = c[(best - 1 + n) % n];
  const next = c[(best + 1) % n];
  return Math.atan2(next[1] - prev[1], next[0] - prev[0]);
}

function drawStartMarker(ctx, px, py, headingRad) {
  const armLen = 46;
  const cos = Math.cos(headingRad);
  const sin = Math.sin(headingRad);

  // Soft accent halo.
  ctx.save();
  ctx.fillStyle = COLORS.startHalo;
  ctx.beginPath();
  ctx.arc(px, py, 16, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Heading arrow shaft.
  ctx.save();
  ctx.strokeStyle = COLORS.start;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + armLen * cos, py + armLen * sin);
  ctx.stroke();
  // Arrow head.
  const hx = px + armLen * cos;
  const hy = py + armLen * sin;
  const head = 10;
  const spread = 0.45;
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx - head * Math.cos(headingRad - spread),
             hy - head * Math.sin(headingRad - spread));
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx - head * Math.cos(headingRad + spread),
             hy - head * Math.sin(headingRad + spread));
  ctx.stroke();
  ctx.restore();

  // Core dot (outlined for visibility on any background).
  ctx.save();
  ctx.fillStyle = COLORS.start;
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.92)";
  ctx.stroke();
  ctx.restore();
}

// Renders BOTH in-progress samples and the start marker on the draw layer.
function renderDrawLayer() {
  drawCtx.clearRect(0, 0, canvasW, canvasH);
  if (samples.length >= 2) {
    const pts = samples.map((s) => [s.x, s.y]);
    drawPolyline(drawCtx, pts, { stroke: COLORS.live, width: 3.5 });
  }
  if (startPoint) {
    let heading = centerlineTangentAtPx(startPoint.px, startPoint.py);
    if (startPoint.flip) heading += Math.PI;
    drawStartMarker(drawCtx, startPoint.px, startPoint.py, heading);
  }
}

function clearPrediction() { predCtx.clearRect(0, 0, canvasW, canvasH); }

function renderPrediction(result) {
  clearPrediction();
  if (!result) return;
  const drawn = (result.drawn_world || []).map(worldToPixel);
  const pred  = (result.predicted_world || []).map(worldToPixel);
  drawPolyline(predCtx, drawn, { stroke: COLORS.drawn, width: 3.5 });
  drawPolyline(predCtx, pred, { stroke: COLORS.prediction, width: 3.5 });

  const dotOutline = (pt, fill) => {
    predCtx.save();
    predCtx.beginPath();
    predCtx.fillStyle = fill;
    predCtx.arc(pt[0], pt[1], 5.5, 0, Math.PI * 2);
    predCtx.fill();
    predCtx.lineWidth = 2;
    predCtx.strokeStyle = "rgba(255,255,255,0.9)";
    predCtx.stroke();
    predCtx.restore();
  };
  // In point-mode drawn_* is empty; show the start marker using the
  // authoritative heading the backend returned.
  if (drawn.length) {
    dotOutline(drawn[0], COLORS.drawn);
  } else if (result.start_world && result.start_heading != null) {
    const [sx, sy] = worldToPixel(result.start_world);
    drawStartMarker(predCtx, sx, sy, -result.start_heading); // world-y is up, pixel-y is down
  }
  if (pred.length) dotOutline(pred[pred.length - 1], COLORS.predictionHead);
}

function renderStats(result) {
  const drawnVx = result.drawn_vx || [];
  const predVx = result.predicted_vx || [];
  const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  const modeLabel = result.mode === "point" ? "From clicked point" : "From drawn line";
  const controllerLabel = {
    pursuit: "Pure-pursuit",
    mpc: "Model-Predictive (MPC)",
  }[result.controller] || result.controller || "—";
  const items = [
    { label: "Track", value: tracksById[result.track_id]?.label || result.track_id || "—" },
    { label: "Model", value: abbreviateModel(result.model_id) },
    { label: "Controller", value: controllerLabel },
    { label: "Mode", value: modeLabel },
    { label: "Predicted steps", value: `${result.num_prediction_steps} · ${(result.num_prediction_steps * 0.04).toFixed(1)} s` },
    { label: "Mean predicted vx", value: `${avg(predVx).toFixed(1)} m/s` },
    { label: "Target speed", value: `${result.v_target.toFixed(1)} m/s` },
    { label: "Stop reason", value: prettyStop(result.stop_reason) },
  ];
  if (drawnVx.length) {
    items.splice(4, 0, { label: "Mean drawn vx", value: `${avg(drawnVx).toFixed(1)} m/s` });
  }
  let html = items
    .map((it) => `<div class="stat"><div class="label">${it.label}</div><div class="value">${it.value}</div></div>`)
    .join("");
  if (result.warnings && result.warnings.length) {
    html += `<div class="stat warning"><div class="label">Warnings</div><div class="value">${result.warnings.join("<br>")}</div></div>`;
  }
  statsEl.innerHTML = html;
}

function abbreviateModel(id) {
  if (!id) return "—";
  const opt = Array.from(modelSelect.options).find((o) => o.value === id);
  return opt ? opt.textContent : id;
}

function prettyStop(code) {
  const map = {
    max_steps: "reached step cap",
    left_track: "left the track",
    unstable_yaw: "unstable yaw rate",
    low_speed: "speed too low",
    nan: "numerical NaN",
  };
  return map[code] || code || "—";
}

// ---------------------------------------------------------------------------
// Start point (click-to-place) logic
// ---------------------------------------------------------------------------

function updateActionPillText() {
  if (!startPoint) return;
  let heading = centerlineTangentAtPx(startPoint.px, startPoint.py);
  if (startPoint.flip) heading += Math.PI;
  const deg = (((heading * 180) / Math.PI) % 360 + 360) % 360;
  const v = vTargetInput.value ? parseFloat(vTargetInput.value) : null;
  const speedTxt = v ? `${v.toFixed(0)} m/s` : `auto speed`;
  actionSub.textContent = `Heading ${deg.toFixed(0)}° · ${speedTxt}${startPoint.flip ? " · flipped" : ""}`;
}

function showActionPill() {
  actionPill.classList.remove("hidden");
  hintPill.classList.add("hidden");
}

function hideActionPill() {
  actionPill.classList.add("hidden");
  hintPill.classList.remove("hidden");
}

function placeStartPoint(px, py) {
  startPoint = { px, py, flip: false };
  samples = [];
  lastResult = null;
  clearPrediction();
  statsEl.innerHTML = "";
  renderDrawLayer();
  showActionPill();
  updateActionPillText();
  setStatus("Start placed. Adjust speed or press Simulate.");
}

function clearStartPoint() {
  startPoint = null;
  hideActionPill();
  renderDrawLayer();
}

function onFlipHeading() {
  if (!startPoint) return;
  startPoint.flip = !startPoint.flip;
  renderDrawLayer();
  updateActionPillText();
}

async function onSimulate() {
  if (!startPoint) return;
  const controllerId = controllerSelect.value || "pursuit";
  setStatus(`Simulating${controllerId === "mpc" ? " (MPC, may take a few seconds)" : ""}…`);
  btnSimulate.disabled = true;

  const body = {
    canvas_w: canvasW,
    canvas_h: canvasH,
    mode: "point",
    start: {
      x: startPoint.px,
      y: startPoint.py,
      flip_heading: !!startPoint.flip,
    },
    controller: controllerId,
    v_target: vTargetInput.value ? parseFloat(vTargetInput.value) : null,
    max_steps: parseInt(maxStepsInput.value, 10) || 300,
    model_id: modelSelect.value || null,
    track_id: trackSelect.value || null,
    zoom: view.zoom,
    pan_x: view.panX,
    pan_y: view.panY,
  };

  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      setStatus(`Error: ${err.detail}`, "error");
      return;
    }
    const result = await r.json();
    lastResult = result;
    // The backend's authoritative start info supersedes the client marker.
    startPoint = null;
    hideActionPill();
    renderDrawLayer();
    renderStats(result);
    await fitView({ suppressSamplesClear: true });
    setStatus(
      `Predicted ${result.num_prediction_steps} steps · ${(result.num_prediction_steps * 0.04).toFixed(1)} s · ${prettyStop(result.stop_reason)}`
    );
  } catch (e) {
    console.error(e);
    setStatus(`Error: ${e.message}`, "error");
  } finally {
    btnSimulate.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Pointer / view handlers
// ---------------------------------------------------------------------------

function pointerPos(ev) {
  const rect = drawCanvas.getBoundingClientRect();
  return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
}

function isPanEvent(ev) { return ev.shiftKey || ev.button === 1; }

function onPointerDown(ev) {
  ev.preventDefault();
  drawCanvas.setPointerCapture(ev.pointerId);
  if (isPanEvent(ev)) {
    panning = true;
    panStart = { x: ev.clientX, y: ev.clientY, panX: view.panX, panY: view.panY };
    stage.classList.add("panning");
    return;
  }
  // Defer the "drawing vs. click" decision until pointerup or until we see
  // enough movement. This way a quick tap places a start point; any drag
  // starts a live drawing.
  pointerDownInfo = {
    x: ev.clientX, y: ev.clientY,
    canvasX: pointerPos(ev).x, canvasY: pointerPos(ev).y,
    t: performance.now(),
  };
  drawing = false;
}

function onPointerMove(ev) {
  if (panning) {
    view.panX = panStart.panX + (ev.clientX - panStart.x);
    view.panY = panStart.panY + (ev.clientY - panStart.y);
    clampPan();
    scheduleTrackRefresh();
    return;
  }
  if (!pointerDownInfo) return;

  const dx = ev.clientX - pointerDownInfo.x;
  const dy = ev.clientY - pointerDownInfo.y;
  const dist = Math.hypot(dx, dy);
  if (!drawing && dist > CLICK_PX_THRESHOLD) {
    // Movement crossed the threshold: commit to a drawing gesture.
    drawing = true;
    samples = [];
    lastResult = null;
    clearPrediction();
    statsEl.innerHTML = "";
    if (startPoint) { startPoint = null; hideActionPill(); }
    drawStartMs = performance.now();
    samples.push({
      x: pointerDownInfo.canvasX, y: pointerDownInfo.canvasY, t: 0,
    });
    setStatus("Recording…");
  }
  if (!drawing) return;

  const p = pointerPos(ev);
  const t = (performance.now() - drawStartMs) / 1000;
  samples.push({ x: p.x, y: p.y, t });
  renderDrawLayer();
}

async function onPointerUp(ev) {
  if (panning) {
    panning = false;
    stage.classList.remove("panning");
    try { drawCanvas.releasePointerCapture(ev.pointerId); } catch (_) {}
    refreshTrack();
    return;
  }
  if (!pointerDownInfo) return;

  const dt = performance.now() - pointerDownInfo.t;
  const dist = Math.hypot(ev.clientX - pointerDownInfo.x, ev.clientY - pointerDownInfo.y);
  const wasClick = !drawing && dist <= CLICK_PX_THRESHOLD && dt <= CLICK_MS_THRESHOLD;

  try { drawCanvas.releasePointerCapture(ev.pointerId); } catch (_) {}

  if (wasClick) {
    // Single click anywhere on the canvas → place a start point.
    const { canvasX, canvasY } = pointerDownInfo;
    pointerDownInfo = null;
    drawing = false;
    placeStartPoint(canvasX, canvasY);
    return;
  }

  pointerDownInfo = null;

  if (!drawing) return;
  drawing = false;

  if (samples.length < 5) {
    setStatus("Line too short. Draw a longer path.", "warn");
    samples = [];
    renderDrawLayer();
    return;
  }

  await runDrawnPredict();
}

async function runDrawnPredict() {
  const controllerId = controllerSelect.value || "pursuit";
  setStatus(`Predicting${controllerId === "mpc" ? " (MPC, may take a few seconds)" : ""}… (${samples.length} samples)`);

  const body = {
    canvas_w: canvasW,
    canvas_h: canvasH,
    mode: "drawn",
    samples,
    controller: controllerId,
    v_target: vTargetInput.value ? parseFloat(vTargetInput.value) : null,
    max_steps: parseInt(maxStepsInput.value, 10) || 300,
    model_id: modelSelect.value || null,
    track_id: trackSelect.value || null,
    zoom: view.zoom,
    pan_x: view.panX,
    pan_y: view.panY,
  };

  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      setStatus(`Error: ${err.detail}`, "error");
      return;
    }
    const result = await r.json();
    lastResult = result;
    renderStats(result);
    // Auto-fit so the whole drawn + predicted trajectory is visible.
    await fitView({ suppressSamplesClear: true });
    setStatus(
      `Predicted ${result.num_prediction_steps} steps · ${(result.num_prediction_steps * 0.04).toFixed(1)} s · ${prettyStop(result.stop_reason)}`
    );
  } catch (e) {
    console.error(e);
    setStatus(`Error: ${e.message}`, "error");
  }
}

function onClear() {
  samples = [];
  lastResult = null;
  startPoint = null;
  hideActionPill();
  drawCtx.clearRect(0, 0, canvasW, canvasH);
  clearPrediction();
  statsEl.innerHTML = "";
  setStatus("Ready.");
}

function clampPan() {
  const slackX = Math.max(0, (canvasW * (view.zoom - 1)) / 2 + 80);
  const slackY = Math.max(0, (canvasH * (view.zoom - 1)) / 2 + 80);
  view.panX = Math.max(-slackX, Math.min(slackX, view.panX));
  view.panY = Math.max(-slackY, Math.min(slackY, view.panY));
}

function onViewChanged({ clearSamples = true } = {}) {
  clampPan();
  updateZoomUI();
  // Stale pixel-space data: wipe an in-progress drawing and any start marker
  // (it was placed in pixel coords, which no longer line up with the world).
  if (clearSamples && samples.length) samples = [];
  if (startPoint) { startPoint = null; hideActionPill(); }
  drawCtx.clearRect(0, 0, canvasW, canvasH);
  scheduleTrackRefresh();
}

function setZoomAnchored(newZoom, anchorX, anchorY) {
  const oldZoom = view.zoom;
  newZoom = clampZoom(newZoom);
  if (newZoom === oldZoom) return false;
  const centerOffX = anchorX - canvasW / 2;
  const centerOffY = anchorY - canvasH / 2;
  view.panX = centerOffX + (view.panX - centerOffX) * (newZoom / oldZoom);
  view.panY = centerOffY + (view.panY - centerOffY) * (newZoom / oldZoom);
  view.zoom = newZoom;
  return true;
}

function onZoomIn()  { if (setZoomAnchored(view.zoom * ZOOM_STEP_FACTOR, canvasW / 2, canvasH / 2)) onViewChanged(); }
function onZoomOut() { if (setZoomAnchored(view.zoom / ZOOM_STEP_FACTOR, canvasW / 2, canvasH / 2)) onViewChanged(); }

async function fitView({ suppressSamplesClear = false } = {}) {
  view.zoom = 1.0;
  view.panX = 0;
  view.panY = 0;
  updateZoomUI();
  if (!suppressSamplesClear && samples.length) { samples = []; }
  if (startPoint) { startPoint = null; hideActionPill(); }
  drawCtx.clearRect(0, 0, canvasW, canvasH);
  clearTimeout(trackRefreshTimer);
  try {
    trackData = await fetchTrack();
    renderTrack();
    if (lastResult) renderPrediction(lastResult);
  } catch (e) {
    setStatus(`Track reload failed: ${e.message}`, "error");
  }
}

function onFit() { fitView(); }

function onTrackSelectChange() {
  applyTrackDefaults(trackSelect.value);
  onViewChanged();
}

function onWheel(ev) {
  ev.preventDefault();
  const rect = drawCanvas.getBoundingClientRect();
  const cx = ev.clientX - rect.left;
  const cy = ev.clientY - rect.top;
  const factor = Math.exp(-ev.deltaY * 0.0025);
  if (!setZoomAnchored(view.zoom * factor, cx, cy)) return;
  onViewChanged();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  resizeStageToWindow();
  setStatus("Loading…");

  try {
    const [tracksInfo, modelInfo, controllerInfo] = await Promise.all([
      fetchTracks(), fetchModels(), fetchControllers(),
    ]);
    populateTrackSelect(tracksInfo);
    populateModelSelect(modelInfo);
    populateControllerSelect(controllerInfo);
  } catch (e) {
    setStatus(`Failed to load options: ${e.message}`, "error");
    return;
  }

  try {
    trackData = await fetchTrack();
    renderTrack();
  } catch (e) {
    setStatus(`Failed to load track: ${e.message}`, "error");
    return;
  }

  setStatus("Ready.");

  drawCanvas.addEventListener("pointerdown", onPointerDown);
  drawCanvas.addEventListener("pointermove", onPointerMove);
  drawCanvas.addEventListener("pointerup", onPointerUp);
  drawCanvas.addEventListener("pointercancel", onPointerUp);
  drawCanvas.addEventListener("wheel", onWheel, { passive: false });
  drawCanvas.addEventListener("contextmenu", (ev) => ev.preventDefault());

  btnClear.addEventListener("click", onClear);
  btnFit.addEventListener("click", onFit);
  btnZoomIn.addEventListener("click", onZoomIn);
  btnZoomOut.addEventListener("click", onZoomOut);
  btnFlip.addEventListener("click", onFlipHeading);
  btnSimulate.addEventListener("click", onSimulate);
  btnCancelStart.addEventListener("click", () => { clearStartPoint(); setStatus("Ready."); });
  trackSelect.addEventListener("change", onTrackSelectChange);
  vTargetInput.addEventListener("input", updateActionPillText);
  updateZoomUI();

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(async () => {
      resizeStageToWindow();
      try {
        trackData = await fetchTrack();
        renderTrack();
        if (lastResult) renderPrediction(lastResult);
      } catch (e) {
        setStatus(`Track reload failed: ${e.message}`, "error");
      }
    }, 200);
  });
}

init();
