# Deep Dynamics Racing Line Predictor

A small web app that lets you either **draw** a partial racing line or
**click a single point** on a racetrack. The backend reconstructs (or
synthesizes) vehicle states from the user input, then rolls a trained
vehicle-dynamics model forward using one of two selectable controllers —
kinematic pure-pursuit, or a **Model-Predictive (MPC) planner** that uses
the same learned dynamics model as its internal simulator. Both the
reconstructed and predicted trajectories are drawn back on the canvas.

The UI is an Apple Maps / Freeform–inspired **light** layout: a borderless
translucent top nav (Track / Model / Controller selectors), a single large
canvas card, and floating translucent pills on the canvas itself — a
hint pill, a bottom-right zoom + fit pill, and a contextual "Start placed"
action pill with **Flip / Simulate / ✕** buttons that only appears when
you've clicked a start point. When a prediction finishes the view
auto-fits to full track so the drawn + predicted trajectories are always
visible end-to-end.

## Install

**Important:** Launch from the same Python environment that has `torch`,
`scikit-learn`, `numpy`, and `pyyaml` installed (the `deep_dynamics`
conda env used everywhere else in this repo). Running `uvicorn` from a
different Python will fail with `ModuleNotFoundError: No module named
'sklearn'` or similar.

```bash
conda activate deep_dynamics
pip install fastapi uvicorn
```

## Run

From the repo root, inside the activated `deep_dynamics` env:

```bash
conda activate deep_dynamics
python -m uvicorn deep_dynamics.interactive.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and either **drag** a racing line on the track
or **single-click** a starting point. The first request after startup
takes a few seconds because the model and track are loaded lazily on the
FastAPI `startup` event.

## Interaction modes

Two ways to get a prediction, handled by the same `/predict` endpoint:

- **Drawn mode** — drag across the track. The backend resamples the path
  to `Ts = 0.04 s`, inverts a kinematic bicycle + quasi-static longitudinal
  model to recover a 15-sample `(vx, vy, yaw_rate, throttle, steering)`
  history, then rolls the dynamics model forward.
- **Point mode** — single-click anywhere on the stage. The backend takes
  your click, looks up the nearest centerline point, uses the centerline
  **tangent** at that point as the starting heading, and synthesizes a
  15-sample steady-state cruise history at your target speed. Press
  **Simulate** to run the rollout. Press **Flip** first to reverse the
  direction 180° (pure-pursuit's centerline is walked backward so the car
  follows the track the other way).

Click vs. drag is disambiguated by a 6-pixel / 320 ms threshold in the
frontend, so light taps reliably place a start point while any deliberate
drag starts a drawing.

## Controllers

`GET /controllers` lists the two rollout controllers; the UI shows them
in the top-nav **Controller** dropdown.

| id | what it does | relative cost |
| --- | --- | --- |
| `pursuit` (default) | Classical kinematic pure-pursuit: nearest centerline point, look ahead `L0 + k·v` metres, compute steering that arcs the car to the lookahead point. Throttle is P on speed error, inverted through the longitudinal model. Fast and model-agnostic. | 1× |
| `mpc` | **Random-shooting MPC** over the trained dynamics model. At every control step we sample K = 8 candidate steering perturbations around pure-pursuit's suggestion, batch-simulate one step of each candidate through the dynamics network, extrapolate each candidate's pose one more kinematic step (so the trajectories actually diverge), and pick the candidate whose predicted pose best tracks the centerline (with a small speed-tracking bonus and a large off-track penalty). The chosen forward pass is re-used so we don't pay for an extra forward pass per step. | ~5× |

MPC literally uses the same trained dynamics network as its planner — it
is the "use the model we have to plan with" flavor of model-based RL
applied at inference time. No offline training required; the gains show
up immediately on the IAC checkpoint.

## Pieces

- `app.py` — FastAPI server exposing:
  - `GET /tracks` — list of selectable tracks (id, label, default zoom).
  - `GET /models` — list of selectable trained checkpoints.
  - `GET /controllers` — list of selectable rollout controllers.
  - `GET /track` — bounds + centerline in pixel coords sized to the browser
    canvas, honoring `track_id`, `zoom`, `pan_x`, `pan_y`.
  - `POST /predict` — unified endpoint: drawn samples **or** a single start
    point, plus `controller` (`pursuit` | `mpc`), `track_id`, `model_id`,
    view params, and prediction knobs.
- `pipeline.py` — model/scaler loading, pixel↔world transform, track
  registry (Putnam CSV loader, synthetic oval generator, F1TENTH
  centerline-with-width loader, auto-discovery of any CSV dropped into
  `tracks/`), drawn-line state estimator, pure-pursuit controller,
  random-shooting MPC controller, synthetic history builder for
  start-from-point mode, and the step-by-step rollout loop (mirrors the
  one in [../visualize/plot_predictions_iac.py](../visualize/plot_predictions_iac.py)).
- `tracks/` — F1TENTH-format centerline CSVs
  (`x_m, y_m, w_tr_right_m, w_tr_left_m`, scaled 1:10 from real life).
  Any `*_centerline.csv` in this folder is auto-registered as a track
  and 10×-upscaled to real-world dimensions on load.
- `static/` — HTML/JS/CSS frontend. Three stacked canvases (track →
  prediction → draw) under a single stage card, plus the floating
  translucent pills that drive the Apple Maps–style interaction model.

## Tracks

| id | description | default zoom |
| --- | --- | --- |
| `simple_oval` (default) | Synthetic elliptical oval (~140 × 70 m, width 14 m). Small, symmetric, easy to draw a racing line on. | 1× |
| `putnam` | Putnam Park Road Course loaded from the repo's boundary CSVs. The dataset used to train the IAC model. | 2× |
| `ims` | Indianapolis Motor Speedway — the superspeedway oval the IAC was raced on. Simple rectangular layout, very drawable. | 1.5× |
| `monza` | Autodromo Nazionale Monza — fast, flowing F1 circuit. | 3× |
| `silverstone` | Silverstone Circuit — technical medium-speed F1 layout. | 3× |
| `austin` | Circuit of the Americas (F1) — mixed rhythm sections. | 3× |

The real circuits (IMS, Monza, Silverstone, Austin) come from the
[f1tenth/f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)
dataset as centerline-plus-width CSVs. They ship at 1:10 scale and are
10×-upscaled on load to recover real-world dimensions (≈22 m track width,
real circuit length). Track edges are reconstructed by offsetting each
centerline point along its local normal.

### Adding more tracks

1. **Drop-in F1TENTH CSV** — grab any `*_centerline.csv` from
   [f1tenth/f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks),
   drop it into `tracks/`, restart the server. It'll be auto-registered
   with a sensible default zoom. Pin a prettier label / zoom by adding an
   entry to `_F1TENTH_OVERRIDES` in `pipeline.py`.
2. **Custom loader** — append an entry to `TRACK_REGISTRY` in
   `pipeline.py` with a loader that returns a `Track` dataclass
   (`inner`, `outer`, `centerline` as `(N, 2)` numpy arrays in meters).

## Models

Selectable checkpoints are defined in `MODEL_REGISTRY` inside `app.py`
and exposed by `GET /models`:

| id | config | weights |
| --- | --- | --- |
| `iac_dumbbounds_291` (default) | `deep_dynamics_iac.yaml` | `output/deep_dynamics_iac/dumbboundsiac/epoch_291.pth` |
| `iac_run2_102` | `deep_dynamics_iac.yaml` | `output/deep_dynamics_iac/iac_run2/epoch_102.pth` |

Each model is lazy-loaded on first use and cached for the process
lifetime. The `deep_dynamics_pinn` checkpoints in `output/` are
intentionally **not** registered (RC-scale vehicle specs + different
`HORIZON`, incompatible with the real-scale IAC + Putnam combo).

## UI controls

### Top nav

- **Track**: which track geometry to draw on. Switching tracks auto-applies
  that track's `default_zoom` and clears any in-flight input.
- **Model**: which trained dynamics checkpoint to use.
- **Controller**: `Pure-pursuit` or `Model-Predictive (MPC)`. See above.
- **Status pill**: shows what the backend is doing ("Recording…",
  "Simulating (MPC, may take a few seconds)…", error message, etc.).
- **Clear**: wipes drawing, start point, and any in-view prediction.

### Floating canvas pills (Apple Maps / Freeform style)

- **Hint pill** (top-left): reminds you of the two ways to input —
  "Drag to draw · Click to place a start point". Hidden when a start
  point is placed.
- **Zoom pill** (bottom-right): `−` / `+` to zoom by 25% steps, the
  percentage read-out, and a **Fit** button that reframes the whole
  circuit.
- **Action pill** (bottom-center, appears when a start is placed):
  - **Flip** — rotate the start heading 180° (and walk the centerline
    backward so pursuit follows the track the other way).
  - **Simulate** — run the prediction from this point.
  - **✕** — cancel the start point.

### Params row

- **Target speed (m/s)**: the speed the controllers try to hold. In
  point-mode it's also the initial speed of the synthesized history. Leave
  blank in drawn-mode to auto-pick from the mean drawn speed.
- **Prediction steps**: number of dynamics-model forward passes to run.
  Each step is `Ts = 0.04 s`, so 300 steps ≈ 12 s of predicted motion.
  Rollout stops early if the path leaves the track or the yaw rate
  explodes.

### Mouse gestures on the canvas

- **Left-button drag** (> 6 px): draw a racing line.
- **Left-button single click** (≤ 6 px, ≤ 320 ms): place a start point.
- **Scroll wheel** (or trackpad pinch): smooth zoom, anchored on the
  cursor so the point under the cursor stays put.
- **Shift + left-drag** or **middle-click drag**: pan the view.

Panning / zooming mid-gesture clears the in-progress drawing and any
unplaced start marker (their pixel coords become stale under the new
transform). A **completed prediction** is stored in world coords and
re-projected through the fresh transform, so you can freely zoom/pan
around a prediction after the fact.

### Auto-fit after prediction

When `/predict` returns, the view is snapped back to zoom = 100%, pan = 0
and the prediction is re-rendered from its world coordinates through the
freshly-fetched transform. Every bundled track fits the canvas at 100%,
so the drawn + predicted trajectory is always visible end-to-end without
manual panning.

View state (zoom + pan) is sent to the backend with both `/track` and
`/predict`, so the pixel↔world transform is identical on both sides — a
drawn line or clicked start in a zoomed view produces the same predicted
trajectory as the same input at zoom = 1.

## Caveats

- The model was trained on IAC data with `vx > 5 m/s`, so drawing very
  slowly clamps the estimated speed; a warning is surfaced in the stats
  panel when this happens.
- The drawn-line → throttle/steering reconstruction uses a kinematic
  bicycle inverse and a quasi-static longitudinal inverse with nominal
  Pacejka / drivetrain coefficients from the midpoint of the cfg's
  parameter ranges. It is approximate and is mostly meant to seed the
  model with a plausible 15-step history; don't treat the inferred
  throttle / steering values as ground truth.
- Training data is single-direction around Putnam, so **Flip** (= running
  a track the "wrong" way) can destabilize the dynamics model on tighter
  tracks even though the pursuit controller walks the centerline
  backward correctly. It works robustly on the symmetric `simple_oval`;
  on F1 tracks it's best-effort.
- MPC is ~5× slower than pure-pursuit because each control step does a
  batched forward pass over K = 8 candidate actions. A 300-step rollout
  runs in ~4 s on CPU. The batch constant + score weights live in
  `MPCConfig` at the top of `pipeline.py` if you want to tune.
- Rollout does a single-batch forward pass per step on CPU, ~25–30 Hz
  for pure-pursuit. Predicting many hundreds of steps will noticeably
  delay the response.
