# RL Racing Agent — Implementation Plan

## Project Overview

**Goal:** Train an SAC (Soft Actor-Critic) agent to drive an IndyCar as fast as possible around a loop track (LVMS or Putnam), using a pre-trained Deep Dynamics model as the physics engine.

**Architecture:**
- **RL Agent (SAC)** outputs throttle and steering increments every Ts seconds
- **Deep Dynamics checkpoint** receives those actions + state history, predicts next body-frame velocities (vx, vy, yaw_rate)
- **Pose integrator** converts body-frame velocities to global (x, y, heading) for track-relative observations and termination checks
- **Track module** provides centerline, boundaries, progress measurement, and curvature lookahead

**Key constraint:** The Deep Dynamics model was trained on specific state/action semantics (incremental commands, feedback channels, StandardScaler normalization). The RL environment must replicate these exactly or the predictions are meaningless.

---

## File Structure

```
rl_racing/
├── env/
│   ├── __init__.py
│   ├── racing_env.py          # Gymnasium env wrapping Deep Dynamics
│   ├── dynamics_wrapper.py    # Loads checkpoint + scaler, runs inference
│   ├── track.py               # Track geometry, boundaries, progress
│   └── pose_integrator.py     # Body-frame vel → global pose updates
├── agent/
│   ├── __init__.py
│   ├── sac.py                 # SAC with automatic entropy tuning
│   ├── networks.py            # Actor (Gaussian policy), twin Q-networks
│   └── replay_buffer.py       # Standard replay buffer
├── configs/
│   └── default.yaml           # All hyperparameters in one place
├── scripts/
│   ├── train.py               # Training loop entry point
│   ├── evaluate.py            # Load policy, run episodes, record metrics
│   └── visualize.py           # Render trajectories on track
├── utils/
│   ├── __init__.py
│   └── logger.py              # TensorBoard + CSV logging
└── README.md
```

---

## Phase 0: Setup and Dependency Verification

**Files to create/edit:** `pyproject.toml` or `requirements.txt`

### Task 0.1 — Python environment

Install into the same environment that has the Deep Dynamics code:

```
gymnasium>=0.29
torch>=2.0
numpy
pyyaml
tensorboard
```

Do NOT install stable-baselines3 — we are writing SAC from scratch for full control over the dynamics interface.

### Task 0.2 — Verify checkpoint loads

Write a throwaway test script that:
1. Loads the trained `.pth` checkpoint and `scaler.pkl`
2. Loads the model YAML config
3. Constructs a dummy input tensor of shape `(1, HORIZON, 7)` (7 features for IAC)
4. Runs `model.forward(x, x_norm)` and confirms output shape is `(1, 3)` — the predicted (vx, vy, yaw_rate)
5. Prints the predicted values to sanity-check they are physically plausible

This confirms your checkpoint, scaler, and model class are all compatible before building anything else.

### Task 0.3 — Verify track data loads

Write a throwaway test that:
1. Loads the LVMS (or Putnam) track CSV from `deep_dynamics/visualize/tracks/`
2. Plots the centerline, inner boundary, and outer boundary with matplotlib
3. Confirms the track forms a closed loop (first point ≈ last point)

---

## Phase 1: Track Module

**File:** `rl_racing/env/track.py`

### Task 1.1 — Track class

```python
class Track:
    def __init__(self, track_csv_path: str):
        """
        Load track CSV. Expected columns: x_center, y_center,
        x_inner, y_inner, x_outer, y_outer (or similar — inspect
        the actual CSV headers in deep_dynamics/visualize/tracks/).

        Store as numpy arrays:
          self.centerline: (N, 2)    # ordered waypoints
          self.inner_bound: (N, 2)
          self.outer_bound: (N, 2)
          self.widths: (N,)          # half-track-width at each waypoint
          self.cumulative_s: (N,)    # arc-length distance along centerline
          self.total_length: float   # full lap distance
          self.headings: (N,)        # tangent angle at each waypoint
          self.curvatures: (N,)      # signed curvature at each waypoint
        """
```

**Implementation notes:**
- Compute `cumulative_s` as cumulative Euclidean distances between consecutive centerline points
- Compute `headings` via `np.arctan2(dy, dx)` on consecutive centerline segments
- Compute `curvatures` via finite differences of heading divided by ds
- Handle wraparound: the track is periodic, so interpolation near index 0/N must wrap

### Task 1.2 — Frenet projection

```python
def cartesian_to_frenet(self, x: float, y: float) -> tuple[float, float, float]:
    """
    Project (x, y) onto centerline.
    Returns:
      s:     arc-length progress along centerline (mod total_length)
      d:     signed lateral deviation (positive = left of centerline)
      phi_t: tangent heading at projection point
    """
```

Find the nearest centerline segment, compute perpendicular projection for sub-segment accuracy. This is called every step — use vectorized nearest-segment search (precompute a KD-tree on centerline points, or just brute-force argmin on ~500 points — it's fast enough).

### Task 1.3 — Boundary check and curvature lookahead

```python
def is_within_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
    """Check if point is inside track boundaries (with optional safety margin)."""

def get_curvature_ahead(self, s: float, lookahead_distances: list[float]) -> np.ndarray:
    """
    Return curvature values at s + each lookahead distance.
    Used in observation to let the agent 'see' upcoming turns.
    """
```

### Task 1.4 — Lap counting

```python
def update_progress(self, s_prev: float, s_new: float) -> tuple[float, bool]:
    """
    Returns (delta_s, crossed_finish_line).
    Handle wraparound: if s_new < s_prev and the jump is large,
    that's a lap completion, not going backwards.
    """
```

---

## Phase 2: Dynamics Wrapper

**File:** `rl_racing/env/dynamics_wrapper.py`

This is the most critical file. It must replicate exactly how the Deep Dynamics model expects input.

### Task 2.1 — Load model and scaler

```python
class DynamicsWrapper:
    def __init__(self, checkpoint_path: str, config_path: str, scaler_path: str, device: str = "cpu"):
        """
        1. Load YAML config → extract MODEL.NAME, MODEL.HORIZON, STATE, ACTIONS, Ts
        2. Instantiate model via string_to_model[MODEL.NAME](config)
        3. Load checkpoint weights
        4. Load sklearn StandardScaler from scaler.pkl
        5. Set model to eval mode, disable gradients
        """
        self.horizon = ...   # from config, e.g. 20
        self.Ts = ...        # from config, e.g. 0.04 for IAC
        self.device = device
```

### Task 2.2 — History buffer management

The model expects a sliding window of H timesteps, each with 7 features (for IAC):
`[vx, vy, yaw_rate, throttle_fb, steering_fb, throttle_cmd, steering_cmd]`

```python
def reset(self, initial_vx: float = 10.0, initial_vy: float = 0.0,
          initial_yaw_rate: float = 0.0):
    """
    Initialize history buffer: shape (HORIZON, 7), fill with
    initial steady-state values. Set throttle_fb/steering_fb to
    some reasonable initial (e.g. throttle that maintains initial_vx,
    steering = 0). Set command columns to 0.
    """
    self.history = np.zeros((self.horizon, 7))
    # Fill columns 0-2 with initial velocities
    # Fill columns 3-4 with initial throttle_fb, steering_fb
    # Columns 5-6 stay 0 (no commands yet)
```

### Task 2.3 — Step function

```python
def step(self, throttle_cmd: float, steering_cmd: float) -> tuple[np.ndarray, dict]:
    """
    1. Build current row:
       - vx, vy, yaw_rate from last prediction (or initial state)
       - throttle_fb = previous throttle_fb + throttle_cmd
       - steering_fb = previous steering_fb + steering_cmd
       - throttle_cmd, steering_cmd as given

    2. Shift history: drop oldest row, append new row

    3. Normalize: Apply scaler.transform to history.reshape(1*H, 7),
       reshape back to (1, H, 7)

    4. Run model.forward(x_raw, x_normalized)
       → prediction: (vx_next, vy_next, yaw_rate_next)

    5. Return (prediction, sysid_params_dict)
    """
```

**Critical details:**
- `throttle_fb` and `steering_fb` are INTEGRATED (accumulated) values — they grow over time as commands are added. This matches `csv_parser.py` where `previous_throttle += throttle_cmd`.
- The scaler was fit on training data — the distribution of values must be similar or the model will see out-of-distribution inputs. Clamp actions to reasonable ranges.
- `x_raw` (un-normalized) goes to the physics head; `x_normalized` goes through the neural network layers. Both must be passed to `model.forward()`.

---

## Phase 3: Pose Integrator

**File:** `rl_racing/env/pose_integrator.py`

### Task 3.1 — Body-frame to global-frame integration

```python
class PoseIntegrator:
    def __init__(self, Ts: float):
        self.Ts = Ts
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # global yaw angle (radians)

    def reset(self, x: float, y: float, heading: float):
        self.x = x
        self.y = y
        self.heading = heading

    def step(self, vx: float, vy: float, yaw_rate: float) -> tuple[float, float, float]:
        """
        Integrate body-frame velocities to global pose.

        Global velocities:
          x_dot = vx * cos(heading) - vy * sin(heading)
          y_dot = vx * sin(heading) + vy * cos(heading)

        Euler update:
          heading += yaw_rate * Ts
          x += x_dot * Ts
          y += y_dot * Ts

        Returns (x, y, heading)
        """
```

This is straightforward but must use the same Ts as the dynamics model.

---

## Phase 4: Gymnasium Environment

**File:** `rl_racing/env/racing_env.py`

### Task 4.1 — Environment class

```python
import gymnasium as gym
from gymnasium import spaces

class RacingEnv(gym.Env):
    def __init__(self, config: dict):
        super().__init__()

        # Components
        self.track = Track(config["track_csv"])
        self.dynamics = DynamicsWrapper(
            config["checkpoint_path"],
            config["model_config_path"],
            config["scaler_path"],
            device=config.get("device", "cpu")
        )
        self.pose = PoseIntegrator(self.dynamics.Ts)

        # Episode config
        self.max_steps = config.get("max_steps", 5000)  # ~200s at 25Hz
        self.initial_speed = config.get("initial_speed", 15.0)  # m/s

        # Track-relative observation config
        self.n_curvature_points = config.get("n_curvature_points", 10)
        self.curvature_lookaheads = [5.0 * i for i in range(1, self.n_curvature_points + 1)]
        # lookahead at 5m, 10m, ..., 50m ahead

        # --- Spaces ---
        # Action: [throttle_cmd, steering_cmd] as increments
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.05]),   # TUNE THESE to match training data range
            high=np.array([0.5, 0.05]),
            dtype=np.float32
        )

        # Observation: see _get_obs() below
        obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
```

### Task 4.2 — Observation design

```python
def _get_obs(self) -> np.ndarray:
    """
    Observation vector (all from the agent's perspective):

    Body-frame dynamics (3):
      - vx (longitudinal speed)
      - vy (lateral speed)
      - yaw_rate

    Track-relative state (3):
      - lateral deviation d (signed, from Frenet projection)
      - heading error (agent heading - track tangent heading)
      - progress speed (ds/dt, how fast progressing along centerline)

    Curvature lookahead (n_curvature_points):
      - curvature at future distances along centerline

    Control state (4):
      - current throttle_fb (integrated throttle)
      - current steering_fb (integrated steering)
      - last throttle_cmd
      - last steering_cmd

    Total: 10 + n_curvature_points
    """
```

**Design rationale:**
- No absolute (x, y) in observations — the agent should learn track-relative behavior that generalizes to any position on the loop
- Curvature lookahead gives the agent information about upcoming turns so it can learn to brake/steer preemptively
- Include control state so the agent knows its accumulated throttle/steering position (since its actions are increments)

### Task 4.3 — Reward function

```python
def _compute_reward(self, vx: float, delta_s: float, d: float,
                    heading_error: float, action: np.ndarray) -> float:
    """
    Reward components:

    1. Progress reward (DOMINANT):
       r_progress = delta_s / self.dynamics.Ts
       This is the forward speed projected onto the centerline.
       Scale: ~30 m/s at racing speed → reward ~30 per step.

    2. Lateral deviation penalty:
       r_lateral = -alpha_d * d^2
       Soft penalty that increases near boundaries.

    3. Heading error penalty:
       r_heading = -alpha_h * heading_error^2
       Encourages alignment with track direction.

    4. Smoothness penalty:
       r_smooth = -alpha_a * (action[0]^2 + action[1]^2)
       Discourages jittery commands.

    5. Speed bonus (optional):
       r_speed = alpha_v * vx
       Direct reward for going fast (somewhat redundant with progress).

    Total: r_progress + r_lateral + r_heading + r_smooth + r_speed

    Recommended starting weights:
      alpha_d = 0.1
      alpha_h = 0.5
      alpha_a = 0.01
      alpha_v = 0.0 (start without, add if needed)
    """
```

### Task 4.4 — Step and reset

```python
def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
    throttle_cmd, steering_cmd = action

    # 1. Dynamics step
    pred, sysid = self.dynamics.step(throttle_cmd, steering_cmd)
    vx, vy, yaw_rate = pred[0]

    # 2. Pose integration
    x, y, heading = self.pose.step(vx, vy, yaw_rate)

    # 3. Track projection
    s, d, track_heading = self.track.cartesian_to_frenet(x, y)
    heading_error = _normalize_angle(heading - track_heading)

    # 4. Progress
    delta_s, lap_complete = self.track.update_progress(self.prev_s, s)
    self.prev_s = s
    self.step_count += 1

    # 5. Reward
    reward = self._compute_reward(vx, delta_s, d, heading_error, action)

    # 6. Termination
    terminated = False
    if not self.track.is_within_bounds(x, y):
        terminated = True
        reward -= 100.0  # crash penalty
    if lap_complete:
        self.lap_count += 1
        # Optional: bonus for completing a lap

    truncated = (self.step_count >= self.max_steps)

    # 7. Info
    info = {
        "vx": vx, "vy": vy, "yaw_rate": yaw_rate,
        "x": x, "y": y, "heading": heading,
        "s": s, "d": d, "heading_error": heading_error,
        "lap_count": self.lap_count,
        "delta_s": delta_s,
    }

    return self._get_obs(), reward, terminated, truncated, info


def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Place car at a random-ish point on track centerline
    # pointing along the track tangent
    start_idx = 0  # or random for variety
    x0, y0 = self.track.centerline[start_idx]
    heading0 = self.track.headings[start_idx]

    self.pose.reset(x0, y0, heading0)
    self.dynamics.reset(initial_vx=self.initial_speed)
    self.prev_s = self.track.cumulative_s[start_idx]
    self.step_count = 0
    self.lap_count = 0

    return self._get_obs(), {}
```

### Task 4.5 — Termination conditions

The episode ends (terminated=True) when:
- Car leaves track boundaries → crash, -100 penalty
- `vx < 1.0` for more than 50 consecutive steps → stuck, -50 penalty
- (Optional) Heading error > π/2 for sustained period → going backwards

The episode is truncated (truncated=True) when:
- `step_count >= max_steps`

---

## Phase 5: SAC Agent

**File:** `rl_racing/agent/networks.py`

### Task 5.1 — Actor (Gaussian policy)

```python
class GaussianActor(nn.Module):
    """
    MLP that outputs mean and log_std for a squashed Gaussian policy.

    Architecture:
      obs_dim → 256 → 256 → (mean: action_dim, log_std: action_dim)

    Forward returns:
      action, log_prob (both differentiable)

    Uses tanh squashing:
      raw_action ~ N(mean, std)
      action = tanh(raw_action)
      scaled to action_space bounds

    Log-prob correction for tanh:
      log_pi = log_pi_raw - sum(log(1 - tanh^2(raw) + eps))
    """
```

### Task 5.2 — Twin Q-Networks

```python
class TwinQNetwork(nn.Module):
    """
    Two independent Q-networks (for clipped double-Q).
    Each: (obs_dim + action_dim) → 256 → 256 → 1
    """
```

**File:** `rl_racing/agent/replay_buffer.py`

### Task 5.3 — Replay buffer

```python
class ReplayBuffer:
    """
    Standard circular buffer storing (obs, action, reward, next_obs, done).
    Size: 1_000_000 transitions.
    Sample returns batch of numpy arrays (converted to tensors in SAC).
    """
```

**File:** `rl_racing/agent/sac.py`

### Task 5.4 — SAC algorithm

```python
class SAC:
    def __init__(self, obs_dim, action_dim, action_low, action_high, config):
        """
        Components:
          - GaussianActor (policy)
          - TwinQNetwork (online)
          - TwinQNetwork (target, EMA-updated)
          - log_alpha (learnable entropy coefficient)
          - Optimizers: Adam for actor, Q, and alpha

        Key hyperparameters (in config):
          lr_actor: 3e-4
          lr_critic: 3e-4
          lr_alpha: 3e-4
          gamma: 0.99
          tau: 0.005         # target network EMA rate
          batch_size: 256
          target_entropy: -action_dim  # automatic entropy tuning target
          initial_random_steps: 10_000  # random exploration before policy
        """

    def select_action(self, obs, deterministic=False):
        """For training: sample from policy. For eval: use mean."""

    def update(self, batch):
        """
        1. Compute Q-targets:
           next_action, next_log_pi = actor(next_obs)
           q_target = reward + gamma * (1 - done) *
                      (min(Q_target_1, Q_target_2)(next_obs, next_action)
                       - alpha * next_log_pi)

        2. Update Q-networks:
           loss_q = MSE(Q1(obs, action), q_target) + MSE(Q2(obs, action), q_target)

        3. Update actor:
           new_action, log_pi = actor(obs)
           loss_actor = mean(alpha * log_pi - min(Q1, Q2)(obs, new_action))

        4. Update alpha:
           loss_alpha = -mean(log_alpha * (log_pi + target_entropy).detach())

        5. Soft-update target networks:
           target_params = tau * online_params + (1 - tau) * target_params
        """

    def save(self, path): ...
    def load(self, path): ...
```

---

## Phase 6: Training Loop

**File:** `rl_racing/scripts/train.py`

### Task 6.1 — Training script

```python
"""
Training loop pseudocode:

1. Load config from configs/default.yaml
2. Create RacingEnv
3. Create SAC agent
4. Create ReplayBuffer
5. Create TensorBoard logger

6. For episode = 1 to max_episodes:
     obs, _ = env.reset()
     episode_reward = 0

     For step = 1 to max_steps_per_episode:
       if total_steps < initial_random_steps:
         action = env.action_space.sample()
       else:
         action = agent.select_action(obs)

       next_obs, reward, terminated, truncated, info = env.step(action)
       buffer.add(obs, action, reward, next_obs, terminated or truncated)

       if total_steps >= initial_random_steps:
         batch = buffer.sample(batch_size)
         metrics = agent.update(batch)
         # Log metrics every N steps

       obs = next_obs
       episode_reward += reward
       total_steps += 1

       if terminated or truncated:
         break

     # Log episode stats: reward, lap_count, max_vx, etc.
     # Save checkpoint every K episodes
     # Run eval episode (deterministic policy) every K episodes
"""
```

---

## Phase 7: Config File

**File:** `rl_racing/configs/default.yaml`

### Task 7.1 — Central configuration

```yaml
# --- Dynamics model ---
dynamics:
  checkpoint_path: "path/to/epoch_best.pth"
  model_config_path: "path/to/model_config.yaml"
  scaler_path: "path/to/scaler.pkl"
  device: "cpu"  # or "cuda"

# --- Track ---
track:
  csv_path: "deep_dynamics/visualize/tracks/LVMS.csv"

# --- Environment ---
env:
  max_steps: 5000
  initial_speed: 15.0  # m/s
  n_curvature_points: 10
  curvature_lookahead_spacing: 5.0  # meters between lookahead points

# --- Action bounds (MUST match training data distribution) ---
action:
  throttle_cmd_range: [-0.5, 0.5]
  steering_cmd_range: [-0.05, 0.05]

# --- Reward weights ---
reward:
  alpha_d: 0.1       # lateral deviation penalty
  alpha_h: 0.5       # heading error penalty
  alpha_a: 0.01      # action smoothness penalty
  alpha_v: 0.0       # speed bonus (start at 0)
  crash_penalty: 100.0
  stuck_penalty: 50.0

# --- SAC ---
sac:
  lr_actor: 3e-4
  lr_critic: 3e-4
  lr_alpha: 3e-4
  gamma: 0.99
  tau: 0.005
  batch_size: 256
  buffer_size: 1_000_000
  initial_random_steps: 10_000
  target_entropy: -2  # = -action_dim
  hidden_dim: 256

# --- Training ---
training:
  max_episodes: 10_000
  eval_every: 50          # episodes
  save_every: 100         # episodes
  log_every: 10           # steps (for TensorBoard)
  checkpoint_dir: "checkpoints/"
  log_dir: "logs/"
```

---

## Phase 8: Evaluation and Visualization

**File:** `rl_racing/scripts/evaluate.py`

### Task 8.1 — Evaluation script

Run N episodes with deterministic policy. Record per-episode: lap time, number of laps, average speed, max lateral deviation, number of crashes. Print aggregate stats.

**File:** `rl_racing/scripts/visualize.py`

### Task 8.2 — Trajectory visualization

```python
"""
1. Run one episode, collect (x, y) at every step + info dict
2. Plot track boundaries (inner, outer) and centerline
3. Overlay the agent's trajectory, color-coded by speed (vx)
4. Optionally animate as a GIF/video
5. Plot time series: vx, vy, yaw_rate, throttle_fb, steering_fb,
   reward components, lateral deviation
"""
```

---

## Implementation Order (Cursor task sequence)

Work through these in order. Each task is self-contained and testable.

| Step | Task | Test |
|------|------|------|
| 1 | Task 0.2 — checkpoint loads | Dummy forward pass produces (1, 3) output |
| 2 | Task 0.3 — track data loads | Plot track, visually confirm it's closed |
| 3 | Task 1.1–1.4 — Track class | Unit tests: Frenet projection, boundary check, lap detection |
| 4 | Task 3.1 — PoseIntegrator | Drive straight at vx=30: after 1s, x should advance ~30m |
| 5 | Task 2.1–2.3 — DynamicsWrapper | Feed constant throttle for 100 steps: vx should increase plausibly |
| 6 | Task 4.1–4.5 — RacingEnv | `env.reset()` + 100 random actions: no crashes, obs shape correct |
| 7 | Task 5.3 — ReplayBuffer | Add 1000 transitions, sample batch, check shapes |
| 8 | Task 5.1–5.2 — Actor + Q-nets | Forward pass with random inputs, check output shapes |
| 9 | Task 5.4 — SAC | One update step with random batch: losses are finite |
| 10 | Task 7.1 — Config file | Load config, construct all components |
| 11 | Task 6.1 — Training loop | Train for 100 episodes: reward trends upward (even slightly) |
| 12 | Task 8.1–8.2 — Eval + Viz | Visualize learned trajectory on track |

---

## Critical Gotchas (read before coding)

1. **Action semantics are INCREMENTS, not absolutes.** The RL agent outputs delta-throttle and delta-steering. These accumulate into `throttle_fb` and `steering_fb` inside the dynamics wrapper. If the agent outputs +0.1 throttle for 20 steps, `throttle_fb` reaches 2.0. This can drift to unrealistic values — you may need to clip `throttle_fb` and `steering_fb` to ranges observed in training data.

2. **Scaler normalization is mandatory.** The neural network was trained on StandardScaler-transformed inputs. Feeding raw values will produce garbage predictions. Load `scaler.pkl` and apply `scaler.transform()` to the flattened history before inference.

3. **model.forward() takes TWO inputs:** raw `x` (for the physics head) and normalized `x_norm` (for the neural network). Both must have shape `(batch, HORIZON, features)`.

4. **Ts must match.** IAC checkpoints use Ts=0.04 (25 Hz). The Euler integration in both the dynamics model and the pose integrator must use this same timestep.

5. **IAC uses 7 features, not 8.** The csv_parser generates 8 columns but DeepDynamicsDataset drops column 7 (the vx lookahead). Your history buffer should have 7 columns: `[vx, vy, yaw_rate, throttle_fb, steering_fb, throttle_cmd, steering_cmd]`.

6. **The model runs on CPU fine** but make sure you call `.to(device)` consistently and don't mix CPU/CUDA tensors.

7. **Distribution shift is your biggest risk.** The Deep Dynamics model was trained on real (or simulated) driving data with a specific range of speeds, slip angles, and commands. If the RL agent explores far outside this distribution (e.g., full brake + full steer), the model's predictions become unreliable. Mitigations: constrain action space tightly; add observation/action clipping; monitor sysid parameter outputs for sanity; consider training-data-aware action penalties.

8. **Track CSV format.** Inspect the actual column headers — they may not be `x_center, y_center`. Adapt `Track.__init__` to whatever the real format is.

9. **No gradient through dynamics.** SAC is model-free — it treats `env.step()` as a black box. You do NOT need to backpropagate through the Deep Dynamics model. Keep it in `torch.no_grad()` eval mode.

10. **Reward scaling matters.** If `delta_s` per step is ~1.2m at 30 m/s (at Ts=0.04), your progress reward is ~1.2 per step and penalties are order 0.01–0.1. This is fine for SAC, but if training is unstable, normalize rewards.