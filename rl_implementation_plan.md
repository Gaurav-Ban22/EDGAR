1. Purpose and scientific framing
This repository implements Deep Dynamics (and variants): a physics-constrained neural network (PCNN) that learns to predict short-horizon vehicle body-axis dynamics while internally estimating physically interpretable parameters of a dynamic single-track (bicycle) model with Pacejka-style lateral forces and a longitudinal drivetrain / drag model. Training is supervised on logged (or simulated) trajectories; the learned model is intended for control-oriented use (e.g. re-parameterizing an MPC inner loop, as in mpc/run_nmpc_orca_deep_dynamics.py).

The codebase does not contain an RL agent, a Gym-style environment, or a reward function. It provides: data pipelines, Torch models, training/eval, optional Ray hyperparameter search, Bayesrace-based NMPC demos, and visualization / error analysis.

2. Repository structure (Python surface area)
Area	Role
deep_dynamics/model/	Core: models.py, build_network.py, train.py, evaluate.py, tune_hyperparameters.py, test_hyperparameters.py
deep_dynamics/cfgs/model/*.yaml	Declarative model + vehicle + optimization specs
deep_dynamics/tools/	csv_parser.py (IAC CSV → .npz), bayesrace_parser.py (Bayesrace .npz → training .npz), rosbag_data_convert.py (ROS2 bags → CSV; environment-specific paths)
deep_dynamics/mpc/	NMPC scripts coupling Bayesrace (Dynamic, tracks, CasADi NLP) with learned parameters from a checkpoint
deep_dynamics/visualize/	Plots: dataset animation, prediction/error vs track, tire curves, MPC comparison; many scripts assume CUDA and hard-coded paths
EDGAR/pyproject.toml	Package deep_dynamics: torch, numpy, matplotlib, scikit-learn, wandb, ray[tune]
3. Problem formulation (what is being learned)
3.1 Predictive task
For each training sample:

Input: A fixed-length history of length H = MODEL.HORIZON (in time steps, not seconds). Each step stacks states and actions as defined in YAML STATE and ACTIONS.
Output / label: The next body-frame velocities ((v_x, v_y, \dot\psi)) after one discrete step, i.e. the same three channels as the first three STATE entries, advanced by one sampling period.
So the network is trained for one-step ahead prediction in ((v_x, v_y, \omega)) space, with dynamics implied by a differentiable bicycle model whose coefficients are network outputs (or directly predicted in the Pacejka variant).

3.2 Sampling time
IAC CSV pipeline: SAMPLING_TIME = 0.04 s (25 Hz) in csv_parser.py; must match differential_equation(..., Ts=0.04) in DeepDynamicsModelIAC / DeepPacejkaModelIAC.
Bayesrace / ORCA configs: Often Ts = 0.02 in non-IAC DeepDynamicsModel / DeepPacejkaModel.
Any RL sim must use the same (T_s) as the dataset and config used to train the checkpoint, or the learned parameters are mis-scaled in time.

4. Data: formats, parsers, feature layout
4.1 IAC CSV → csv_parser.write_dataset
Reads a header row; column names are matched with row[i].split("(")[0] (strips units in parentheses).
Required fields include: vx, vy, omega, delta, brake_ped_cmd, throttle_ped_cmd, x, y, phi.
Start gate: While (|v_x| < 5) m/s, the parser updates “previous” throttle/steer but does not append commands; once speed exceeds 5 m/s, logging starts and incremental commands are built:
throttle_cmd = throttle - previous_throttle
steering_cmd = steering - previous_steer
then previous_* are updated by adding these increments (integrating commands into absolute throttle/steer consistent with odometry rows).
Odometry row: [vx, vy, yaw_rate, throttle_fb, steering_fb] — here throttle_fb and steering_fb are the absolute pedal/angle after integration, not raw deltas.
Feature tensor per sample: shape (H, 8):
Columns 0–4: last five entries of odometry over the window (vx, vy, ω, throttle_fb, steer_fb).
Columns 5–6: throttle_cmd, steering_cmd over the window.
Column 7: vx at a fixed lead index — implemented as odometry[i+5:i+H+5, 0] (a 5-step offset lookahead in (v_x); this is a deliberate data artifact to match the published pipeline, not a generic “8th state”).
Label: odometry[i+H][:3] → next-step ((v_x,v_y,\omega)).
Saves np.savez(..., features=, labels=, poses=) with full pose array for visualization.
DeepDynamicsDataset uses only the first 7 features per timestep (features[:,:,:7]), dropping the 8th column for training. So the 8th channel is unused in the IAC dataset class despite being computed.

4.2 Bayesrace .npz → bayesrace_parser.write_dataset
Expects keys: dstates, inputs, states, vrefs.
Builds analogous sliding windows; saves features, labels (poses included in return; save path may omit poses depending on version — check current file).
4.3 Normalization
DatasetBase applies sklearn.preprocessing.StandardScaler fit on training split to flattened features (N*H, F) and stores X_norm for network input. Labels are not normalized. Scaler is pickled next to checkpoints for evaluation.

5. Neural architecture (build_network.py + ModelBase)
5.1 Input dimension
First layer input size = (num_states + num_actions) * HORIZON for FFNN path. For configs where the first layer is GRU, create_module builds nn.GRU(input_size // H, H, num_layers, batch_first=True) — i.e. per-timestep input width (num_states + num_actions) after the Flatten inserted in ModelBase for RNN (implementation detail: Flatten is between RNN and following layers in the module list).

5.2 Forward pass (ModelBase.forward)
x: Raw feature sequence (batch, H, F) — used by physics head.
x_norm: Scaler-normalized features — fed through feed_forward.
Stack: Either GRU → flatten → dense tower, or MLP on flattened x_norm.
Last tensor ff: Penultimate representation before the parameter head.
differential_equation(x, ff): Maps current physical state (from last timestep of x) and network outputs to predicted next ((v_x,v_y,\omega)).
Returns (prediction, h, sysid) where sysid is the raw parameter vector (guarded or unguarded depending on model class).

5.3 Two model families
A. DeepDynamicsModel / DeepDynamicsModelIAC

Final GuardLayer: Dense → Sigmoid → elementwise sigmoid * (Max−Min) + Min for each entry in PARAMETERS (Pacejka B–E, drivetrain Cm1/Cm2, drag Cr0/Cr2, Iz, slip offsets Shf, Shr, lateral offsets Svf, Svr, etc.).
So the network never outputs out-of-range physical coefficients by construction (bounded outputs).
B. DeepPacejkaModel / DeepPacejkaModelIAC

Last layer is linear (no sigmoid guard); parameters are unbounded in the network (paper compares this “Deep Pacejka” variant).
5.4 Vehicle-specific YAML
VEHICLE_SPECS: lf, lr, mass (and for Pacejka non-IAC, Iz in VEHICLE_SPECS where used). IAC DeepDynamics learns Iz as a parameter with large min/max (500–2000) in yaml.

PARAMETERS lists learned symbols; optional ground-truth values in Bayesrace yaml are for evaluation only (unpack_sys_params returns both dicts).

6. Physics head: differentiable bicycle + Pacejka (differential_equation)
Common structure (IAC Deep Dynamics example):

Unpack parameters from ff into sys_param_dict (Bf, Cf, Df, Ef, Br, Cr, Dr, Er, Cm1, Cm2, Cr0, Cr2, Iz, Shf, Shr, Svf, Svr, …).
Unpack last-timestep state/actions from x[:, -1, :]:
States: VX, VY, YAW_RATE, THROTTLE_FB, STEERING_FB
Actions: THROTTLE_CMD, STEERING_CMD
Total steering / throttle used in forces:
steering = STEERING_FB + STEERING_CMD
throttle = THROTTLE_FB + THROTTLE_CMD
Slip angles (\alpha_f), (\alpha_r) from bicycle geometry (uses lf, lr, VX, VY, YAW_RATE) plus optional shifts Shf, Shr.
Lateral forces (F_{fy}, F_{ry}) via Magic Formula–style sin(C * atan(B*α - E*(B*α - atan(B*α)))) with vertical shifts Svf, Svr.
Longitudinal: (F_{rx} = (Cm_1 - Cm_2 v_x)\cdot \text{throttle} - Cr_0 - Cr_2 v_x^2) (Deep Dynamics); Deep Pacejka instead predicts Frx as a direct parameter in sys_param_dict for the longitudinal channel.
Planar dynamics:
(\dot v_x = \frac{1}{m}(F_{rx} - F_{fy}\sin\delta) + v_y \omega)
(\dot v_y = \frac{1}{m}(F_{ry} + F_{fy}\cos\delta) - v_x \omega)
(\dot\omega = \frac{1}{I_z}(F_{fy} l_f \cos\delta - F_{ry} l_r))
Discrete update: Euler step: x_next[:3] = x_last[:3] + Ts * dxdt.
Training/eval objective: MSE between this one-step integrated prediction and logged next velocity.

Note: In DeepDynamicsModel (non-IAC), line 177 names Frx_desired but line 181 uses Frx in dxdt — likely a bug (NameError unless another path defines Frx); IAC class uses Frx consistently. RL planning should treat IAC checkpoints as the reference implementation path you actually trained.

7. Training (train.py)
Loads yaml → instantiates string_to_model[MODEL.NAME].
Loads .npz → string_to_dataset[MODEL.NAME].
80/20 random split (torch.manual_seed(0)).
Loss: weighted_mse_loss with weights [1,1,1] on the three outputs (structure allows per-axis weighting).
Optimization: Adam (or as per yaml), full model.to(device); batches use drop_last=True on train.
Checkpointing: Saves epoch_{k}.pth when validation loss improves; saves scaler.pkl once per experiment.
Optional: Weights & Biases (wandb), Ray Tune reporting in tune_hyperparameters.py (Optuna search, ASHA scheduler, hard-coded dataset and filesystem paths in that file).
8. Evaluation (evaluate.py)
Loads model in eval=True (MSE with reduction='none' for per-sample vectors).
Batch size 1, full pass over dataset.
Prints RMSE, max absolute error per dimension, mean inference time, optional coefficient statistics vs yaml ground truth if --eval_coeffs.
9. MPC integration (mpc/run_nmpc_orca_deep_dynamics.py)
External dependencies: casadi, bayes_race (tracks ETHZMobil, vehicle Dynamic, setupNLP, pure pursuit, etc.).

Conceptual loop:

Ground truth simulation advances a full nonlinear Bayesrace Dynamic model with true ORCA params (model.sim_continuous).
After a short warm-up, a window of past dstates and inputs is assembled to match ddm.horizon, scaled, and fed to the trained DeepDynamics network.
The output parameter vector overwrites entries in a params dict passed to Dynamic(**params), building a new approximate vehicle model dpm_model.
CasADi NMPC (setupNLP) is re-built with this updated model (expensive; done each iteration in the script).
MPC solves for control; first input applied; repeat.
So this repo’s “closed loop” demo is simulator = Bayesrace, learned model = time-varying physical parameters for MPC, not learned dynamics alone as the sole integrator.

10. Visualization (selected behavior)
plot_dataset.py: Time-series PNGs of states/commands.
plot_predictions_iac.py: Rolls out learned differential_equation in a loop: integrates pose in global frame using predicted ((v_x,v_y,\omega)) while re-using recorded future commands from features for the non-velocity channels — a partial open-loop test hybridized with logged inputs.
Track boundaries: CSVs under visualize/tracks/ (LVMS, Putnam).
11. Interfaces relevant to an RL agent (mapping)
Concept	In this codebase
Observation	History of length H: body velocities, integrated throttle/steer, command deltas; IAC training drops channel 8. RL must decide whether to match 7- or 8-d history and same semantics.
Action	THROTTLE_CMD, STEERING_CMD as increments added to feedback channels in the physics head (not necessarily the same as raw simulator API).
Dynamics	One-step Euler update in vehicle frame; no full pose state inside the network — pose must be integrated externally (as in visualization).
Reward / termination	Not defined here.
Environment	Not included. Closest “env” is Bayesrace + NMPC scripts or your own sim.
For RL in a simulated car:

Align sim’s state/control interfaces with yaml STATE/ACTIONS and (T_s).
Choose whether the policy predicts increments (consistent with training) or absolute commands (then map inside the env).
Decide if the learned Deep Dynamics is used as world model for MBRL, as differentiable layer, or only for analysis; note distribution shift vs IAC logs.
If using only this Torch model as transition function: implement same unpack_state_actions, same differential_equation, and same scaler normalization for the network input.
12. Implementation caveats (for a planning agent)
Device: Many scripts call .cuda() unconditionally; CPU/MPS require edits.
tune_hyperparameters.py / test_hyperparameters.py: Contain absolute paths from the original author’s machine.
rosbag_data_convert.py: Hard-coded ROS message paths under /home/chros/... — not portable without refactoring.
DeepDynamicsModel (non-IAC) possible Frx / Frx_desired inconsistency — verify before research use.
RNN + forward loop: Subtle indexing (feed_forward[0] reused inside loop for RNN); GRU hidden size tied to HORIZON in config.
