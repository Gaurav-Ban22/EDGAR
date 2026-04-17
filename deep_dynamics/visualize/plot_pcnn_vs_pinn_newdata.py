#!/usr/bin/env python3
"""
Zero-Shot Domain Transfer Visualization:
Plots Original PCNN vs Modified PCNN (w/ weight shift+downforce) vs PCNN+PINN Hybrid
against IAC ground truth.
All models were trained ONLY on the 41g RC car (ETHZ) — no retraining on IAC data.
"""

import sys, os, pickle
import yaml, torch, numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from deep_dynamics.model.models import (
    string_to_model, string_to_dataset, ModelBase, create_module,
    nn, device
)


class OriginalDeepDynamicsModel(ModelBase):
    """
    Exact replica of the original DeepDynamicsModel from the main branch (paper ODE).
    No weight shift, no downforce, no Ffy_max scaling.
    Uses sys_param_dict["Df"] and sys_param_dict["Dr"] directly in Pacejka.
    """
    def __init__(self, param_dict, eval=False):
        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module(
                    "DENSE",
                    param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"],
                    param_dict["MODEL"]["HORIZON"],
                    len(param_dict["PARAMETERS"]),
                    activation="Sigmoid",
                )
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"] - param_dict["PARAMETERS"][i]["Min"]
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
                return guard_output

        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, Ts=0.02):
        """Original paper ODE — no weight shift, no downforce."""
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        alphaf = steering - torch.atan2(
            self.vehicle_specs["lf"] * state_action_dict["YAW_RATE"] + state_action_dict["VY"],
            torch.abs(state_action_dict["VX"]),
        ) + sys_param_dict["Shf"]
        alphar = torch.atan2(
            self.vehicle_specs["lr"] * state_action_dict["YAW_RATE"] - state_action_dict["VY"],
            torch.abs(state_action_dict["VX"]),
        ) + sys_param_dict["Shr"]
        Frx = (sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * state_action_dict["VX"]) * throttle \
              - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * (state_action_dict["VX"] ** 2)
        # --- KEY DIFFERENCE: No Ffy_max/Fry_max scaling, raw Df/Dr ---
        Ffy = sys_param_dict["Svf"] + sys_param_dict["Df"] * torch.sin(
            sys_param_dict["Cf"] * torch.atan(
                sys_param_dict["Bf"] * alphaf
                - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))
            )
        )
        Fry = sys_param_dict["Svr"] + sys_param_dict["Dr"] * torch.sin(
            sys_param_dict["Cr"] * torch.atan(
                sys_param_dict["Br"] * alphar
                - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))
            )
        )
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:, 0] = 1 / self.vehicle_specs["mass"] * (Frx - Ffy * torch.sin(steering)) \
                      + state_action_dict["VY"] * state_action_dict["YAW_RATE"]
        dxdt[:, 1] = 1 / self.vehicle_specs["mass"] * (Fry + Ffy * torch.cos(steering)) \
                      - state_action_dict["VX"] * state_action_dict["YAW_RATE"]
        dxdt[:, 2] = 1 / sys_param_dict["Iz"] * (
            Ffy * self.vehicle_specs["lf"] * torch.cos(steering)
            - Fry * self.vehicle_specs["lr"]
        )
        dxdt *= Ts
        return x[:, -1, :3] + dxdt


def collect_predictions(model, dataset_path, scaler_path, dataset_class_name):
    """Run inference with an already-loaded model. Returns (preds, gt) numpy arrays."""
    data_npy = np.load(dataset_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    dataset = string_to_dataset[dataset_class_name](
        data_npy["features"], data_npy["labels"], scaler
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    preds_list, gt_list = [], []
    with torch.no_grad():
        for inputs, labels, norm_inputs in loader:
            inputs = inputs.to(device)
            norm_inputs = norm_inputs.to(device)
            if model.is_rnn:
                h = model.init_hidden(inputs.shape[0]).data
                output, h, _ = model(inputs, norm_inputs, h)
            else:
                output, _, _ = model(inputs, norm_inputs)
            preds_list.append(output.squeeze().cpu().numpy())
            gt_list.append(labels.squeeze().cpu().numpy())

    return np.array(preds_list), np.array(gt_list)


import glob

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not checkpoints:
        return None
    # Sort by epoch number to get the highest one reliably
    checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("epoch_", "").replace(".pth", "")))
    return checkpoints[-1]

if __name__ == "__main__":
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATASET = os.path.join(BASE, "deep_dynamics/data/LVMS_23_01_04_A_5.npz")

    # ---- Model configs and weights ----
    pcnn_cfg_path = os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics.yaml")
    pcnn_wts = os.path.join(BASE, "../output/deep_dynamics/deep_dynamics_pcnn_baseline/epoch_282.pth")
    pcnn_scaler = os.path.join(BASE, "../output/deep_dynamics/deep_dynamics_pcnn_baseline/scaler.pkl")

    hybrid_cfg_path = os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn.yaml")
    hybrid_dir = os.path.join(BASE, "../output/deep_dynamics_pcnnpinn/deep_dynamics_hybrid_v2/")
    hybrid_wts = get_latest_checkpoint(hybrid_dir)
    hybrid_scaler = os.path.join(hybrid_dir, "scaler.pkl")

    with open(pcnn_cfg_path, "rb") as f:
        pcnn_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(hybrid_cfg_path, "rb") as f:
        hybrid_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # --- 1. Original DeepDynamics (paper ODE — no weight shift, no downforce) ---
    print("Loading Original DeepDynamics (paper ODE) ...")
    original_model = OriginalDeepDynamicsModel(pcnn_cfg, eval=True)
    original_model.to(device)
    original_model.load_state_dict(torch.load(pcnn_wts, map_location=device))
    original_model.eval()
    original_preds, gt = collect_predictions(original_model, DATASET, pcnn_scaler, "DeepDynamics")

    # --- 2. Modified PCNN (with weight shift + downforce additions) ---
    print("Loading Modified PCNN (w/ weight shift + downforce) ...")
    pcnn_model = string_to_model["DeepDynamics"](pcnn_cfg, eval=True)
    pcnn_model.to(device)
    pcnn_model.load_state_dict(torch.load(pcnn_wts, map_location=device))
    pcnn_model.eval()
    pcnn_preds, _ = collect_predictions(pcnn_model, DATASET, pcnn_scaler, "DeepDynamics")

    # --- 3. PCNN+PINN Hybrid ---
    print("Loading PCNN+PINN Hybrid ...")
    hybrid_model = string_to_model["DeepDynamicsPCNNPINN"](hybrid_cfg, eval=True)
    hybrid_model.to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_wts, map_location=device))
    hybrid_model.eval()
    hybrid_preds, _ = collect_predictions(hybrid_model, DATASET, hybrid_scaler, "DeepDynamicsPCNNPINN")

    # ---- Plot ----
    N = len(gt)
    t = np.arange(N) * 0.04  # IAC sampling time

    state_names = ["Longitudinal Velocity (m/s)", "Lateral Velocity (m/s)", "Yaw Rate (rad/s)"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle(
        "Zero-Shot Domain Transfer: RC Car Models → IAC Full-Scale Racecar (LVMS)\n"
        "No Retraining — Frozen Weights Only",
        fontsize=14, fontweight="bold",
    )

    colors = {
        "gt": "#2C3E50",
        "original": "#9B59B6",   # purple
        "pcnn": "#E74C3C",       # red
        "hybrid": "#27AE60",     # green
    }

    for i, ax in enumerate(axes):
        ax.plot(t, gt[:, i], label="Ground Truth", color=colors["gt"], linewidth=1.3, alpha=0.9)
        ax.plot(t, original_preds[:, i], label="Original PCNN (paper)", color=colors["original"], linewidth=0.9, alpha=0.75)
        ax.plot(t, pcnn_preds[:, i], label="PCNN + Weight Shift/Downforce", color=colors["pcnn"], linewidth=0.9, alpha=0.75)
        ax.plot(t, hybrid_preds[:, i], label="PCNN+PINN Hybrid", color=colors["hybrid"], linewidth=0.9, alpha=0.75)
        ax.set_ylabel(state_names[i], fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # RMSE annotations
        orig_rmse = np.sqrt(np.mean((original_preds[:, i] - gt[:, i]) ** 2))
        pcnn_rmse = np.sqrt(np.mean((pcnn_preds[:, i] - gt[:, i]) ** 2))
        hybrid_rmse = np.sqrt(np.mean((hybrid_preds[:, i] - gt[:, i]) ** 2))
        ax.text(
            0.01, 0.90,
            f"Original: {orig_rmse:.4f}  |  +WeightShift: {pcnn_rmse:.4f}  |  Hybrid: {hybrid_rmse:.4f}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(BASE, "../output/iac_zero_shot_3way_comparison.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()
