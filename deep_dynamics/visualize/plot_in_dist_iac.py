#!/usr/bin/env python3
"""
In-Distribution Evaluation Visualization:
Plots Paper's PCNN vs Our PCNN (w/ weight shift) vs PCNN+PINN Hybrid
against IAC ground truth.
All models are trained directly on the IAC dataset (LVMS_23_01_04_A_5.npz).
"""

import sys, os, pickle, glob
import yaml, torch, numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from deep_dynamics.model.models import (
    string_to_model, string_to_dataset, ModelBase, create_module,
    nn, device
)

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("epoch_", "").replace(".pth", "")))
    return checkpoints[-1]

def collect_predictions(model, dataset_path, scaler_path, dataset_class_name):
    """Run inference with an already-loaded model. Returns (preds, gt) numpy arrays."""
    data_npy = np.load(dataset_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    dataset = string_to_dataset[dataset_class_name](
        data_npy["features"], data_npy["labels"], scaler
    )
    # Using the exact same dataset splits normally takes place, but we'll plot a continuous segment
    # For a fair visual, plotting the first 1000 steps of the validation split or just the whole dataset sequentially.
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

if __name__ == "__main__":
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATASET = os.path.join(BASE, "deep_dynamics/data/Putnam_park2023_run2_1_5.npz")

    # ---- 1. Paper's PCNN (DeepDynamicsIAC) ----
    paper_cfg_path = os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_iac.yaml")
    paper_dir = os.path.join(BASE, "../output/deep_dynamics_iac/iac_paper_pcnn/")
    paper_wts = get_latest_checkpoint(paper_dir)
    paper_scaler = os.path.join(paper_dir, "scaler.pkl")

    with open(paper_cfg_path, "rb") as f:
        paper_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    print(f"Loading Paper PCNN from {paper_wts}...")
    paper_model = string_to_model["DeepDynamicsIAC"](paper_cfg, eval=True)
    paper_model.to(device)
    paper_model.load_state_dict(torch.load(paper_wts, map_location=device))
    paper_model.eval()
    paper_preds, gt = collect_predictions(paper_model, DATASET, paper_scaler, "DeepDynamicsIAC")

    # ---- 2. Our PCNN (DeepDynamics with Weight Shift & Downforce) ----
    pcnn_cfg_path = os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_weightshift_iac.yaml")
    pcnn_dir = os.path.join(BASE, "../output/deep_dynamics_weightshift_iac/iac_weightshift_pcnn_v2/")
    pcnn_wts = get_latest_checkpoint(pcnn_dir)
    pcnn_scaler = os.path.join(pcnn_dir, "scaler.pkl")

    with open(pcnn_cfg_path, "rb") as f:
        pcnn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    print(f"Loading Modified PCNN from {pcnn_wts}...")
    pcnn_model = string_to_model["DeepDynamics"](pcnn_cfg, eval=True)
    pcnn_model.to(device)
    pcnn_model.load_state_dict(torch.load(pcnn_wts, map_location=device))
    pcnn_model.eval()
    pcnn_preds, _ = collect_predictions(pcnn_model, DATASET, pcnn_scaler, "DeepDynamics")

    # ---- 3. PCNN+PINN Hybrid ----
    hybrid_cfg_path = os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_iac.yaml")
    hybrid_dir = os.path.join(BASE, "../output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/")
    hybrid_wts = get_latest_checkpoint(hybrid_dir)
    hybrid_scaler = os.path.join(hybrid_dir, "scaler.pkl")

    with open(hybrid_cfg_path, "rb") as f:
        hybrid_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    print(f"Loading Hybrid PINN from {hybrid_wts}...")
    hybrid_model = string_to_model["DeepDynamicsPCNNPINN"](hybrid_cfg, eval=True)
    hybrid_model.to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_wts, map_location=device))
    hybrid_model.eval()
    hybrid_preds, _ = collect_predictions(hybrid_model, DATASET, hybrid_scaler, "DeepDynamicsPCNNPINN")

    # ---- Plot ----
    # Plotting only a slice to see detail since dataset is length 2428
    start_idx, end_idx = 500, 1500
    N = end_idx - start_idx
    t = np.arange(N) * 0.04  # IAC sampling time

    state_names = ["Longitudinal Velocity (m/s)", "Lateral Velocity (m/s)", "Yaw Rate (rad/s)"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle(
        "Out-Of-Distribution Overfit Evaluation: Putnam Park Circuit\n"
        "Models natively trained ONLY on Las Vegas Motor Speedway",
        fontsize=14, fontweight="bold",
    )

    colors = {
        "gt": "#2C3E50",
        "paper": "#9B59B6",      # purple
        "mod": "#E74C3C",        # red
        "hybrid": "#27AE60",     # green
    }

    for i, ax in enumerate(axes):
        ax.plot(t, gt[start_idx:end_idx, i], label="Ground Truth", color=colors["gt"], linewidth=1.3, alpha=0.9)
        ax.plot(t, paper_preds[start_idx:end_idx, i], label="Original PCNN (paper)", color=colors["paper"], linewidth=0.9, alpha=0.75)
        ax.plot(t, pcnn_preds[start_idx:end_idx, i], label="PCNN + Weight Shift/Downforce", color=colors["mod"], linewidth=0.9, alpha=0.75)
        ax.plot(t, hybrid_preds[start_idx:end_idx, i], label="PCNN+PINN Hybrid", color=colors["hybrid"], linewidth=0.9, alpha=0.75)
        ax.set_ylabel(state_names[i], fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (s) [Slice: t=1000:3000]", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(BASE, "../output/iac_ood_putnampark_3way.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()
