#!/usr/bin/env python3
"""
Bird's-eye trajectory reconstruction.

Takes the ground-truth heading (phi) and each model's predicted
body-frame velocities (VX, VY, yaw_rate) and integrates them into
2D (X, Y) global positions.  Plots against the real GPS track outline.
"""

import sys, os, pickle, glob
import yaml, torch, numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from deep_dynamics.model.models import string_to_model, string_to_dataset, device


# ------------------------------------------------------------------ helpers
def get_latest_checkpoint(d):
    ch = glob.glob(os.path.join(d, "epoch_*.pth"))
    if not ch:
        return None
    ch.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    return ch[-1]


def collect_predictions(model, dataset_path, scaler_path, dataset_cls):
    data_npy = np.load(dataset_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    ds = string_to_dataset[dataset_cls](data_npy["features"], data_npy["labels"], scaler)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    preds, gt = [], []
    with torch.no_grad():
        for inp, lab, ninp in loader:
            inp, ninp = inp.to(device), ninp.to(device)
            if model.is_rnn:
                h = model.init_hidden(inp.shape[0]).data
                o, _, _ = model(inp, ninp, h)
            else:
                o, _, _ = model(inp, ninp)
            preds.append(o.cpu().numpy())
            gt.append(lab.cpu().numpy())
    return np.concatenate(preds), np.concatenate(gt)


def integrate_trajectory(vx, vy, yaw_rate, dt, x0=0.0, y0=0.0, phi0=0.0):
    """Dead-reckoning integration of body-frame velocities into global XY."""
    n = len(vx)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    phi = np.zeros(n + 1)
    x[0], y[0], phi[0] = x0, y0, phi0

    for i in range(n):
        phi[i + 1] = phi[i] + yaw_rate[i] * dt
        # body → global
        x[i + 1] = x[i] + (vx[i] * np.cos(phi[i]) - vy[i] * np.sin(phi[i])) * dt
        y[i + 1] = y[i] + (vx[i] * np.sin(phi[i]) + vy[i] * np.cos(phi[i])) * dt

    return x[1:], y[1:]  # drop initial condition to align with predictions


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # ---------- ground-truth GPS from original CSV ----------
    csv_path = os.path.join(BASE, "deep_dynamics/data/LVMS_23_01_04_B.csv")
    df = pd.read_csv(csv_path, comment="#", header=None)
    # columns: 0=time, 1=x, 2=y, 3=vx, 4=vy, 5=phi, 6=delta, 7=omega ...
    gps_x = df.iloc[:, 1].values
    gps_y = df.iloc[:, 2].values
    gps_phi = df.iloc[:, 5].values
    gps_vx = df.iloc[:, 3].values
    gps_vy = df.iloc[:, 4].values

    dt = 0.04  # 25 Hz telemetry

    # The dataset corresponds to the CSV but is shifted by Horizon
    DATASET = os.path.join(BASE, "deep_dynamics/data/LVMS_23_01_04_B_5.npz")
    
    # We slice out a 2500 step window (~100 seconds) so integration drift doesn't obscure the path
    start_csv_row = 1500  
    slice_len = 2500
    
    x0 = gps_x[start_csv_row]
    y0 = gps_y[start_csv_row]
    phi0 = gps_phi[start_csv_row]

    # ---------- load models ----------
    models_info = [
        {
            "label": "Original PCNN (paper)",
            "color": "#9B59B6",
            "cfg": os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_iac.yaml"),
            "dir": os.path.join(BASE, "../output/deep_dynamics_iac/iac_paper_pcnn/"),
            "cls": "DeepDynamicsIAC",
        },
        {
            "label": "PCNN + Weight Shift",
            "color": "#E74C3C",
            "cfg": os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_weightshift_iac.yaml"),
            "dir": os.path.join(BASE, "../output/deep_dynamics_weightshift_iac/iac_weightshift_pcnn_v2/"),
            "cls": "DeepDynamics",
        },
        {
            "label": "PCNN+PINN Hybrid",
            "color": "#27AE60",
            "cfg": os.path.join(BASE, "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_iac.yaml"),
            "dir": os.path.join(BASE, "../output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/"),
            "cls": "DeepDynamicsPCNNPINN",
        },
    ]

    trajectories = {}
    for info in models_info:
        with open(info["cfg"], "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        wts = get_latest_checkpoint(info["dir"])
        scaler = os.path.join(info["dir"], "scaler.pkl")
        print(f"Loading {info['label']} from {wts} ...")
        m = string_to_model[info["cls"]](cfg, eval=True).to(device)
        m.load_state_dict(torch.load(wts, map_location=device))
        m.eval()
        preds, gt = collect_predictions(m, DATASET, scaler, info["cls"])
        
        # Apply slice offset (preds index 0 corresponds to csv row 5, so subtract 5)
        p_start = start_csv_row - 5
        preds_slice = preds[p_start : p_start+slice_len]
        tx, ty = integrate_trajectory(preds_slice[:, 0], preds_slice[:, 1], preds_slice[:, 2], dt, x0, y0, phi0)
        trajectories[info["label"]] = (tx, ty, info["color"])

    # Ground-truth trajectory from model's own gt labels
    gt_slice = gt[p_start : p_start+slice_len]
    gt_tx, gt_ty = integrate_trajectory(gt_slice[:, 0], gt_slice[:, 1], gt_slice[:, 2], dt, x0, y0, phi0)

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(
        "Bird's-Eye Trajectory: Las Vegas Motor Speedway Run B\n"
        "Same Track, Different Laps. Models verifying their in-distribution consistency.",
        fontsize=14, fontweight="bold",
    )

    # Real GPS track (only graphing the slice)
    slice_end = start_csv_row + slice_len
    ax.plot(gps_x[start_csv_row:slice_end], gps_y[start_csv_row:slice_end], color="#BDC3C7", linewidth=6, alpha=0.5, label="Raw GPS Reference Outline", zorder=1)

    # Ground truth (integrated from labels)
    ax.plot(gt_tx, gt_ty, color="#2C3E50", linewidth=2.0, label="Ground Truth (integrated)", zorder=2)

    # Model trajectories
    for label, (tx, ty, color) in trajectories.items():
        ax.plot(tx, ty, color=color, linewidth=1.5, alpha=0.85, label=label, zorder=3)

    # Start marker
    ax.scatter([x0], [y0], s=120, color="#F39C12", edgecolors="k", zorder=5, label="Start")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(BASE, "../output/trajectory_lvms_b.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    plt.show()
