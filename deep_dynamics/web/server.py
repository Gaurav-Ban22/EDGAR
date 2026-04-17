import os
import glob
import yaml
import torch
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE)

from deep_dynamics.model.models import string_to_model, string_to_dataset, device

app = FastAPI(title="Deep Dynamics Trajectory Server")

# Serve the static website
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "deep_dynamics/web/static")), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(BASE, "deep_dynamics/web/static/index.html"))

# ----------------- Data Mappings -----------------
TRACKS = {
    "VEGAS_A": {
        "csv": "LVMS_23_01_04_A.csv",
        "npz": "LVMS_23_01_04_A_5.npz"
    },
    "VEGAS_B": {
        "csv": "LVMS_23_01_04_B.csv",
        "npz": "LVMS_23_01_04_B_5.npz"
    },
    "PUTNAM_2": {
        "csv": "Putnam_park2023_run2_1.csv",
        "npz": "Putnam_park2023_run2_1_5.npz"
    },
    "ETHZ_MOBIL": {
        "csv": None,
        "npz": "DYN-PP-ETHZMobil_5.npz"
    }
}

MODELS_INDY = [
    {
        "label": "Original PCNN",
        "color": "#9B59B6",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_iac.yaml",
        "dir": "../output/deep_dynamics_iac/iac_paper_pcnn/",
        "cls": "DeepDynamicsIAC"
    },
    {
        "label": "PCNN + Weight Shift",
        "color": "#E74C3C",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_weightshift_iac.yaml",
        "dir": "../output/deep_dynamics_weightshift_iac/iac_weightshift_pcnn_v2/",
        "cls": "DeepDynamics"
    },
    {
        "label": "Hybrid PCNN+PINN",
        "color": "#27AE60",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_iac.yaml",
        "dir": "../output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/",
        "cls": "DeepDynamicsPCNNPINN"
    }
]

MODELS_RC = [
    {
        "label": "Original PCNN (RC Baseline)",
        "color": "#9B59B6",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics.yaml",
        "dir": "../output/deep_dynamics/deep_dynamics_pcnn_absolute_base/",
        "cls": "DeepDynamics"
    },
    {
        "label": "Hybrid PCNN+PINN (RC)",
        "color": "#27AE60",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pinn.yaml",
        "dir": "../output/deep_dynamics_pcnnpinn/deep_dynamics_hybrid_v2/",
        "cls": "DeepDynamicsPCNNPINN"
    }
]

# Simple in-memory cache to avoid recomputing integrations
trajectory_cache = {}

def get_latest_checkpoint(checkpoint_dir):
    ch = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
    if not ch: return None
    ch.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    return ch[-1]

def integrate_trajectory(vx, vy, yaw_rate, dt=0.04, x0=0.0, y0=0.0, phi0=0.0):
    # Keeping this inside server.py just in case, but no longer used in /api/trajectories
    n = len(vx)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    phi = np.zeros(n + 1)
    x[0], y[0], phi[0] = x0, y0, phi0
    for i in range(n):
        phi[i + 1] = phi[i] + yaw_rate[i] * dt
        x[i + 1] = x[i] + (vx[i] * np.cos(phi[i]) - vy[i] * np.sin(phi[i])) * dt
        y[i + 1] = y[i] + (vx[i] * np.sin(phi[i]) + vy[i] * np.cos(phi[i])) * dt
    return x[1:], y[1:]

@app.get("/api/trajectories")
def get_trajectories(car: str, track: str):
    cache_key = f"{car}_{track}"
    if cache_key in trajectory_cache:
        return trajectory_cache[cache_key]
    
    if track not in TRACKS:
        raise HTTPException(status_code=400, detail="Unknown track")
    
    if car == "INDY":
        active_models = MODELS_INDY
    elif car == "RC":
        active_models = MODELS_RC
    else:
        raise HTTPException(status_code=400, detail="Unknown vehicle class")

    npz_path = os.path.join(BASE, "deep_dynamics/data", TRACKS[track]["npz"])
    if not os.path.exists(npz_path):
        raise HTTPException(status_code=500, detail=f"Dataset {npz_path} not found. Must parse first.")
        
    gps_x, gps_y, gps_phi = [], [], []
    x0, y0, phi0 = 0.0, 0.0, 0.0
    
    if TRACKS[track]["csv"] is not None:
        csv_path = os.path.join(BASE, "deep_dynamics/data", TRACKS[track]["csv"])
        df = pd.read_csv(csv_path, comment="#", header=None)
        gps_x_raw = df.iloc[:, 1].values
        gps_y_raw = df.iloc[:, 2].values
        gps_phi_raw = df.iloc[:, 5].values
        
        gps_x = gps_x_raw.tolist()
        gps_y = gps_y_raw.tolist()
        gps_phi = gps_phi_raw.tolist()
    
    data_npy = np.load(npz_path)
    features, labels = data_npy["features"], data_npy["labels"]
    
    response_data = {
        "gps": {
            "x": gps_x,
            "y": gps_y,
            "phi": gps_phi
        },
        "models": [],
        "ground_truth": {}
    }
    
    gt_calculated = False 
    
    for info in active_models:
        abs_cfg = os.path.join(BASE, info["cfg"])
        abs_dir = os.path.join(BASE, info["dir"])
        
        with open(abs_cfg, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        wts = get_latest_checkpoint(abs_dir)
        scaler_path = os.path.join(abs_dir, "scaler.pkl")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
        m = string_to_model[info["cls"]](cfg, eval=True).to(device)
        m.load_state_dict(torch.load(wts, map_location=device))
        m.eval()
        
        dataset_cls = string_to_dataset[info["cls"]]
        ds = dataset_cls(features, labels, scaler)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        
        preds_list, gt_list = [], []
        with torch.no_grad():
            for inp, lab, ninp in loader:
                inp, ninp = inp.to(device), ninp.to(device)
                if m.is_rnn:
                    h = m.init_hidden(inp.shape[0]).data
                    o, _, _ = m(inp, ninp, h)
                else:
                    o, _, _ = m(inp, ninp)
                preds_list.append(o.cpu().numpy())
                gt_list.append(lab.cpu().numpy())
                
        preds = np.concatenate(preds_list)
        gt_arr = np.concatenate(gt_list)
        
        response_data["models"].append({
            "name": info["label"],
            "color": info["color"],
            "vx": preds[:, 0].tolist(),
            "vy": preds[:, 1].tolist(),
            "yaw_rate": preds[:, 2].tolist()
        })
        
        if not gt_calculated:
            response_data["ground_truth"] = {
                "name": "Ground Truth (IMU)",
                "color": "#2C3E50",
                "vx": gt_arr[:, 0].tolist(),
                "vy": gt_arr[:, 1].tolist(),
                "yaw_rate": gt_arr[:, 2].tolist()
            }
            gt_calculated = True
            
    # Cache result
    trajectory_cache[cache_key] = response_data
    return response_data
