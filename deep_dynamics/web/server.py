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
        "npz": "LVMS_23_01_04_A_5.npz"
    },
    "VEGAS_B": {
        "npz": "LVMS_23_01_04_B_5.npz"
    },
    "PUTNAM_2": {
        "npz": "Putnam_park2023_run2_1_5.npz"
    },
    "ETHZ_MOBIL": {
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
    },
    {
        "label": "Multi-Step Hybrid",
        "color": "#E67E22",
        "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_multistep_iac.yaml",
        "dir": "deep_dynamics/output/deep_dynamics_pcnnpinn_multistep_iac/iac_hybrid_multistep/",
        "cls": "DeepDynamicsPCNNPINNMultiStep"
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
        
    data_npy = np.load(npz_path, allow_pickle=True)
    features, labels = data_npy["features"], data_npy["labels"]
    
    # --- FIX #1: Use the POSES array from NPZ for GPS data ---
    # The poses array is aligned with the NPZ indices (filtered data).
    # Previously we read GPS from the raw CSV which has a completely different
    # number of rows (e.g. VEGAS_A: 39,811 CSV rows vs 13,414 NPZ rows).
    # poses[i] = [x, y, phi, vx, vy, vtheta, throttle, steering]
    gps_x, gps_y, gps_phi = [], [], []
    if "poses" in data_npy.files:
        poses = data_npy["poses"]
        gps_x = poses[:, 0].tolist()
        gps_y = poses[:, 1].tolist()
        gps_phi = poses[:, 2].tolist()
    
    # --- FIX #2: Trim trailing zero labels ---
    # The CSV parser allocates N entries but only fills N-5, leaving 5 zero labels.
    # Find and trim them so they don't corrupt the trajectory.
    num_valid = len(labels)
    for i in range(len(labels) - 1, -1, -1):
        if np.all(labels[i] == 0):
            num_valid = i
        else:
            break
    features = features[:num_valid]
    labels = labels[:num_valid]
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
