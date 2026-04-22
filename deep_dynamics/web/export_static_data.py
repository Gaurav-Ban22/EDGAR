import sys, os, json, yaml, torch, numpy as np, pickle, glob

# Use real-time path discovery
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE)

from deep_dynamics.model.models import string_to_model, string_to_dataset, device

# Data definitions mirrored from server.py to ensure parity
TRACKS = {
    "VEGAS_A": {"npz": "LVMS_23_01_04_A_5.npz"},
    "VEGAS_B": {"npz": "LVMS_23_01_04_B_5.npz"},
    "PUTNAM_2": {"npz": "Putnam_park2023_run2_1_5.npz"},
    "ETHZ_MOBIL": {"npz": "DYN-PP-ETHZMobil_5.npz"}
}

MODELS_INDY = [
    {"label": "Original PCNN", "color": "#9B59B6", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_iac.yaml", "dir": "../output/deep_dynamics_iac/iac_paper_pcnn/", "cls": "DeepDynamicsIAC"},
    {"label": "PCNN + Weight Shift", "color": "#E74C3C", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_weightshift_iac.yaml", "dir": "../output/deep_dynamics_weightshift_iac/iac_weightshift_pcnn_v2/", "cls": "DeepDynamics"},
    {"label": "Hybrid PCNN+PINN", "color": "#27AE60", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_iac.yaml", "dir": "../output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/", "cls": "DeepDynamicsPCNNPINN"}
]

MODELS_RC = [
    {"label": "Original PCNN (RC Baseline)", "color": "#9B59B6", "cfg": "deep_dynamics/cfgs/model/deep_dynamics.yaml", "dir": "../output/deep_dynamics/deep_dynamics_pcnn_absolute_base/", "cls": "DeepDynamics"},
    {"label": "Hybrid PCNN+PINN (RC)", "color": "#27AE60", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pinn.yaml", "dir": "../output/deep_dynamics_pcnnpinn/deep_dynamics_hybrid_v2/", "cls": "DeepDynamicsPCNNPINN"}
]

def get_latest_checkpoint(checkpoint_dir):
    abs_dir = os.path.abspath(os.path.join(BASE, checkpoint_dir))
    ch = glob.glob(os.path.join(abs_dir, "epoch_*.pth"))
    if not ch: return None
    ch.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    return ch[-1]

def generate_json(car, track):
    if car == "INDY": active_models = MODELS_INDY
    else: active_models = MODELS_RC
    
    npz_path = os.path.join(BASE, "deep_dynamics/data", TRACKS[track]["npz"])
    data_npy = np.load(npz_path, allow_pickle=True)
    features, labels = data_npy["features"], data_npy["labels"]
    
    # Trim trailing zero labels (CSV parser leaves 5 zero entries at end)
    num_valid = len(labels)
    for i in range(len(labels) - 1, -1, -1):
        if np.all(labels[i] == 0):
            num_valid = i
        else:
            break
    features = features[:num_valid]
    labels = labels[:num_valid]
    
    # Use aligned GPS from NPZ poses array (not raw CSV!)
    gps_x, gps_y, gps_phi = [], [], []
    if "poses" in data_npy.files:
        poses = data_npy["poses"]
        gps_x = poses[:, 0].tolist()
        gps_y = poses[:, 1].tolist()
        gps_phi = poses[:, 2].tolist()
    
    response_data = {
        "gps": {"x": gps_x, "y": gps_y, "phi": gps_phi}, 
        "models": [], 
        "ground_truth": {
            "name": "Ground Truth (IMU)",
            "color": "#2C3E50",
            "vx": labels[:, 0].tolist(),
            "vy": labels[:, 1].tolist(),
            "yaw_rate": labels[:, 2].tolist()
        }
    }
    
    # Model predictions — output is (Batch, 3), NOT (Batch, 10, 3)
    for info in active_models:
        abs_cfg = os.path.join(BASE, info["cfg"])
        with open(abs_cfg, "r") as f: cfg = yaml.load(f, Loader=yaml.SafeLoader)
        wts = get_latest_checkpoint(info["dir"])
        scaler_path = os.path.abspath(os.path.join(BASE, info["dir"], "scaler.pkl"))
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        
        m = string_to_model[info["cls"]](cfg, eval=True).to(device)
        m.load_state_dict(torch.load(wts, map_location=device))
        m.eval()
        
        ds = string_to_dataset[info["cls"]](features, labels, scaler)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        
        preds_list = []
        with torch.no_grad():
            for inp, lab, ninp in loader:
                inp, ninp = inp.to(device), ninp.to(device)
                if m.is_rnn:
                    h = m.init_hidden(inp.shape[0]).data
                    o, _, _ = m(inp, ninp, h)
                else:
                    o, _, _ = m(inp, ninp)
                preds_list.append(o.cpu().numpy())
        
        preds = np.concatenate(preds_list)

        response_data["models"].append({
            "name": info["label"], 
            "color": info["color"],
            "vx": preds[:, 0].tolist(), 
            "vy": preds[:, 1].tolist(), 
            "yaw_rate": preds[:, 2].tolist()
        })
            
    return response_data

export_dir = os.path.join(BASE, "deep_dynamics/web/static/data")
os.makedirs(export_dir, exist_ok=True)

for car in ["INDY", "RC"]:
    tracks = ["VEGAS_A", "VEGAS_B", "PUTNAM_2"] if car == "INDY" else ["ETHZ_MOBIL"]
    for track in tracks:
        print(f"Exporting {car}_{track}...")
        data = generate_json(car, track)
        with open(os.path.join(export_dir, f"{car}_{track}.json"), "w") as f:
            json.dump(data, f)
            
print(f"\nSuccessfully exported all static data to {export_dir}")
