import torch
import numpy as np
import yaml
import pickle
import os
import matplotlib.pyplot as plt
from deep_dynamics.model.models import string_to_model, string_to_dataset
import sys
import glob

def evaluate_model(cfg_path, model_path, data_npy, scaler, device):
    with open(cfg_path, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"], scaler)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    predictions = []
    
    for inputs, labels, norm_inputs in test_data_loader:
        if model.is_rnn:
            h = model.init_hidden(inputs.shape[0]).data
            inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
            output, _, _ = model(inputs, norm_inputs, h)
        else:
            inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
            output, _, _ = model(inputs, norm_inputs)
            
        predictions.append(output.squeeze().cpu().detach().numpy())
        
    return np.concatenate(predictions, axis=0)

def integrate_trajectory(vx, vy, yaw_rate, dt, x0, y0, phi0):
    num_steps = len(vx)
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    phi = np.zeros(num_steps + 1)
    
    x[0] = x0
    y[0] = y0
    phi[0] = phi0
    
    for i in range(num_steps):
        next_phi = phi[i] + yaw_rate[i] * dt
        next_x = x[i] + (vx[i] * np.cos(phi[i]) - vy[i] * np.sin(phi[i])) * dt
        next_y = y[i] + (vx[i] * np.sin(phi[i]) + vy[i] * np.cos(phi[i])) * dt
        
        x[i+1] = next_x
        y[i+1] = next_y
        phi[i+1] = next_phi
        
    return x, y

def plot_birdseye():
    slice_start = 1871
    slice_end = 4100
    dt = 0.04
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dataset_path = "deep_dynamics/data/Putnam_park2023_run4_1_5.npz"
    data_npy = np.load(dataset_path)
    poses = data_npy["poses"]
    
    scaler_path = "deep_dynamics/output/deep_dynamics_pcnnpinn_multistep_iac/iac_hybrid_multistep/scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    models_config = [
        {"name": "Original Hybrid", "color": "#27AE60", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_iac.yaml", "dir": "deep_dynamics/output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/epoch_*.pth"},
        {"name": "Multi-Step Hybrid", "color": "#E67E22", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_multistep_iac.yaml", "dir": "deep_dynamics/output/deep_dynamics_pcnnpinn_multistep_iac/iac_hybrid_multistep/epoch_299.pth"},
        {"name": "Torque Vectoring Hybrid", "color": "#3498DB", "cfg": "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_tv_iac.yaml", "dir": "deep_dynamics/output/deep_dynamics_pcnnpinn_tv_iac/iac_hybrid_tv/epoch_399.pth"}
    ]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    fig.suptitle("Bird's-Eye Trajectory Reconstruction: Putnam Park\nModels trained on LVMS (oval) $\\rightarrow$ evaluated on Putnam Park (road course)", fontsize=14, fontweight='bold')
    
    # 1. Plot full GPS track outline
    gps_x = poses[:, 0]
    gps_y = poses[:, 1]
    ax.plot(gps_x, gps_y, label='GPS Track Outline', color='#E0E4E5', linewidth=8, alpha=0.8, zorder=0)
    
    # 2. Extract starting coordinates
    x0 = poses[slice_start, 0]
    y0 = poses[slice_start, 1]
    phi0 = poses[slice_start, 2]
    
    # 3. Plot Ground Truth (from poses)
    gt_x = poses[slice_start:slice_end, 0]
    gt_y = poses[slice_start:slice_end, 1]
    ax.plot(gt_x, gt_y, label='Ground Truth (GPS)', color='#34495E', linewidth=2, zorder=1)
    
    # Evaluate models and plot
    for mc in models_config:
        search_pattern = mc["dir"]
        # Fallback for parent directory output if not in deep_dynamics
        if not glob.glob(search_pattern):
            search_pattern = "../" + search_pattern.replace("deep_dynamics/output", "output")
        
        candidates = glob.glob(search_pattern)
        if not candidates:
            print(f"Weights not found for {mc['name']} using {search_pattern}")
            continue
        best_model = max(candidates, key=os.path.getctime)
        print(f"Evaluating {mc['name']} with {best_model}")
        
        try:
            preds = evaluate_model(mc["cfg"], best_model, data_npy, scaler, device)
            preds_slice = preds[slice_start:slice_end]
            
            vx = preds_slice[:, 0]
            vy = preds_slice[:, 1]
            yr = preds_slice[:, 2]
            
            p_x, p_y = integrate_trajectory(vx, vy, yr, dt, x0, y0, phi0)
            ax.plot(p_x, p_y, label=mc['name'], color=mc['color'], linewidth=1.5, alpha=0.85, zorder=2)
            
        except Exception as e:
            print(f"Failed to evaluate {mc['name']}: {e}")
            
    # Mark the start
    ax.scatter([x0], [y0], facecolors='#F39C12', edgecolors='black', s=100, zorder=5, label='Start')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('birdseye_comparison.png', dpi=300)
    print("Saved birdseye_comparison.png")

if __name__ == '__main__':
    plot_birdseye()
