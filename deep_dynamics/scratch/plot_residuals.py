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
    ground_truth = []
    
    for inputs, labels, norm_inputs in test_data_loader:
        if model.is_rnn:
            h = model.init_hidden(inputs.shape[0]).data
            inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
            output, _, _ = model(inputs, norm_inputs, h)
        else:
            inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
            output, _, _ = model(inputs, norm_inputs)
            
        predictions.append(output.squeeze().cpu().detach().numpy())
        ground_truth.append(labels.squeeze().cpu().detach().numpy())
        
    return np.concatenate(predictions, axis=0), np.concatenate(ground_truth, axis=0)

def plot_residuals(slice_start, slice_end):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    dataset_path = "deep_dynamics/data/Putnam_park2023_run4_1_5.npz"
    data_npy = np.load(dataset_path)
    
    # Needs a scaler. We can pick the scaler from the multi-step model
    scaler_path = "deep_dynamics/output/deep_dynamics_pcnnpinn_multistep_iac/iac_hybrid_multistep/scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    # Model 1: Hybrid PCNN+PINN
    cfg_1 = "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_iac.yaml"
    
    # Model 2: Multi-Step Hybrid PCNN+PINN
    cfg_2 = "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_multistep_iac.yaml"
    model_2 = "deep_dynamics/output/deep_dynamics_pcnnpinn_multistep_iac/iac_hybrid_multistep/epoch_299.pth"
    
    # Model 3: Torque Vectoring Hybrid
    cfg_3 = "deep_dynamics/cfgs/model/deep_dynamics_pcnnpinn_tv_iac.yaml"
    model_3_candidates = glob.glob("deep_dynamics/output/deep_dynamics_pcnnpinn_tv_iac/iac_hybrid_tv/epoch_*.pth")
    if not model_3_candidates:
        print("Model 3 (TV) weights not found")
        sys.exit(1)
    model_3 = max(model_3_candidates, key=os.path.getctime)
    
    # we need to find actual checkpoint for model 1, we can search dynamically
    model_1_candidates = glob.glob("../output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/epoch_*.pth")
    if not model_1_candidates:
        model_1_candidates = glob.glob("deep_dynamics/output/deep_dynamics_pcnnpinn_iac/iac_hybrid_pinn/epoch_*.pth")
    
    if not model_1_candidates:
        print("Model 1 weights not found")
        sys.exit(1)
    # pick the latest
    model_1 = max(model_1_candidates, key=os.path.getctime)
    print(f"Using {model_1}, {model_2}, and {model_3}")

    preds_1, gt = evaluate_model(cfg_1, model_1, data_npy, scaler, device)
    preds_2, _ = evaluate_model(cfg_2, model_2, data_npy, scaler, device)
    preds_3, _ = evaluate_model(cfg_3, model_3, data_npy, scaler, device)
    
    # Slice Data
    time_arr = np.linspace(0, (slice_end-slice_start)*0.04, slice_end-slice_start)
    gt_slice = gt[slice_start:slice_end]
    p1_slice = preds_1[slice_start:slice_end]
    p2_slice = preds_2[slice_start:slice_end]
    p3_slice = preds_3[slice_start:slice_end]
    
    mae_1_vx = np.mean(np.abs(p1_slice[:,0] - gt_slice[:,0]))
    mae_1_vy = np.mean(np.abs(p1_slice[:,1] - gt_slice[:,1]))
    mae_1_yr = np.mean(np.abs(p1_slice[:,2] - gt_slice[:,2]))
    
    mae_2_vx = np.mean(np.abs(p2_slice[:,0] - gt_slice[:,0]))
    mae_2_vy = np.mean(np.abs(p2_slice[:,1] - gt_slice[:,1]))
    mae_2_yr = np.mean(np.abs(p2_slice[:,2] - gt_slice[:,2]))
    
    mae_3_vx = np.mean(np.abs(p3_slice[:,0] - gt_slice[:,0]))
    mae_3_vy = np.mean(np.abs(p3_slice[:,1] - gt_slice[:,1]))
    mae_3_yr = np.mean(np.abs(p3_slice[:,2] - gt_slice[:,2]))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    fig.suptitle('Out-Of-Distribution Overfit Evaluation: Putnam Park Circuit\nComparing Hybrid, Multi-Step, and Torque Vectoring physics models', fontsize=14, fontweight='bold')
    
    # Plot 1: Longitudinal Velocity
    axes[0].plot(time_arr, gt_slice[:,0], label='Ground Truth', color='#34495E', linewidth=1.5)
    axes[0].plot(time_arr, p1_slice[:,0], label='Original Hybrid', color='#27AE60', linewidth=1, alpha=0.7)
    axes[0].plot(time_arr, p2_slice[:,0], label='Multi-Step Hybrid', color='#E67E22', linewidth=1, alpha=0.8)
    axes[0].plot(time_arr, p3_slice[:,0], label='Torque Vectoring Hybrid', color='#3498DB', linewidth=1, alpha=0.9)
    axes[0].set_ylabel('Longitudinal Velocity (m/s)')
    axes[0].text(0.01, 0.95, f'Hybrid: {mae_1_vx:.4f} | Multi-Step: {mae_2_vx:.4f} | TV: {mae_3_vx:.4f}', transform=axes[0].transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].legend(loc='upper right', prop={'size': 8})
    
    # Plot 2: Lateral Velocity
    axes[1].plot(time_arr, gt_slice[:,1], label='Ground Truth', color='#34495E', linewidth=1.5)
    axes[1].plot(time_arr, p1_slice[:,1], label='Original Hybrid', color='#27AE60', linewidth=1, alpha=0.7)
    axes[1].plot(time_arr, p2_slice[:,1], label='Multi-Step Hybrid', color='#E67E22', linewidth=1, alpha=0.8)
    axes[1].plot(time_arr, p3_slice[:,1], label='Torque Vectoring Hybrid', color='#3498DB', linewidth=1, alpha=0.9)
    axes[1].set_ylabel('Lateral Velocity (m/s)')
    axes[1].text(0.01, 0.95, f'Hybrid: {mae_1_vy:.4f} | Multi-Step: {mae_2_vy:.4f} | TV: {mae_3_vy:.4f}', transform=axes[1].transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].legend(loc='upper right', prop={'size': 8})
    
    # Plot 3: Yaw Rate
    axes[2].plot(time_arr, gt_slice[:,2], label='Ground Truth', color='#34495E', linewidth=1.5)
    axes[2].plot(time_arr, p1_slice[:,2], label='Original Hybrid', color='#27AE60', linewidth=1, alpha=0.7)
    axes[2].plot(time_arr, p2_slice[:,2], label='Multi-Step Hybrid', color='#E67E22', linewidth=1, alpha=0.8)
    axes[2].plot(time_arr, p3_slice[:,2], label='Torque Vectoring Hybrid', color='#3498DB', linewidth=1, alpha=0.9)
    axes[2].set_ylabel('Yaw Rate (rad/s)')
    axes[2].set_xlabel(f'Time (s) [Slice: t={slice_start}:{slice_end}]')
    axes[2].text(0.01, 0.95, f'Hybrid: {mae_1_yr:.4f} | Multi-Step: {mae_2_yr:.4f} | TV: {mae_3_yr:.4f}', transform=axes[2].transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].legend(loc='upper right', prop={'size': 8})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('residuals_comparison.png', dpi=300)
    print("Saved residuals_comparison.png")

if __name__ == '__main__':
    # run slice 1000:3000
    plot_residuals(1000, 3000)
