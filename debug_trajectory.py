#!/usr/bin/env python3
"""Diagnostic script: check data alignment, dt, and integration drift."""
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# --- 1. Check NPZ vs CSV alignment ---
for track_name, npz_name, csv_name in [
    ("VEGAS_A", "LVMS_23_01_04_A_5.npz", "LVMS_23_01_04_A.csv"),
    ("VEGAS_B", "LVMS_23_01_04_B_5.npz", "LVMS_23_01_04_B.csv"),
    ("PUTNAM_2", "Putnam_park2023_run2_1_5.npz", "Putnam_park2023_run2_1.csv"),
]:
    npz_path = os.path.join(BASE, "deep_dynamics/data", npz_name)
    csv_path = os.path.join(BASE, "deep_dynamics/data", csv_name)
    
    if not os.path.exists(npz_path) or not os.path.exists(csv_path):
        print(f"\n=== {track_name}: SKIPPED (file not found) ===")
        continue
    
    print(f"\n{'='*60}")
    print(f"  {track_name}")
    print(f"{'='*60}")
    
    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    has_poses = "poses" in data.files
    print(f"NPZ keys: {data.files}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}")
    print(f"Has poses:      {has_poses}")
    
    if has_poses:
        poses = data["poses"]
        print(f"Poses shape:    {poses.shape}")
    
    # Load raw CSV
    df = pd.read_csv(csv_path, comment="#", header=None)
    print(f"\nRaw CSV rows:   {len(df)}")
    print(f"CSV columns:    {df.shape[1]}")
    
    # The server reads GPS from columns 1,2,5 of raw CSV
    gps_x_csv = df.iloc[:, 1].values.astype(float)
    gps_y_csv = df.iloc[:, 2].values.astype(float)
    gps_phi_csv = df.iloc[:, 5].values.astype(float)
    
    print(f"\n--- ALIGNMENT MISMATCH ---")
    print(f"GPS array length (raw CSV):     {len(gps_x_csv)}")
    print(f"GT/Model array length (NPZ):    {len(labels)}")
    print(f"Difference (MISALIGNED ROWS):   {len(gps_x_csv) - len(labels)}")
    
    if has_poses:
        print(f"Poses array length:             {len(poses)}")
        print(f"Poses-Labels diff:              {len(poses) - len(labels)}")
        
        # Check: do poses GPS match the expected positions?
        # Poses[i] = [x, y, phi, vx, vy, vtheta, throttle, steering]
        # Labels[i] = [vx, vy, yaw_rate] at future timestep
        print(f"\n--- POSES vs CSV SPOT CHECK (first 5 poses) ---")
        for i in range(min(5, len(poses))):
            print(f"  poses[{i}]: x={poses[i,0]:.3f}, y={poses[i,1]:.3f}, phi={poses[i,2]:.4f}")
        print(f"  ...")
        for i in range(min(5, len(gps_x_csv))):
            print(f"  csv[{i}]:   x={gps_x_csv[i]:.3f}, y={gps_y_csv[i]:.3f}, phi={gps_phi_csv[i]:.4f}")
    
    # --- 2. Check for zero-padded labels at the end ---
    print(f"\n--- ZERO-PADDED LABELS CHECK ---")
    zero_count = 0
    for i in range(len(labels) - 1, -1, -1):
        if np.all(labels[i] == 0):
            zero_count += 1
        else:
            break
    print(f"Trailing zero labels: {zero_count}")
    
    # --- 3. Check label statistics ---
    print(f"\n--- LABEL STATISTICS ---")
    print(f"  VX  range: [{labels[:,0].min():.4f}, {labels[:,0].max():.4f}], mean={labels[:,0].mean():.4f}")
    print(f"  VY  range: [{labels[:,1].min():.4f}, {labels[:,1].max():.4f}], mean={labels[:,1].mean():.4f}")
    print(f"  YAW range: [{labels[:,2].min():.4f}, {labels[:,2].max():.4f}], mean={labels[:,2].mean():.4f}")
    
    # --- 4. Check time spacing from CSV ---
    if df.shape[1] > 0:
        times = df.iloc[:, 0].values.astype(float)
        dts = np.diff(times)
        print(f"\n--- TIME SPACING (from CSV column 0) ---")
        print(f"  Mean dt:   {dts.mean():.6f} s")
        print(f"  Std dt:    {dts.std():.6f} s")
        print(f"  Min dt:    {dts.min():.6f} s")
        print(f"  Max dt:    {dts.max():.6f} s")
        print(f"  Expected:  0.04 s (25 Hz)")
    
    # --- 5. Dead-reckoning drift test on GROUND TRUTH ---
    # Use poses (aligned GPS) for initial conditions if available
    if has_poses:
        # Pick a window that covers ~1 lap
        start_idx = 0
        end_idx = min(len(labels), 3000)  # ~120 seconds at 25Hz
        
        gt_vx = labels[start_idx:end_idx, 0]
        gt_vy = labels[start_idx:end_idx, 1]
        gt_yr = labels[start_idx:end_idx, 2]
        
        # These are the FILTERED poses, aligned with NPZ indices
        x0, y0, phi0 = poses[start_idx + 5, 0], poses[start_idx + 5, 1], poses[start_idx + 5, 2]
        
        # Integrate with dt=0.04
        dt = 0.04
        n = len(gt_vx)
        x_int = np.zeros(n + 1)
        y_int = np.zeros(n + 1)
        phi_int = np.zeros(n + 1)
        x_int[0], y_int[0], phi_int[0] = x0, y0, phi0
        
        for i in range(n):
            phi_int[i+1] = phi_int[i] + gt_yr[i] * dt
            x_int[i+1] = x_int[i] + (gt_vx[i] * np.cos(phi_int[i]) - gt_vy[i] * np.sin(phi_int[i])) * dt
            y_int[i+1] = y_int[i] + (gt_vx[i] * np.sin(phi_int[i]) + gt_vy[i] * np.cos(phi_int[i])) * dt
        
        # Compare final integrated position to actual GPS position
        actual_end_idx = min(start_idx + 5 + end_idx, len(poses) - 1)
        actual_x_end = poses[actual_end_idx, 0]
        actual_y_end = poses[actual_end_idx, 1]
        actual_phi_end = poses[actual_end_idx, 2]
        
        drift_x = x_int[-1] - actual_x_end
        drift_y = y_int[-1] - actual_y_end
        drift_phi = phi_int[-1] - actual_phi_end
        drift_pos = np.sqrt(drift_x**2 + drift_y**2)
        
        print(f"\n--- DEAD-RECKONING DRIFT (GT over {n} steps = {n*dt:.1f}s) ---")
        print(f"  Start GPS:     ({x0:.2f}, {y0:.2f}), phi={phi0:.4f}")
        print(f"  Integrated end:({x_int[-1]:.2f}, {y_int[-1]:.2f}), phi={phi_int[-1]:.4f}")
        print(f"  Actual GPS end:({actual_x_end:.2f}, {actual_y_end:.2f}), phi={actual_phi_end:.4f}")
        print(f"  Position drift: {drift_pos:.2f} m")
        print(f"  Heading drift:  {np.degrees(drift_phi):.2f} degrees")
        
        # --- 6. Alternative: use GPS phi at each step instead of integrating yaw_rate ---
        x_gps = np.zeros(n + 1)
        y_gps = np.zeros(n + 1)
        x_gps[0], y_gps[0] = x0, y0
        
        for i in range(n):
            pose_idx = min(start_idx + 5 + i, len(poses) - 1)
            phi_gps = poses[pose_idx, 2]
            x_gps[i+1] = x_gps[i] + (gt_vx[i] * np.cos(phi_gps) - gt_vy[i] * np.sin(phi_gps)) * dt
            y_gps[i+1] = y_gps[i] + (gt_vx[i] * np.sin(phi_gps) + gt_vy[i] * np.cos(phi_gps)) * dt
        
        drift_x2 = x_gps[-1] - actual_x_end
        drift_y2 = y_gps[-1] - actual_y_end
        drift_pos2 = np.sqrt(drift_x2**2 + drift_y2**2)
        
        print(f"\n--- WITH GPS HEADING (no yaw_rate integration) ---")
        print(f"  Integrated end:({x_gps[-1]:.2f}, {y_gps[-1]:.2f})")
        print(f"  Position drift: {drift_pos2:.2f} m")
        print(f"  IMPROVEMENT:    {drift_pos/max(drift_pos2,0.001):.1f}x better")

print(f"\n{'='*60}")
print("DONE")
