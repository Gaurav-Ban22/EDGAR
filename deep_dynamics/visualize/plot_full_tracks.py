#!/usr/bin/env python3
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 1. Plot Full Putnam Park Outline
csv_path_putnam = os.path.join(BASE, "deep_dynamics/data/Putnam_park2023_run4_1.csv")
df_putnam = pd.read_csv(csv_path_putnam, comment="#", header=None)
x_p = df_putnam.iloc[:, 1].values
y_p = df_putnam.iloc[:, 2].values

axes[0].plot(x_p, y_p, color="#34495E", linewidth=3)
axes[0].scatter([x_p[0]], [y_p[0]], color="#E74C3C", s=150, edgecolors="k", label="Start Line")
axes[0].set_aspect("equal")
axes[0].set_title("Full Putnam Park Circuit\n(Raw GPS Coordinates)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("X Position (m)")
axes[0].set_ylabel("Y Position (m)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

# 2. Plot Full LVMS Outline
csv_path_lvms = os.path.join(BASE, "deep_dynamics/data/LVMS_23_01_04_A.csv")
df_lvms = pd.read_csv(csv_path_lvms, comment="#", header=None)
x_l = df_lvms.iloc[:, 1].values
y_l = df_lvms.iloc[:, 2].values

axes[1].plot(x_l, y_l, color="#34495E", linewidth=3)
axes[1].scatter([x_l[0]], [y_l[0]], color="#E74C3C", s=150, edgecolors="k", label="Start Line")
axes[1].set_aspect("equal")
axes[1].set_title("Full Las Vegas Motor Speedway\n(Raw GPS Coordinates)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("X Position (m)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
out_path = os.path.join(BASE, "../output/full_gps_tracks.png")
plt.savefig(out_path, dpi=200)
print("Saved to", out_path)
