#!/usr/bin/env python3
"""Plot per-sample residuals (predicted - ground truth) for a trained model."""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from deep_dynamics.model.models import string_to_model, string_to_dataset

device = torch.device("cpu")


def collect_residuals(model, data_loader):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, norm_inputs in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            norm_inputs = norm_inputs.to(device)
            if model.is_rnn:
                h = model.init_hidden(inputs.shape[0])
                h = h.data
                output, h, _ = model(inputs, norm_inputs, h)
            else:
                output, _, _ = model(inputs, norm_inputs)
            all_preds.append(output.squeeze().cpu().numpy())
            all_labels.append(labels.squeeze().cpu().numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    return preds, labels


def plot_residuals(preds, labels, save_path=None):
    residuals = preds - labels
    state_names = ["vx (m/s)", "vy (m/s)", "yaw rate (rad/s)"]
    n_samples = np.arange(len(residuals))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Prediction Residuals (predicted − ground truth)", fontsize=14)

    for i in range(3):
        r = residuals[:, i]

        ax_ts = axes[i, 0]
        ax_ts.plot(n_samples, r, linewidth=0.4, alpha=0.7)
        ax_ts.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax_ts.set_ylabel(state_names[i])
        ax_ts.set_title(f"{state_names[i]}  —  time series")
        if i == 2:
            ax_ts.set_xlabel("sample index")

        ax_hist = axes[i, 1]
        ax_hist.hist(r, bins=80, edgecolor="black", linewidth=0.3, alpha=0.75)
        ax_hist.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax_hist.set_title(
            f"{state_names[i]}  —  μ={np.mean(r):.4e}  σ={np.std(r):.4e}"
        )
        if i == 2:
            ax_hist.set_xlabel("residual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot residuals for a trained model.")
    parser.add_argument("model_cfg", type=str, help="YAML config file")
    parser.add_argument("dataset_file", type=str, help=".npz dataset")
    parser.add_argument("model_state_dict", type=str, help="Checkpoint .pth file")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    args = parser.parse_args()

    with open(args.model_cfg, "rb") as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)

    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
    model.to(device)
    model.load_state_dict(torch.load(args.model_state_dict, map_location=device))

    data_npy = np.load(args.dataset_file)
    with open(os.path.join(os.path.dirname(args.model_state_dict), "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](
        data_npy["features"], data_npy["labels"], scaler
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    preds, labels = collect_residuals(model, loader)
    plot_residuals(preds, labels, save_path=args.save)
