#!/usr/bin/env python
"""Compute SHAP values and generate a stacked horizontal bar chart
visualising the contribution of each physiological parameter,
arranged in descending order of significance.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline import build_feature_table, SafeXGBClassifier
from src.exclusions import filter_feature_df

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]
FINAL_FEATS = ["csi", "hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "sd2_sd1", "ss"]

# Display name mapping to make the plot clean and standard
FEATURE_DISPLAY_NAMES = {
    "csi": "CSI",
    "hr": "HR",
    "hrv_rmssd": "HRV",
    "mic_rms": "MIC",
    "resp_amp": "AMP",
    "rr": "RR",
    "rrv": "RRV",
    "sd2_sd1": "SD2/SD1",
    "ss": "SS",
}


def main():
    print("Loading model and data...")
    # Load model and data
    clf_safe: SafeXGBClassifier = joblib.load("outputs/xgboost_model.joblib")
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    X = df[FINAL_FEATS].to_numpy(float)
    
    # Calculate SHAP values in probability space
    print("Computing SHAP values in probability space...")
    np.random.seed(0)
    bg = shap.sample(X, 50)
    X_explain = shap.sample(X, 500)
    
    explainer = shap.Explainer(clf_safe.predict_proba, bg)
    shap_vals = explainer(X_explain)
    
    classes = list(clf_safe.classes_)
    print(f"Classes order in model: {classes}")
    
    # Compute mean absolute SHAP values for each class
    # shap_vals.values shape: (n_samples, n_features, n_classes)
    class_indices = {c: classes.index(c) for c in CLASSES}
    mean_abs_vals = {}
    
    for c in CLASSES:
        idx = class_indices[c]
        mean_abs_vals[c] = np.abs(shap_vals.values[:, :, idx]).mean(axis=0)


    # Build DataFrame
    shap_df = pd.DataFrame({"feature": FINAL_FEATS})
    for c in CLASSES:
        shap_df[c] = mean_abs_vals[c]
    shap_df["total"] = shap_df[CLASSES].sum(axis=1)
    
    # Sort in ascending order of total significance
    shap_df = shap_df.sort_values("total", ascending=True) # Ascending for plotting bottom-up in horizontal bar
    
    # Map feature names
    shap_df["display_name"] = shap_df["feature"].map(FEATURE_DISPLAY_NAMES)
    
    print("\nCalculated Mean Shapley Values:")
    print(shap_df.sort_values("total", ascending=False).to_string(index=False))
    
    # Create the Plot
    # Example style: clean white background, specific colors, legend at the top
    fig, ax = plt.subplots(figsize=(5.5, 6.5), facecolor="white")
    ax.set_facecolor("white")
    
    # Curated premium color palette
    colors = {
        "rest": "#4ea8de",        # sky blue
        "meditation": "#52b788",  # sage green
        "plank": "#f4a261",       # warm orange
        "math": "#7209b7"         # deep purple
    }
    
    y = np.arange(len(shap_df))
    
    # Stacked horizontal bar chart
    left = np.zeros(len(shap_df))
    for c in CLASSES:
        ax.barh(
            y, 
            shap_df[c], 
            left=left, 
            color=colors[c], 
            label=c.capitalize(), 
            edgecolor="white", 
            linewidth=0.5
        )
        left += shap_df[c]
    
    # Styling
    ax.set_yticks(y)
    ax.set_yticklabels(shap_df["display_name"], color="black", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Shapley value", color="black", fontsize=11, labelpad=8)
    
    # Grid lines
    ax.xaxis.grid(True, linestyle="-", color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    
    # Hide top and right spines, color left and bottom
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#cccccc")
        
    ax.tick_params(colors="black", labelsize=10)
    
    # Legend at top (4 columns for the 4 classes)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        labelcolor="black"
    )
    
    # Faint Title / Letter 'G' in corner
    # Add 'G' in top-left above the plot
    fig.text(0.05, 0.95, "G", fontsize=18, fontweight="bold", color="black", va="top")
    
    plt.tight_layout()
    
    # Save figure
    out_dir = Path("figures/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shapley_importance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
if __name__ == "__main__":
    main()

