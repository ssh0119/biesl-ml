#!/usr/bin/env python
"""Compute SHAP values and generate a stacked horizontal bar chart
for the 14-feature Plan C XGBoost model (9 baseline + 5 delta features).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline import build_feature_table, add_delta_features, SafeXGBClassifier
from src.exclusions import filter_feature_df

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]
FEATS_BASELINE = ["hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "ss"]
FEATS_PLAN_C   = FEATS_BASELINE + ["delta_hr", "delta_hrv_rmssd", "delta_ss", "delta_rr", "delta_resp_amp"]

FEATURE_DISPLAY_NAMES = {
    "hr": "HR",
    "hrv_rmssd": "HRV-RMSSD",
    "mic_rms": "MIC-RMS",
    "resp_amp": "Resp Amp",
    "rr": "RR",
    "rrv": "RRV",
    "ss": "SS",
    "delta_hr": "?HR",
    "delta_hrv_rmssd": "?HRV-RMSSD",
    "delta_ss": "?SS",
    "delta_rr": "?RR",
    "delta_resp_amp": "?Resp Amp",
}


def main():
    print("Building feature table...")
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    print("Adding baseline delta features...")
    df = add_delta_features(df)

    X = df[FEATS_PLAN_C].to_numpy(float)
    y = df["activity"].to_numpy()

    # Train 14-feature XGBoost with balanced weighting on ALL data
    print("Training 14-feature XGBoost for SHAP analysis...")
    clf = SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        eval_metric="mlogloss", tree_method="hist", device="cpu",
        class_weight="balanced",
    )
    clf.fit(X, y)

    # Compute SHAP values in probability space
    print("Computing SHAP values (this may take a few minutes)...")
    np.random.seed(0)
    bg = shap.sample(X, 50)
    X_explain = shap.sample(X, 500)

    explainer = shap.Explainer(clf.predict_proba, bg)
    shap_vals = explainer(X_explain)

    classes = list(clf.classes_)
    print(f"Classes order in model: {classes}")

    # Compute mean absolute SHAP values per class
    class_indices = {c: classes.index(c) for c in CLASSES}
    mean_abs_vals = {}
    for c in CLASSES:
        idx = class_indices[c]
        mean_abs_vals[c] = np.abs(shap_vals.values[:, :, idx]).mean(axis=0)

    # Build DataFrame
    shap_df = pd.DataFrame({"feature": FEATS_PLAN_C})
    for c in CLASSES:
        shap_df[c] = mean_abs_vals[c]
    shap_df["total"] = shap_df[CLASSES].sum(axis=1)
    shap_df = shap_df.sort_values("total", ascending=True)
    shap_df["display_name"] = shap_df["feature"].map(FEATURE_DISPLAY_NAMES)

    print("\nMean |SHAP| values per feature (descending):")
    print(shap_df.sort_values("total", ascending=False).to_string(index=False))

    # ---- Plot: Stacked horizontal bar chart ----
    fig, ax = plt.subplots(figsize=(7, 8), facecolor="white")
    ax.set_facecolor("white")

    colors = {
        "rest":       "#4ea8de",
        "meditation": "#52b788",
        "plank":      "#f4a261",
        "math":       "#7209b7",
    }

    y_pos = np.arange(len(shap_df))
    left = np.zeros(len(shap_df))
    for c in CLASSES:
        ax.barh(
            y_pos, shap_df[c], left=left,
            color=colors[c], label=c.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        left += shap_df[c].values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_df["display_name"], color="black", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", color="black", fontsize=12, labelpad=8)
    ax.set_title("Feature Importance ??12-Feature Plan C XGBoost\n(SHAP probability-space, class-stacked)",
                 fontsize=13, fontweight="bold", color="black", pad=12)

    ax.xaxis.grid(True, linestyle="-", color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#cccccc")
    ax.tick_params(colors="black", labelsize=10)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=4, frameon=False, fontsize=10,
        handlelength=1.2, labelcolor="black",
    )

    plt.tight_layout()

    out_dir = Path("figures/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shapley_12feat_plan_c.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved SHAP plot -> {out_path}")


if __name__ == "__main__":
    main()

