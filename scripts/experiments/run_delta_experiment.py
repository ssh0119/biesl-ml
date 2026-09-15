#!/usr/bin/env python
"""LORO experiment: compare baseline 9-feature vs delta-normalised + samp_en.

Runs four configurations side-by-side:
  A) Baseline:    9 original features (current production model)
  B) + samp_en:   10 features (add Sample Entropy)
  C) + deltas:    12 features (9 + delta_hr, delta_hrv_rmssd, delta_ss)
  D) Combined:    13 features (9 + samp_en + 3 deltas)

Usage (from project root):
    python scripts/experiments/run_delta_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder

from src.pipeline import build_feature_table, add_delta_features, SafeXGBClassifier
from src.exclusions import filter_feature_df

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]

# Feature sets for each experiment variant
FEATS_BASELINE = ["csi", "hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "sd2_sd1", "ss"]
FEATS_SAMPEN   = FEATS_BASELINE + ["samp_en"]
FEATS_DELTA    = FEATS_BASELINE + ["delta_hr", "delta_hrv_rmssd", "delta_ss"]
FEATS_COMBINED = FEATS_BASELINE + ["samp_en", "delta_hr", "delta_hrv_rmssd", "delta_ss"]

VARIANTS = {
    "A_baseline":  FEATS_BASELINE,
    "B_samp_en":   FEATS_SAMPEN,
    "C_deltas":    FEATS_DELTA,
    "D_combined":  FEATS_COMBINED,
}


def _pad_missing(train_df, feat_cols, classes):
    missing = set(classes) - set(train_df["activity"].unique())
    if not missing:
        return train_df
    pad = pd.DataFrame([{**{c: np.nan for c in feat_cols},
                         "rec_name": f"_pad_{m}", "subject": "_pad",
                         "stressor": "_pad", "phase": m, "activity": m,
                         "t_start": 0.0, "t_end": 0.0}
                        for m in missing])
    return pd.concat([train_df, pad], ignore_index=True)


def _make_xgb():
    return SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        eval_metric="mlogloss", tree_method="hist", device="cpu",
    )


def run_loro(df, feat_cols, label="variant"):
    """Run Leave-One-Recording-Out CV and return results dict."""
    le = LabelEncoder().fit(CLASSES)
    recs = sorted(df["rec_name"].unique())

    y_true_all, y_pred_all = [], []
    fold_results = []

    for r in recs:
        test_df = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue

        train_df = _pad_missing(train_df, feat_cols, CLASSES)
        clf = _make_xgb()
        clf.fit(train_df[feat_cols].to_numpy(float), le.transform(train_df["activity"]))
        pred = clf.predict(test_df[feat_cols].to_numpy(float))
        y_true = le.inverse_transform(le.transform(test_df["activity"]))
        y_pred = le.inverse_transform(pred)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)

        # Track math->rest for the worst fold
        math_to_rest = sum((t == "math") & (p == "rest") for t, p in zip(y_true, y_pred))
        n_math = sum(y_true == "math") if hasattr(y_true, "__len__") else sum(t == "math" for t in y_true)

        fold_results.append({
            "rec": r, "acc": acc, "f1m": f1m,
            "math_to_rest": math_to_rest, "n_math": n_math,
        })
        print(f"  [{label:12s}] {r:35s} acc={acc:.3f} f1={f1m:.3f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASSES)
    pooled_acc = accuracy_score(y_true_all, y_pred_all)
    pooled_f1 = f1_score(y_true_all, y_pred_all, labels=CLASSES, average="macro", zero_division=0)
    per_f1 = f1_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)
    per_pre = precision_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)
    per_rec = recall_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)

    return {
        "label": label,
        "features": feat_cols,
        "n_features": len(feat_cols),
        "pooled_acc": pooled_acc,
        "pooled_f1": pooled_f1,
        "per_class_f1": {c: float(f) for c, f in zip(CLASSES, per_f1)},
        "per_class_precision": {c: float(p) for c, p in zip(CLASSES, per_pre)},
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, per_rec)},
        "confusion": cm.tolist(),
        "folds": fold_results,
    }


def plot_comparison(results: dict, out_path: str):
    """Plot side-by-side confusion matrices and a metrics comparison bar chart."""
    n_variants = len(results)
    fig = plt.figure(figsize=(6 * n_variants + 4, 7), facecolor="white")

    gs = gridspec.GridSpec(1, n_variants + 1, figure=fig,
                           width_ratios=[1] * n_variants + [0.8], wspace=0.35)

    class_labels = [c.capitalize() for c in CLASSES]

    for vi, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[vi])
        cm = np.array(res["confusion"])

        # Row-normalise for consistent coloring
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = cm.astype(float) / np.where(row_sums == 0, 1, row_sums) * 100.0

        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                color = "white" if pct[i, j] > 50 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({pct[i,j]:.0f}%)",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold" if i == j else "normal", color=color)

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(class_labels, fontsize=8, rotation=25, ha="right")
        ax.set_yticklabels(class_labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        if vi == 0:
            ax.set_ylabel("True Label", fontsize=9)
        ax.set_title(
            f"{name}\n{res['n_features']} feats | Acc={res['pooled_acc']:.1%} | F1={res['pooled_f1']:.3f}",
            fontsize=9, fontweight="bold", pad=8,
        )
        ax.tick_params(length=0)

    # Right panel: per-class F1 comparison
    ax_bar = fig.add_subplot(gs[n_variants])
    x = np.arange(len(CLASSES))
    width = 0.8 / n_variants
    colors = ["#e94560", "#0f3460", "#533483", "#16c79a"]

    for vi, (name, res) in enumerate(results.items()):
        f1s = [res["per_class_f1"][c] for c in CLASSES]
        bars = ax_bar.bar(x + vi * width - 0.4 + width / 2, f1s, width,
                          label=name, color=colors[vi % len(colors)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=5.5,
                        fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(class_labels, fontsize=8, rotation=20, ha="right")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("F1 Score", fontsize=9)
    ax_bar.set_title("Per-Class F1 Comparison", fontsize=9, fontweight="bold", pad=8)
    ax_bar.legend(fontsize=6, loc="lower right")
    ax_bar.tick_params(length=0)

    fig.suptitle("LORO XGBoost: Baseline vs Delta-Normalised Features",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


def main():
    t_start = time.time()

    # Build feature table once
    print("Building feature table...")
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    # Add delta features
    print("Adding delta features (baseline normalisation)...")
    df = add_delta_features(df)

    print(f"\nDataset: {len(df)} windows, columns: {sorted(df.columns.tolist())}")
    print(f"Delta feature columns present: {[c for c in df.columns if c.startswith('delta_')]}")

    # Run all variants
    all_results = {}
    for name, feats in VARIANTS.items():
        print(f"\n{'='*80}")
        print(f"  VARIANT: {name}  ({len(feats)} features)")
        print(f"  Features: {feats}")
        print(f"{'='*80}")
        all_results[name] = run_loro(df, feats, label=name)

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Variant':>14s} | {'#Feats':>6s} | {'Acc':>6s} | {'MacroF1':>7s} | "
          f"{'Rest F1':>7s} | {'Medi F1':>7s} | {'Plank F1':>8s} | {'Math F1':>7s}")
    print("-" * 90)
    for name, res in all_results.items():
        pf = res["per_class_f1"]
        print(f"{name:>14s} | {res['n_features']:>6d} | {res['pooled_acc']:>6.1%} | "
              f"{res['pooled_f1']:>7.3f} | {pf['rest']:>7.3f} | {pf['meditation']:>7.3f} | "
              f"{pf['plank']:>8.3f} | {pf['math']:>7.3f}")

    # nvt_5_26_math_7_10 fold comparison (worst case)
    print(f"\n--- nvt_5_26_math_7_10 fold (worst case) ---")
    for name, res in all_results.items():
        for f in res["folds"]:
            if "nvt_5_26_math" in f["rec"]:
                print(f"  {name:>14s}: acc={f['acc']:.3f} f1={f['f1m']:.3f} "
                      f"math->rest={f['math_to_rest']}/{f['n_math']}")

    # Save results
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "delta_experiment_results.json")
    # Convert numpy types for JSON serialisation
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(out_json, "w") as fh:
        json.dump(_clean(all_results), fh, indent=2)
    print(f"\nSaved results -> {out_json}")

    # Plot comparison
    plot_path = os.path.join("figures", "ablation", "delta_experiment_cm.png")
    plot_comparison(all_results, plot_path)

    print(f"\nTotal time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

