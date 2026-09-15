#!/usr/bin/env python
"""LORO experiment for Plan C ONLY (12 features with baseline delta-normalisation).

Usage (from project root):
    python scripts/evaluation/run_plan_c.py
"""
from __future__ import annotations

import argparse
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
FEATS_BASELINE = ["hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "ss"]
FEATS_PLAN_C   = FEATS_BASELINE + ["delta_hr", "delta_hrv_rmssd", "delta_ss", "delta_rr", "delta_resp_amp"]


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
        class_weight="balanced",
    )


def run_loro_plan_c(df):
    """Run Leave-One-Recording-Out CV for Plan C."""
    feat_cols = FEATS_PLAN_C
    le = LabelEncoder().fit(CLASSES)
    recs = sorted(df["rec_name"].unique())

    y_true_all, y_pred_all = [], []
    fold_results = []

    print(f"Running Plan C LORO across {len(recs)} recordings...")
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

        phase2_to_rest = sum((t != "rest") & (p == "rest") for t, p in zip(y_true, y_pred))
        n_phase2 = sum(t != "rest" for t in y_true)

        fold_results.append({
            "rec": r, "acc": acc, "f1m": f1m,
            "phase2_to_rest": phase2_to_rest, "n_phase2": n_phase2,
        })
        print(f"  [Plan C] {r:35s} acc={acc:.3f} f1={f1m:.3f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASSES)
    pooled_acc = accuracy_score(y_true_all, y_pred_all)
    pooled_f1 = f1_score(y_true_all, y_pred_all, labels=CLASSES, average="macro", zero_division=0)
    per_f1 = f1_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)
    per_pre = precision_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)
    per_rec = recall_score(y_true_all, y_pred_all, labels=CLASSES, average=None, zero_division=0)

    res = {
        "label": "Plan C (12 features: restored delta_SS and RRV, no CSI/SD2_SD1)",
        "features": feat_cols,
        "n_features": len(feat_cols),
        "pooled_acc": float(pooled_acc),
        "pooled_f1": float(pooled_f1),
        "per_class_f1": {c: float(f) for c, f in zip(CLASSES, per_f1)},
        "per_class_precision": {c: float(p) for c, p in zip(CLASSES, per_pre)},
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, per_rec)},
        "confusion": cm.tolist(),
        "folds": fold_results,
    }

    return res, y_true_all, y_pred_all


def plot_plan_c_cm(res, out_path):
    """Plot detailed confusion matrix and metrics for Plan C."""
    fig = plt.figure(figsize=(12, 6), facecolor="white")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.35)

    class_labels = [c.capitalize() for c in CLASSES]
    cm = np.array(res["confusion"])
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = cm.astype(float) / np.where(row_sums == 0, 1, row_sums) * 100.0

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if pct[i, j] > 50 else "black"
            ax0.text(j, i, f"{cm[i,j]}\n({pct[i,j]:.1f}%)",
                     ha="center", va="center", fontsize=9,
                     fontweight="bold" if i == j else "normal", color=color)

    ax0.set_xticks(range(len(CLASSES)))
    ax0.set_yticks(range(len(CLASSES)))
    ax0.set_xticklabels(class_labels, fontsize=10)
    ax0.set_yticklabels(class_labels, fontsize=10)
    ax0.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax0.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax0.set_title(f"Plan C Confusion Matrix\nAccuracy: {res['pooled_acc']:.1%} | Macro F1: {res['pooled_f1']:.3f}",
                  fontsize=12, fontweight="bold", pad=10)

    # Bar chart per-class metrics
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(len(CLASSES))
    w = 0.25
    f1s = [res["per_class_f1"][c] for c in CLASSES]
    pres = [res["per_class_precision"][c] for c in CLASSES]
    recs = [res["per_class_recall"][c] for c in CLASSES]

    bars1 = ax1.bar(x - w, f1s, w, label="F1", color="#e94560", alpha=0.85)
    bars2 = ax1.bar(x, pres, w, label="Precision", color="#0f3460", alpha=0.85)
    bars3 = ax1.bar(x + w, recs, w, label="Recall", color="#533483", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_title("Plan C Per-Class Metrics", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def main(out_dir: str = "outputs"):
    t0 = time.time()
    print("Building feature table with updated dataset...")
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    print("Adding baseline delta features...")
    df = add_delta_features(df)

    res, y_true_all, y_pred_all = run_loro_plan_c(df)

    print("\n" + "=" * 80)
    print("  PLAN C (UPDATED DATASET) RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Windows : {len(df)}")
    print(f"Pooled Acc    : {res['pooled_acc']:.1%}")
    print(f"Pooled MacroF1: {res['pooled_f1']:.3f}")
    print("\nPer-class Performance:")
    for c in CLASSES:
        print(f"  {c:12s} -> F1: {res['per_class_f1'][c]:.3f} | Precision: {res['per_class_precision'][c]:.3f} | Recall: {res['per_class_recall'][c]:.3f}")

    print("\nConfusion Matrix:")
    print("Labels:", CLASSES)
    print(np.array(res["confusion"]))

    # Save JSON
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "plan_c_updated_results.json")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nWrote results -> {out_json}")

    # Plot
    plot_path = os.path.join(out_dir, "plan_c_updated_cm.png")
    plot_plan_c_cm(res, plot_path)

    print(f"Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for the evaluation JSON and confusion-matrix PNG.",
    )
    args = parser.parse_args()
    main(out_dir=args.out_dir)

