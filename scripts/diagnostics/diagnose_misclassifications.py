#!/usr/bin/env python
"""Diagnose misclassifications per recording under LORO cross-validation.

Usage:
    python scripts/diagnostics/diagnose_misclassifications.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.exclusions import filter_feature_df
from src.pipeline import build_feature_table, SafeXGBClassifier

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]
FINAL_FEATS = ["csi", "hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "sd2_sd1", "ss"]


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


def main():
    print("Building feature table...")
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    le = LabelEncoder().fit(CLASSES)
    recs = sorted(df["rec_name"].unique())

    diagnostics = []

    print("\nRunning Leave-One-Recording-Out CV and tracking misclassifications...")
    for r in recs:
        test_df = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue

        train_df = _pad_missing(train_df, FINAL_FEATS, CLASSES)
        clf = SafeXGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=0,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cpu",
        )
        clf.fit(train_df[FINAL_FEATS].to_numpy(float), le.transform(train_df["activity"]))
        pred_encoded = clf.predict(test_df[FINAL_FEATS].to_numpy(float))
        y_pred = le.inverse_transform(pred_encoded)

        y_true = test_df["activity"].to_numpy()

        correct = (y_true == y_pred).sum()
        total = len(y_true)
        wrong = total - correct
        error_rate = wrong / total if total > 0 else 0.0

        # Find specific errors
        errors = []
        for t, p in zip(y_true, y_pred):
            if t != p:
                errors.append(f"{t} -> {p}")
        error_counts = Counter(errors)
        error_summary = ", ".join(f"{k}: {v}" for k, v in error_counts.most_common(3))

        diagnostics.append({
            "Recording": r,
            "Total": total,
            "Wrong": wrong,
            "Error Rate": error_rate,
            "Error Details": error_summary,
        })

    diag_df = pd.DataFrame(diagnostics).sort_values("Error Rate", ascending=False)
    
    print("\n" + "=" * 90)
    print(" LORO Misclassification Diagnosis (Sorted by Error Rate)")
    print("=" * 90)
    print(diag_df.to_string(index=False, formatters={
        "Error Rate": lambda x: f"{x:.1%}"
    }))
    print("=" * 90)


if __name__ == "__main__":
    main()

