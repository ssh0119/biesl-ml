#!/usr/bin/env python
"""Train and save the final optimal 12-feature Plan C XGBoost model on all data.

Features (12):
  hr, hrv_rmssd, mic_rms, resp_amp, rr, rrv, ss,
  delta_hr, delta_hrv_rmssd, delta_ss, delta_rr, delta_resp_amp

Usage:
    python scripts/training/train_final_xgb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.exclusions import filter_feature_df
from src.pipeline import build_feature_table, add_delta_features, SafeXGBClassifier

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]
FINAL_FEATS = [
    "hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "ss",
    "delta_hr", "delta_hrv_rmssd", "delta_ss", "delta_rr", "delta_resp_amp",
]
OUT_DIR = Path("outputs")


def main():
    print("=" * 70)
    print(" Building feature table (all recordings in data/)...")
    print("=" * 70)
    df_raw = build_feature_table(include_temp=False)
    
    print("\nApplying partial exclusions...")
    df_raw = filter_feature_df(df_raw)
    
    print(f"Filtering to target classes: {CLASSES}")
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    print("Adding baseline delta features...")
    df = add_delta_features(df)
    
    print(f"Dataset shape: {df.shape[0]} windows")
    print("Window count per class:")
    for c in CLASSES:
        count = (df["activity"] == c).sum()
        print(f"  {c:<12s}: {count:4d}")
        
    print(f"\nSelecting 12 final features:\n  {FINAL_FEATS}")
    X = df[FINAL_FEATS].to_numpy(float)
    y = df["activity"].to_numpy()
    
    # Initialize SafeXGBClassifier with optimal hyperparameters
    print("\nTraining final XGBoost model...")
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
        class_weight="balanced",
    )
    
    # Fit the model on all data
    clf.fit(X, y)
    
    # Save the pipeline/model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Joblib serialization (for Python pipeline inference)
    model_path = OUT_DIR / "xgboost_model.joblib"
    joblib.dump(clf, model_path)
    print(f"Saved joblib model to -> {model_path}")
    
    # 2. Native XGBoost Booster model (for continuation/fine-tuning in future training)
    native_xgb_path = OUT_DIR / "xgboost_model.json"
    clf.clf_.save_model(str(native_xgb_path))
    print(f"Saved native XGBoost model to -> {native_xgb_path}")
    
    # 3. Save feature names and classes metadata
    meta_path = OUT_DIR / "feature_names.joblib"
    joblib.dump({"feature_names": FINAL_FEATS, "classes": CLASSES}, meta_path)
    print(f"Saved feature metadata to -> {meta_path}")

    # 4. Save metadata JSON for readability
    meta_json = OUT_DIR / "model_metadata.json"
    with open(meta_json, "w") as fh:
        json.dump({
            "model_type": "SafeXGBClassifier",
            "n_features": len(FINAL_FEATS),
            "features": FINAL_FEATS,
            "classes": CLASSES,
            "n_samples": len(df),
            "hyperparameters": {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "class_weight": "balanced",
            },
        }, fh, indent=2)
    print(f"Saved metadata JSON to -> {meta_json}")
    
    print("\nTraining & Saving complete!")


if __name__ == "__main__":
    main()


