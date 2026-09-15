#!/usr/bin/env python
"""Analyze features and classifications for mta_6_3_math_8 under LORO CV.
"""
from __future__ import annotations

import os
import sys
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
    # 1. Load data
    print("Loading feature table...")
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()

    # 2. Split train/test (LORO: test on mta_6_3_math_8)
    rec_target = "mta_6_3_math_8"
    test_df = df[df.rec_name == rec_target].copy()
    train_df = df[df.rec_name != rec_target].copy()

    if len(test_df) == 0:
        print(f"Error: Target recording {rec_target} not found in the dataset.")
        return

    train_df = _pad_missing(train_df, FINAL_FEATS, CLASSES)

    # 3. Fit model
    le = LabelEncoder().fit(CLASSES)
    X_train = train_df[FINAL_FEATS].to_numpy(float)
    y_train = le.transform(train_df["activity"])

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
    clf.fit(X_train, y_train)

    # 4. Predict
    X_test = test_df[FINAL_FEATS].to_numpy(float)
    pred_encoded = clf.predict(X_test)
    test_df["pred"] = le.inverse_transform(pred_encoded)
    test_df["correct"] = test_df["activity"] == test_df["pred"]

    print("\n" + "=" * 80)
    print(f" Detailed Analysis for {rec_target}")
    print("=" * 80)
    print(f"Total windows: {len(test_df)}")
    print(f"Correct: {test_df['correct'].sum()}, Incorrect: {(~test_df['correct']).sum()}")
    print(f"Accuracy: {test_df['correct'].mean():.2%}")

    print("\nClassification breakdown by true class:")
    for c in CLASSES:
        sub = test_df[test_df["activity"] == c]
        if len(sub) == 0:
            continue
        corr = sub["correct"].sum()
        tot = len(sub)
        print(f"  {c:<12s}: {corr}/{tot} correct ({corr/tot:.1%})")
        # count predictions
        preds = sub["pred"].value_counts()
        for p, count in preds.items():
            if p != c:
                print(f"    -> predicted as {p}: {count}")

    # 5. Print training set mean values vs target recording class mean values
    print("\n" + "=" * 80)
    print(" Feature Comparison: Training Means vs. mta_6_3_math_8 True Class Means")
    print("=" * 80)
    
    # Let's get means for training set (restricted to actual target classes in train, excluding pads)
    train_clean = train_df[~train_df["rec_name"].str.startswith("_pad")].copy()
    
    # We will build a table of: Feature | Train Rest | Target Rest | Train Math | Target Math
    comparison_data = []
    for f in FINAL_FEATS:
        row = {"Feature": f}
        for c in ["rest", "math"]:
            tr_mean = train_clean[train_clean["activity"] == c][f].mean()
            te_mean = test_df[test_df["activity"] == c][f].mean()
            row[f"Train {c.capitalize()}"] = f"{tr_mean:.3f}"
            row[f"Target {c.capitalize()}"] = f"{te_mean:.3f}"
        comparison_data.append(row)
    
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))

    # 6. Show examples of misclassified windows in math
    print("\n" + "=" * 80)
    print(" Sample of Misclassified 'math' Windows in mta_6_3_math_8")
    print("=" * 80)
    
    math_wrong = test_df[(test_df["activity"] == "math") & (~test_df["correct"])]
    if len(math_wrong) > 0:
        # Sort by t_start to see timeline
        math_wrong = math_wrong.sort_values("t_start")
        # Print top 15 errors
        cols_to_print = ["t_start", "pred"] + FINAL_FEATS
        print(math_wrong[cols_to_print].head(15).to_string(index=False, formatters={
            "t_start": lambda x: f"{x:.1f}s",
            **{f: (lambda x: f"{x:.2f}") for f in FINAL_FEATS}
        }))
    else:
        print("No misclassified math windows.")

    # 7. Show overall summary of math predictions
    print("\nSummary of math predictions over time:")
    math_all = test_df[test_df["activity"] == "math"].sort_values("t_start")
    print(f"Math phase starts at {math_all['t_start'].min():.1f}s and ends at {math_all['t_end'].max():.1f}s")
    
    # Let's print a chronological sequence of predictions for the math phase (in chunks of 10 windows)
    chunks = []
    chunk_size = 12
    for i in range(0, len(math_all), chunk_size):
        chunk = math_all.iloc[i:i+chunk_size]
        start_t = chunk["t_start"].min()
        end_t = chunk["t_end"].max()
        preds = chunk["pred"].tolist()
        pred_summary = "".join(["R" if p == "rest" else "M" if p == "meditation" else "P" if p == "plank" else "X" for p in preds])
        chunks.append(f"  {start_t:5.1f}s - {end_t:5.1f}s: {pred_summary}  (R=Rest, M=Medi, P=Plank, X=Math)")
    print("\n".join(chunks))


if __name__ == "__main__":
    main()

