#!/usr/bin/env python
"""Feature selection analysis for V4 feature set.

1. Trains XGBoost on all data -> reports gain-based feature importance.
2. Computes pairwise Pearson correlation between features (identify redundancy).
3. Leave-One-Feature-Out LORO: removes one feature at a time and reports
   change in pooled macro-F1 vs V4 baseline.
4. Greedy backward elimination: iteratively removes the least-useful feature
   while pooled F1 keeps improving.

V4 features (10):
  csi, delta_hr, hr, hrv_rmssd, mic_rms, resp_amp, rr, rrv, sd2_sd1, ss

Usage (from project root):
    uv run python scripts/experiments/run_feature_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from src.pipeline import build_feature_table, SafeXGBClassifier
from src.local_eval import META_COLS
from src.exclusions import filter_feature_df

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]


# ---------------------------------------------------------------------------
# Helpers shared with previous ablation scripts
# ---------------------------------------------------------------------------

def _add_delta(df, col):
    df = df.copy()
    s = df.sort_values(["rec_name", "t_start"])
    s[f"delta_{col}"] = s.groupby("rec_name")[col].diff()
    df = df.merge(s[["rec_name", "t_start", f"delta_{col}"]],
                  on=["rec_name", "t_start"], how="left", suffixes=("", "_dup"))
    return df.loc[:, ~df.columns.str.endswith("_dup")]


def _metrics(y_true, y_pred, classes):
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=classes,
                                average="macro", zero_division=0)),
        per_class_f1={c: float(f1_score(y_true, y_pred, labels=[c],
                                        average="macro", zero_division=0))
                      for c in classes},
        confusion=confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    )


def _pooled_f1(folds, classes):
    cm = np.sum([np.asarray(f["confusion"]) for f in folds], axis=0).astype(float)
    f1s = []
    for i in range(len(classes)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0)
    return float(np.mean(f1s)), cm


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
    import os
    use_gpu = os.environ.get("BIESL_USE_GPU", "0") == "1"
    return SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        eval_metric="mlogloss", tree_method="hist",
        device=("cuda" if use_gpu else "cpu"),
    )


def loro_eval(df, feat_cols, classes=CLASSES):
    """LORO evaluation returning pooled F1, pooled acc, and per-fold list."""
    le = LabelEncoder().fit(classes)
    recs = sorted(df["rec_name"].unique())
    folds = []
    for r in recs:
        test_df  = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue
        train_df = _pad_missing(train_df, feat_cols, classes)
        clf = _make_xgb()
        clf.fit(train_df[feat_cols].to_numpy(float),
                le.transform(train_df["activity"]))
        pred = clf.predict(test_df[feat_cols].to_numpy(float))
        folds.append(_metrics(
            le.inverse_transform(le.transform(test_df["activity"])),
            le.inverse_transform(pred), classes))
    pf1, cm = _pooled_f1(folds, classes)
    pacc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return pf1, pacc, folds, cm


# ---------------------------------------------------------------------------
# 1. Build V4 dataset
# ---------------------------------------------------------------------------

print("=" * 70)
print(" Building feature table ...")
print("=" * 70)
df_raw = build_feature_table(include_temp=False)
df_raw = filter_feature_df(df_raw)
df_raw = df_raw[df_raw["activity"].isin(CLASSES)].copy()

df = _add_delta(df_raw, col="hr")
df = df.drop(columns=["samp_en"], errors="ignore")  # V4 = V3 + mic_rms

V4_FEATS = sorted(set(df.columns) - META_COLS)
print(f"V4 features ({len(V4_FEATS)}): {V4_FEATS}")

# ---------------------------------------------------------------------------
# 2. Baseline V4 LORO score
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Baseline V4 LORO (all 10 features)")
print("=" * 70)
base_f1, base_acc, base_folds, base_cm = loro_eval(df, V4_FEATS)
print(f"  Pooled F1 = {base_f1:.4f}   Pooled Acc = {base_acc:.4f}")

# ---------------------------------------------------------------------------
# 3. Feature importance from a single full-data XGBoost fit
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Feature importance (XGBoost 'gain', trained on all 4099 windows)")
print("=" * 70)
le_all = LabelEncoder().fit(CLASSES)
imp_model = _make_xgb()
X_all = df[V4_FEATS].to_numpy(float)
y_all = le_all.transform(df["activity"])

# Impute NaN before full-data fit (importance needs clean data)
imp_fill = SimpleImputer(strategy="median")
X_all_imp = imp_fill.fit_transform(X_all)
imp_model.fit(X_all_imp, y_all)

# XGBoost internal gain scores
gains = imp_model.clf_.get_booster().get_score(importance_type="gain")
importance = [(feat, gains.get(f"f{i}", 0.0)) for i, feat in enumerate(V4_FEATS)]
importance.sort(key=lambda x: x[1], reverse=True)

print(f"  {'feature':<14s}  {'gain':>10s}   {'rank':>4s}")
for rank, (feat, gain) in enumerate(importance, 1):
    print(f"  {feat:<14s}  {gain:>10.1f}   #{rank}")

# ---------------------------------------------------------------------------
# 4. Pairwise Pearson correlation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Pairwise Pearson correlation (top correlated pairs > 0.70)")
print("=" * 70)
X_df = df[V4_FEATS].copy()
X_df_imp = pd.DataFrame(imp_fill.transform(X_df), columns=V4_FEATS)
corr = X_df_imp.corr()
pairs = []
for i in range(len(V4_FEATS)):
    for j in range(i + 1, len(V4_FEATS)):
        r = corr.iloc[i, j]
        pairs.append((abs(r), V4_FEATS[i], V4_FEATS[j], r))
pairs.sort(reverse=True)
print(f"  {'feature A':<14s}  {'feature B':<14s}  {'|r|':>6s}  note")
for absr, a, b, r in pairs[:10]:
    note = "*** HIGH" if absr > 0.85 else ("** mod" if absr > 0.70 else "")
    print(f"  {a:<14s}  {b:<14s}  {absr:>6.3f}  {note}")

# ---------------------------------------------------------------------------
# 5. Leave-One-Feature-Out LORO
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Leave-One-Feature-Out  (impact of dropping each feature)")
print("=" * 70)
print(f"  Baseline pooled F1 = {base_f1:.4f}")
print(f"  {'removed feature':<16s}  {'pooled F1':>10s}  {'? F1':>8s}  verdict")

lofo_results = {}
for feat in V4_FEATS:
    remaining = [f for f in V4_FEATS if f != feat]
    pf1, pacc, _, _ = loro_eval(df, remaining)
    delta = pf1 - base_f1
    verdict = ("[remove]" if delta >= 0
               else ("[~ok]   " if delta >= -0.005
                     else "[keep]  "))
    print(f"  {feat:<16s}  {pf1:>10.4f}  {delta:>+8.4f}  {verdict}")
    lofo_results[feat] = {"pooled_f1": pf1, "delta": delta}

# Sort by impact
lofo_sorted = sorted(lofo_results.items(), key=lambda x: x[1]["delta"], reverse=True)
safe_to_remove = [f for f, v in lofo_sorted if v["delta"] >= -0.002]
print(f"\n  Features safe to remove (delta_F1 >= -0.002): {safe_to_remove}")

# ---------------------------------------------------------------------------
# 6. Greedy backward elimination
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Greedy backward elimination (remove worst feature each round)")
print("=" * 70)
current_feats = list(V4_FEATS)
current_f1 = base_f1
history = [(list(current_feats), current_f1)]

while len(current_feats) > 1:
    best_delta = -999
    best_feat_to_drop = None
    for feat in current_feats:
        remaining = [f for f in current_feats if f != feat]
        pf1, _, _, _ = loro_eval(df, remaining)
        delta = pf1 - current_f1
        if delta > best_delta:
            best_delta = delta
            best_feat_to_drop = feat
            best_f1_after = pf1

    if best_delta < -0.010:   # stop if removing anything costs > 1% F1
        print(f"  Stopping: removing any feature costs > 0.010 F1")
        break

    print(f"  Remove '{best_feat_to_drop}':  "
          f"{current_f1:.4f} -> {best_f1_after:.4f}  (?={best_delta:+.4f})")
    current_feats.remove(best_feat_to_drop)
    current_f1 = best_f1_after
    history.append((list(current_feats), current_f1))

print(f"\n  Final feature set ({len(current_feats)}): {sorted(current_feats)}")
print(f"  Final pooled F1  : {current_f1:.4f}  (baseline: {base_f1:.4f})")

# Run final LORO with best reduced set and print per-class
if len(current_feats) < len(V4_FEATS):
    print("\n  Per-class F1 with reduced feature set:")
    fin_f1, fin_acc, fin_folds, fin_cm = loro_eval(df, current_feats)
    fin_cm_f = fin_cm.astype(float)
    for i, c in enumerate(CLASSES):
        tp = fin_cm_f[i, i]; fp = fin_cm_f[:, i].sum() - tp; fn = fin_cm_f[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1c = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        print(f"    {c:<12s}  F1={f1c:.3f}  prec={pr:.3f}  rec={rc:.3f}")
    print(f"\n  Pooled F1={fin_f1:.4f}   Pooled Acc={fin_acc:.4f}")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
Path("outputs").mkdir(exist_ok=True)
out = {
    "baseline_v4": {"pooled_f1": base_f1, "pooled_acc": base_acc,
                    "features": V4_FEATS},
    "feature_importance": {f: g for f, g in importance},
    "lofo": lofo_results,
    "greedy_elimination": {
        "history": [(feats, f1) for feats, f1 in history],
        "final_features": sorted(current_feats),
        "final_pooled_f1": current_f1,
    },
}
with open("outputs/feature_selection.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\n  -> wrote outputs/feature_selection.json")

