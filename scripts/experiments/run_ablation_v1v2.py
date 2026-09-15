#!/usr/bin/env python
"""Two XGBoost ablation experiments on top of the baseline 8-feature set.

VERSION 1  ??Lagged Sample Entropy
  Features: the 8 baseline features PLUS
      samp_en_lag1  (samp_en one window earlier, i.e. 2 s ago)
      samp_en_lag2  (samp_en two windows earlier, i.e. 4 s ago)
  Lags are computed within each recording, ordered by t_start.
  Windows with no predecessor get NaN (the XGBoost imputer handles NaN natively).

VERSION 2  ??Delta HR instead of Sample Entropy
  Features: the 8 baseline features with samp_en REMOVED and replaced by
      delta_hr  = hr_t  - hr_{t-1}  (change in heart rate vs previous window)
  Computed within each recording; first window of each recording gets NaN.

Both versions run LORO (Leave-One-Recording-Out) XGBoost evaluation,
emit row-normalized confusion matrix PNGs, and write a JSON results file.

Usage (from project root):
    uv run python scripts/experiments/run_ablation_v1v2.py
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

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from src.pipeline import build_feature_table
from src.local_eval import META_COLS
from src.exclusions import filter_feature_df
from src.pipeline import SafeXGBClassifier

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]


# ---------------------------------------------------------------------------
# Helper: add temporal lag / delta features PER RECORDING
# ---------------------------------------------------------------------------

def _add_lags(df: pd.DataFrame, col: str, n_lags: int = 2) -> pd.DataFrame:
    """Add `col_lag1`, `col_lag2`, ??ordered by t_start within each recording."""
    df = df.copy()
    df_sorted = df.sort_values(["rec_name", "t_start"])
    for lag in range(1, n_lags + 1):
        lagged = (
            df_sorted.groupby("rec_name")[col]
            .shift(lag)
        )
        df_sorted[f"{col}_lag{lag}"] = lagged
    # merge back to preserve original row order
    df = df.merge(
        df_sorted[["rec_name", "t_start"] + [f"{col}_lag{i}" for i in range(1, n_lags + 1)]],
        on=["rec_name", "t_start"],
        how="left",
        suffixes=("", "_dup"),
    )
    # drop any accidental duplicate columns
    df = df.loc[:, ~df.columns.str.endswith("_dup")]
    return df


def _add_delta(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add `delta_<col>` = col_t - col_{t-1} within each recording."""
    df = df.copy()
    df_sorted = df.sort_values(["rec_name", "t_start"])
    delta = df_sorted.groupby("rec_name")[col].diff()
    df_sorted[f"delta_{col}"] = delta
    df = df.merge(
        df_sorted[["rec_name", "t_start", f"delta_{col}"]],
        on=["rec_name", "t_start"],
        how="left",
        suffixes=("", "_dup"),
    )
    df = df.loc[:, ~df.columns.str.endswith("_dup")]
    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, classes) -> dict:
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        per_class_f1={c: float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
                      for c in classes},
        confusion=confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    )


def _pooled_metrics(cm_total, classes) -> dict:
    cm = np.asarray(cm_total, dtype=float)
    f1s = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return dict(
        pooled_accuracy=acc,
        pooled_macro_f1=float(np.mean(list(f1s.values()))),
        pooled_per_class_f1={c: float(v) for c, v in f1s.items()},
    )


def _agg(folds, classes) -> dict:
    acc = [f["accuracy"] for f in folds]
    f1m = [f["macro_f1"] for f in folds]
    per = {c: [f["per_class_f1"][c] for f in folds] for c in classes}
    cm  = np.sum([np.asarray(f["confusion"]) for f in folds], axis=0)
    return dict(
        mean_accuracy=float(np.mean(acc)), std_accuracy=float(np.std(acc)),
        mean_macro_f1=float(np.mean(f1m)), std_macro_f1=float(np.std(f1m)),
        per_class_f1_mean={c: float(np.mean(per[c])) for c in classes},
        confusion_total=cm.tolist(),
        **_pooled_metrics(cm, classes),
    )


def _pad_missing(train_df, feat_cols, classes):
    missing = set(classes) - set(train_df["activity"].unique())
    if not missing:
        return train_df
    pad = pd.DataFrame([
        {**{c: np.nan for c in feat_cols},
         "rec_name": f"_pad_{m}", "subject": "_pad", "stressor": "_pad",
         "phase": m, "activity": m, "t_start": 0.0, "t_end": 0.0}
        for m in missing
    ])
    return pd.concat([train_df, pad], ignore_index=True)


def _make_xgb():
    import os
    use_gpu = os.environ.get("BIESL_USE_GPU", "0") == "1"
    return SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=0, n_jobs=-1, eval_metric="mlogloss",
        tree_method="hist", device=("cuda" if use_gpu else "cpu"),
    )


def loro_xgb(df: pd.DataFrame, classes: list[str], label: str) -> dict:
    """Run LORO XGBoost and return aggregated result dict."""
    le = LabelEncoder().fit(classes)
    feat_cols = sorted(set(df.columns) - META_COLS)
    recs = sorted(df["rec_name"].unique())
    print(f"\n[{label}] features ({len(feat_cols)}): {feat_cols}")
    print(f"[{label}] recordings: {len(recs)}")

    folds = []
    for r in recs:
        test_df  = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue
        train_df = _pad_missing(train_df, feat_cols, classes)
        clf = _make_xgb()
        clf.fit(train_df[feat_cols].to_numpy(float), le.transform(train_df["activity"]))
        pred = clf.predict(test_df[feat_cols].to_numpy(float))
        fold_res = _metrics(
            le.inverse_transform(le.transform(test_df["activity"])),
            le.inverse_transform(pred),
            classes,
        )
        folds.append(fold_res)
        print(f"  fold [{r:30s}]  acc={fold_res['accuracy']:.3f}  f1={fold_res['macro_f1']:.3f}")

    result = _agg(folds, classes)
    print(f"\n  LORO mean macro-F1 : {result['mean_macro_f1']:.3f} 짹 {result['std_macro_f1']:.3f}")
    print(f"  Pooled macro-F1   : {result['pooled_macro_f1']:.3f}")
    print(f"  Pooled accuracy   : {result['pooled_accuracy']:.3f}")
    return result


# ---------------------------------------------------------------------------
# Confusion matrix figure
# ---------------------------------------------------------------------------

def _save_cm_png(cm, classes, title: str, path: Path):
    cm = np.asarray(cm, dtype=float)
    rs = cm.sum(axis=1, keepdims=True)
    pct = cm / np.where(rs == 0, 1, rs) * 100.0
    supports = cm.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(classes), 1.4 + 1.0 * len(classes)))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels([f"{c}\n(n={n})" for c, n in zip(classes, supports)])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if pct[i, j] >= 50 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045).set_label("% of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" Building baseline feature table (all 19 recordings, no temp) ...")
    print("=" * 72)
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df_raw = df_raw[df_raw["activity"].isin(CLASSES)].copy()
    print(f"  Total windows: {len(df_raw)}  {dict(df_raw['activity'].value_counts())}")

    results = {}

    # -----------------------------------------------------------------------
    # VERSION 1: Baseline 8 features + samp_en_lag1 + samp_en_lag2
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" VERSION 1: Baseline 8 features + samp_en_lag1 + samp_en_lag2")
    print("=" * 72)
    df_v1 = _add_lags(df_raw, col="samp_en", n_lags=2)
    res_v1 = loro_xgb(df_v1, CLASSES, label="v1_lagged_sampen")
    results["v1_lagged_sampen"] = res_v1
    _save_cm_png(
        res_v1["confusion_total"], CLASSES,
        "XGBoost LORO  ??v1: +samp_en_lag1/lag2",
        Path("figures") / "ablation" / "loro__xgb_v1_lagged_sampen.png",
    )

    # -----------------------------------------------------------------------
    # VERSION 2: 7 features (no samp_en) + delta_hr
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" VERSION 2: Baseline without samp_en + delta_hr")
    print("=" * 72)
    df_v2 = _add_delta(df_raw, col="hr")
    df_v2 = df_v2.drop(columns=["samp_en"], errors="ignore")
    res_v2 = loro_xgb(df_v2, CLASSES, label="v2_delta_hr")
    results["v2_delta_hr"] = res_v2
    _save_cm_png(
        res_v2["confusion_total"], CLASSES,
        "XGBoost LORO  ??v2: -samp_en +delta_hr",
        Path("figures") / "ablation" / "loro__xgb_v2_delta_hr.png",
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" SUMMARY  (pooled metrics over all held-out recordings)")
    print("=" * 72)
    header = " / ".join(CLASSES)
    print(f"  {'version':<30s}  {'pooled_acc':>10s}  {'pooled_f1':>10s}   per-class F1  ({header})")
    for key, res in results.items():
        pc = "  /  ".join(f"{res['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
        print(f"  {key:<30s}  {res['pooled_accuracy']:>10.3f}  {res['pooled_macro_f1']:>10.3f}   {pc}")

    # Also show baseline for reference
    print("\n  (baseline XGBoost from split_reports.json shown for comparison)")
    baseline_path = Path("outputs") / "split_reports.json"
    if baseline_path.exists():
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        xgb_base = baseline.get("with_math", {}).get("classical", {}).get("loro", {}).get("xgboost", {})
        if xgb_base:
            pc = "  /  ".join(f"{xgb_base['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
            print(f"  {'baseline_xgb (8 feat)':<30s}  {xgb_base['pooled_accuracy']:>10.3f}  {xgb_base['pooled_macro_f1']:>10.3f}   {pc}")

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / "ablation_v1v2.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()

