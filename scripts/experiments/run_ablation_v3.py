#!/usr/bin/env python
"""VERSION 3 ablation: V2 (no samp_en, +delta_hr) AND +resp_amp.

Feature set (8 features total):
    csi, hr, hrv_rmssd, sd2_sd1, ss, rr, rrv  ??kept from baseline
    delta_hr   = HR_t - HR_{t-1}               ??replaces samp_en
    resp_amp                                    ??new: peak-to-trough BR amplitude

Evaluation: LORO XGBoost (SafeXGBClassifier) over 19 recordings.

Usage (from project root):
    uv run python scripts/experiments/run_ablation_v3.py
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

from src.pipeline import build_feature_table, SafeXGBClassifier
from src.local_eval import META_COLS
from src.exclusions import filter_feature_df

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]


# ---------------------------------------------------------------------------
# Temporal delta feature
# ---------------------------------------------------------------------------

def _add_delta(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add delta_<col> = col_t - col_{t-1} within each recording."""
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
# Metrics & confusion matrix
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


def loro_xgb(df: pd.DataFrame, classes: list[str], label: str) -> dict:
    le = LabelEncoder().fit(classes)
    feat_cols = sorted(set(df.columns) - META_COLS)
    recs = sorted(df["rec_name"].unique())
    print(f"\n[{label}]")
    print(f"  features ({len(feat_cols)}): {feat_cols}")
    print(f"  recordings: {len(recs)}, windows: {len(df)}")

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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" Building feature table (8 baseline + resp_amp = 9 features) ...")
    print("=" * 72)
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df_raw = df_raw[df_raw["activity"].isin(CLASSES)].copy()
    print(f"  Total windows: {len(df_raw)}  {dict(df_raw['activity'].value_counts())}")

    # Sanity check: confirm resp_amp is in columns
    feat_cols_raw = sorted(set(df_raw.columns) - META_COLS)
    print(f"  Raw feature columns ({len(feat_cols_raw)}): {feat_cols_raw}")

    print("\n" + "=" * 72)
    print(" VERSION 3: V2 base (no samp_en, +delta_hr) + resp_amp")
    print(" Features: csi, hr, hrv_rmssd, sd2_sd1, ss, rr, rrv, delta_hr, resp_amp")
    print("=" * 72)

    # Build V3 dataframe
    df_v3 = _add_delta(df_raw, col="hr")          # adds delta_hr
    df_v3 = df_v3.drop(columns=["samp_en"], errors="ignore")  # remove samp_en

    res_v3 = loro_xgb(df_v3, CLASSES, label="v3_delta_hr_resp_amp")

    figpath = Path("figures") / "ablation" / "loro__xgb_v3_delta_hr_resp_amp.png"
    _save_cm_png(
        res_v3["confusion_total"], CLASSES,
        "XGBoost LORO  ??v3: -samp_en +delta_hr +resp_amp",
        figpath,
    )

    # ---------------------------------------------------------------------------
    # Compare with prior versions
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" COMPARISON  (pooled metrics over all held-out recordings)")
    print("=" * 72)
    header = " / ".join(CLASSES)
    print(f"  {'version':<32s}  {'pooled_acc':>10s}  {'pooled_f1':>10s}   per-class F1  ({header})")

    # Current result
    pc = "  /  ".join(f"{res_v3['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
    print(f"  {'v3 (delta_hr + resp_amp)':<32s}  {res_v3['pooled_accuracy']:>10.3f}  {res_v3['pooled_macro_f1']:>10.3f}   {pc}")

    # Load prior results for comparison
    abl_path = Path("outputs") / "ablation_v1v2.json"
    if abl_path.exists():
        with open(abl_path) as fh:
            prior = json.load(fh)
        for key, label in [("v2_delta_hr", "v2 (delta_hr, no resp_amp)"),
                            ("v1_lagged_sampen", "v1 (samp_en lag1/lag2)")]:
            r = prior.get(key, {})
            if r:
                pc = "  /  ".join(f"{r['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
                print(f"  {label:<32s}  {r['pooled_accuracy']:>10.3f}  {r['pooled_macro_f1']:>10.3f}   {pc}")

    base_path = Path("outputs") / "split_reports.json"
    if base_path.exists():
        with open(base_path) as fh:
            base = json.load(fh)
        r = base.get("with_math", {}).get("classical", {}).get("loro", {}).get("xgboost", {})
        if r:
            pc = "  /  ".join(f"{r['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
            print(f"  {'baseline (8 feat, samp_en)':<32s}  {r['pooled_accuracy']:>10.3f}  {r['pooled_macro_f1']:>10.3f}   {pc}")

    # Save
    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / "ablation_v3.json"
    with open(out_path, "w") as fh:
        json.dump({"v3_delta_hr_resp_amp": res_v3}, fh, indent=2)
    print(f"\n  -> wrote {out_path}")
    print(f"  -> confusion matrix: {figpath}")


if __name__ == "__main__":
    main()

