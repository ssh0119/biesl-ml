#!/usr/bin/env python
"""Train + evaluate the 4-class model and emit confusion-matrix PNGs.

5 classes: rest / mental1 / mental2 / physical1 / physical2.
Models: KNN / RandomForest / XGBoost / 1D-CNN.
Protocol: LORO (leave one recording out) only. Random 70:15:15 has been
dropped ??at a 2-s anchor step, windows 2 s apart are near-duplicates so
the random split leaks heavily and the score collapses to ~1.0 for every
model. LORO is the only honest cross-subject estimate.

Writes:
  outputs/split_reports.json                       ??all numbers
  figures/with_math/confusion/loro__*.png          ??row-normalized % heatmaps

Usage:
    uv run python scripts/evaluation/run_split_reports.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score   # noqa: E402
from sklearn.preprocessing import LabelEncoder                            # noqa: E402

from src.pipeline import build_feature_table          # noqa: E402
from src.local_eval import (                           # noqa: E402
    META_COLS, MODEL_FACTORIES, _stratified_three_way_split,
)
from src.raw_windows import local_raw_windows, stack_windows   # noqa: E402
from src.dl_train import EPOCHS, predict, train_one_fold       # noqa: E402
from src.exclusions import filter_feature_df, filter_raw_windows   # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
CONFIGS = {
    "phase2_folders": ["rest", "mental1", "mental2", "physical1", "physical2"],
}


# ---- metrics + confusion ---------------------------------------------------

def _metrics(y_true, y_pred, classes) -> dict:
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        per_class_f1={c: float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
                      for c in classes},
        confusion=confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    )


def _pooled_metrics(cm, classes) -> dict:
    """Precision/recall/F1 computed from the POOLED (summed) confusion matrix.

    This is the correct LORO aggregation for this dataset: each recording
    contains only `rest` + ONE stressor, so per-fold macro-F1 averaging charges
    zeros for the 2-3 classes absent from each fold's test set and badly
    understates performance. Pooling all held-out predictions first avoids that.
    """
    cm = np.asarray(cm, dtype=float)
    f1s = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return dict(pooled_accuracy=acc,
                pooled_macro_f1=float(np.mean(list(f1s.values()))),
                pooled_per_class_f1={c: float(v) for c, v in f1s.items()})


def _agg(folds, classes) -> dict:
    acc = [f["accuracy"] for f in folds]
    f1m = [f["macro_f1"] for f in folds]
    per = {c: [f["per_class_f1"][c] for f in folds] for c in classes}
    cm = np.sum([np.asarray(f["confusion"]) for f in folds], axis=0)
    return dict(
        # per-fold averaged (fine for stratified random split; understates LORO)
        mean_accuracy=float(np.mean(acc)), std_accuracy=float(np.std(acc)),
        mean_macro_f1=float(np.mean(f1m)), std_macro_f1=float(np.std(f1m)),
        per_class_f1_mean={c: float(np.mean(per[c])) for c in classes},
        confusion_total=cm.tolist(),
        # pooled over all held-out predictions (correct for LORO here)
        **_pooled_metrics(cm, classes),
    )


def _save_cm_png(cm, classes, title, path: Path):
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


# ---- classical -------------------------------------------------------------

def _pad(train_df, feat_cols, classes):
    missing = set(classes) - set(train_df["activity"].unique())
    if not missing:
        return train_df
    import pandas as pd
    pad = pd.DataFrame([{**{c: np.nan for c in feat_cols},
                         "rec_name": f"_pad_{m}", "subject": "_pad", "stressor": "_pad",
                         "phase": m, "activity": m, "t_start": 0.0, "t_end": 0.0}
                        for m in missing])
    return pd.concat([train_df, pad], ignore_index=True)


def _clf_fit_predict(train_df, test_df, feat_cols, le, classes, model):
    train_df = _pad(train_df, feat_cols, classes)
    clf = MODEL_FACTORIES[model]()
    clf.fit(train_df[feat_cols].to_numpy(float), le.transform(train_df["activity"]))
    pred = clf.predict(test_df[feat_cols].to_numpy(float))
    return _metrics(le.inverse_transform(le.transform(test_df["activity"])),
                    le.inverse_transform(pred), classes)


def classical_loro(df, classes, models):
    le = LabelEncoder().fit(classes)
    feat_cols = sorted(set(df.columns) - META_COLS)
    recs = sorted(df["rec_name"].unique())
    out = {}
    for m in models:
        folds = [_clf_fit_predict(df[df.rec_name != r], df[df.rec_name == r], feat_cols, le, classes, m)
                 for r in recs if df[df.rec_name == r]["activity"].nunique() >= 1]
        out[m] = _agg(folds, classes)
    return out


def classical_random(df, classes, models, seeds=SEEDS):
    le = LabelEncoder().fit(classes)
    feat_cols = sorted(set(df.columns) - META_COLS)
    y_all = df["activity"].to_numpy()
    out = {}
    for m in models:
        folds = []
        for s in seeds:
            itr, iva, ite = _stratified_three_way_split(len(df), y_all, seed=s)
            folds.append(_clf_fit_predict(df.iloc[itr], df.iloc[ite], feat_cols, le, classes, m))
        out[m] = _agg(folds, classes)
    return out


# ---- CNN -------------------------------------------------------------------

def _cnn_eval(Xtr, ytr, Xva, yva, Xte, yte, classes, le):
    model, _ = train_one_fold(Xtr, ytr, Xva, yva,
                              in_channels=Xtr.shape[1], n_classes=len(classes), epochs=EPOCHS)
    pred = predict(model, Xte)
    return _metrics(le.inverse_transform(yte), le.inverse_transform(pred), classes)


def cnn_loro(X, y, recs, classes, le):
    uniq = sorted(set(recs.tolist()))
    folds = []
    for r in uniq:
        te = recs == r
        pool_X, pool_y, pool_r = X[~te], y[~te], recs[~te]
        others = sorted(set(pool_r.tolist()))
        vrec = others[0]
        vm = pool_r == vrec
        folds.append(_cnn_eval(pool_X[~vm], pool_y[~vm], pool_X[vm], pool_y[vm],
                               X[te], y[te], classes, le))
    return _agg(folds, classes)


def cnn_random(X, y, classes, le, seeds=SEEDS):
    folds = []
    for s in seeds:
        itr, iva, ite = _stratified_three_way_split(len(X), y, seed=s)
        folds.append(_cnn_eval(X[itr], y[itr], X[iva], y[iva], X[ite], y[ite], classes, le))
    return _agg(folds, classes)


# ---- driver ----------------------------------------------------------------

def main():
    print("Building classical feature table + CNN raw windows (all recordings) ...")
    df_all = build_feature_table(include_temp=False)
    win_all = local_raw_windows(include_temp=False)
    n_df0, n_w0 = len(df_all), len(win_all)
    df_all = filter_feature_df(df_all)
    win_all = filter_raw_windows(win_all)
    print(f"  partial exclusions: classical {n_df0}->{len(df_all)}  "
          f"CNN {n_w0}->{len(win_all)}")
    models = ("knn", "randomforest", "xgboost")

    results = {}
    for cfg, classes in CONFIGS.items():
        print(f"\n{'='*70}\n {cfg}: classes={classes}\n{'='*70}")
        df = df_all[df_all["activity"].isin(classes)].copy()
        wins = [w for w in win_all if w.activity in classes]
        X, y_raw = stack_windows(wins)
        # remap CNN integer labels to a compact 0..k-1 over the *string* classes
        le = LabelEncoder().fit(classes)
        y = le.transform([w.activity for w in wins])
        recs = np.array([w.rec_name for w in wins])
        print(f"  classical windows: {len(df)}  {dict(Counter(df['activity']))}")
        print(f"  CNN windows:       {len(X)}  {dict(Counter(le.inverse_transform(y)))}")

        # Random 70:15:15 was dropped: at a 2-s anchor step, windows are
        # near-duplicates of their neighbours, so a window-stratified random
        # split leaks heavily and the score collapses to ~1.0 for every
        # model. LORO is the only honest cross-subject estimate.
        res = {
            "classes": classes,
            "classical": {
                "loro": classical_loro(df, classes, models),
            },
            # "cnn": {
            #     "loro": cnn_loro(X, y, recs, classes, le),
            # },
        }
        results[cfg] = res

        # confusion PNGs (LORO only)
        figdir = Path("figures") / cfg / "confusion"
        for m in models:
            _save_cm_png(res["classical"]["loro"][m]["confusion_total"], classes,
                         f"{m.upper()}  LORO  ({cfg})", figdir / f"loro__{m}.png")
        # _save_cm_png(res["cnn"]["loro"]["confusion_total"], classes,
        #              f"1D-CNN  LORO  ({cfg})", figdir / "loro__cnn.png")

        # console summary
        print(f"\n  {'model':<14s}{'LORO f1':>9s}")
        for m in models:
            print(f"  {m:<14s}{res['classical']['loro'][m]['mean_macro_f1']:>9.3f}")
        # print(f"  {'1D-CNN':<14s}{res['cnn']['loro']['mean_macro_f1']:>9.3f}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/split_reports.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n  -> wrote outputs/split_reports.json")


if __name__ == "__main__":
    main()

