#!/usr/bin/env python
"""Three ablation experiments, all building on V3 base features:
    (V3 base = -samp_en, +delta_hr, +resp_amp)

V4 ??V3 base + mic_rms
    XGBoost LORO.  mic_rms = RMS energy of the Shannon-energy envelope;
    physical exertion raises acoustic energy well above mental-stress levels.

V5 ??V3 base + SMOTE oversampling of the minority class (plank)
    XGBoost LORO.  SMOTE is applied PER FOLD to training data only,
    targeting the median class count so plank is no longer starved.

V6 ??V3 base ??LSTM on feature sequences
    For each window at position t, the LSTM sees a context window of the
    preceding SEQ_LEN windows (sorted by t_start within each recording).
    Trained with weighted cross-entropy to handle imbalance.
    LORO evaluation: train on 18 recordings, test on 1.

Usage (from project root):
    uv run python scripts/experiments/run_ablation_v4v5v6.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.pipeline import build_feature_table, SafeXGBClassifier
from src.local_eval import META_COLS
from src.exclusions import filter_feature_df

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]
SEQ_LEN = 10          # LSTM context: 10 windows = 20 s of history
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 60
LSTM_LR = 1e-3
LSTM_BATCH = 64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_delta(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    s = df.sort_values(["rec_name", "t_start"])
    s[f"delta_{col}"] = s.groupby("rec_name")[col].diff()
    df = df.merge(s[["rec_name", "t_start", f"delta_{col}"]],
                  on=["rec_name", "t_start"], how="left",
                  suffixes=("", "_dup"))
    return df.loc[:, ~df.columns.str.endswith("_dup")]


def _metrics(y_true, y_pred, classes) -> dict:
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=classes,
                                average="macro", zero_division=0)),
        per_class_f1={c: float(f1_score(y_true, y_pred, labels=[c],
                                        average="macro", zero_division=0))
                      for c in classes},
        confusion=confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    )


def _pooled_metrics(cm_total, classes) -> dict:
    cm = np.asarray(cm_total, dtype=float)
    f1s = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return dict(pooled_accuracy=acc,
                pooled_macro_f1=float(np.mean(list(f1s.values()))),
                pooled_per_class_f1={c: float(v) for c, v in f1s.items()})


def _agg(folds, classes) -> dict:
    acc = [f["accuracy"] for f in folds]
    f1m = [f["macro_f1"] for f in folds]
    per = {c: [f["per_class_f1"][c] for f in folds] for c in classes}
    cm  = np.sum([np.asarray(f["confusion"]) for f in folds], axis=0)
    return dict(
        mean_accuracy=float(np.mean(acc)), std_accuracy=float(np.std(acc)),
        mean_macro_f1=float(np.mean(f1m)), std_macro_f1=float(np.std(f1m)),
        per_class_f1_mean={c: float(np.mean(per[c])) for c in classes},
        confusion_total=cm.tolist(), **_pooled_metrics(cm, classes),
    )


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
    use_gpu = os.environ.get("BIESL_USE_GPU", "0") == "1"
    return SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        eval_metric="mlogloss", tree_method="hist",
        device=("cuda" if use_gpu else "cpu"),
    )


def _save_cm_png(cm, classes, title: str, path: Path):
    cm = np.asarray(cm, dtype=float)
    rs  = cm.sum(axis=1, keepdims=True)
    pct = cm / np.where(rs == 0, 1, rs) * 100.0
    supports = cm.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(classes),
                                    1.4 + 1.0 * len(classes)))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels([f"{c}\n(n={n})" for c, n in zip(classes, supports)])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i,j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if pct[i,j] >= 50 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045).set_label("% of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> saved {path}")


def _loro_xgb(df, classes, label, clf_fn=None):
    if clf_fn is None:
        clf_fn = _make_xgb
    le = LabelEncoder().fit(classes)
    feat_cols = sorted(set(df.columns) - META_COLS)
    recs = sorted(df["rec_name"].unique())
    print(f"\n[{label}]")
    print(f"  features ({len(feat_cols)}): {feat_cols}")
    folds = []
    for r in recs:
        test_df  = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue
        train_df = _pad_missing(train_df, feat_cols, classes)
        clf = clf_fn()
        clf.fit(train_df[feat_cols].to_numpy(float),
                le.transform(train_df["activity"]))
        pred = clf.predict(test_df[feat_cols].to_numpy(float))
        fold_res = _metrics(
            le.inverse_transform(le.transform(test_df["activity"])),
            le.inverse_transform(pred), classes)
        folds.append(fold_res)
        print(f"  fold [{r:30s}]  acc={fold_res['accuracy']:.3f}"
              f"  f1={fold_res['macro_f1']:.3f}")
    result = _agg(folds, classes)
    print(f"\n  LORO mean macro-F1 : {result['mean_macro_f1']:.3f}"
          f" 짹 {result['std_macro_f1']:.3f}")
    print(f"  Pooled macro-F1   : {result['pooled_macro_f1']:.3f}")
    print(f"  Pooled accuracy   : {result['pooled_accuracy']:.3f}")
    return result


# ---------------------------------------------------------------------------
# V4 ??+mic_rms
# ---------------------------------------------------------------------------

def run_v4(df_base):
    print("\n" + "=" * 72)
    print(" V4: V3 base + mic_rms")
    print("=" * 72)
    # mic_rms is already in df_base (extracted by features.py)
    # Apply V3 transformations: -samp_en, +delta_hr
    df = _add_delta(df_base, col="hr")
    df = df.drop(columns=["samp_en"], errors="ignore")
    # mic_rms stays in
    result = _loro_xgb(df, CLASSES, label="v4_mic_rms")
    _save_cm_png(result["confusion_total"], CLASSES,
                 "XGBoost LORO ??v4: V3 + mic_rms",
                 Path("figures") / "ablation" / "loro__xgb_v4_mic_rms.png")
    return result


# ---------------------------------------------------------------------------
# V5 ??SMOTE on minority classes
# ---------------------------------------------------------------------------

def run_v5(df_base):
    print("\n" + "=" * 72)
    print(" V5: V3 base + SMOTE (per-fold, training only)")
    print("=" * 72)

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("  [ERROR] imbalanced-learn not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "imbalanced-learn"],
                       check=True, capture_output=True)
        from imblearn.over_sampling import SMOTE

    df = _add_delta(df_base, col="hr")
    df = df.drop(columns=["samp_en"], errors="ignore")
    # mic_rms stays in (same as V4, showing SMOTE effect on V4 features)

    le = LabelEncoder().fit(CLASSES)
    feat_cols = sorted(set(df.columns) - META_COLS)
    recs = sorted(df["rec_name"].unique())
    print(f"  features ({len(feat_cols)}): {feat_cols}")

    folds = []
    for r in recs:
        test_df  = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue
        train_df = _pad_missing(train_df, feat_cols, CLASSES)

        X_tr = train_df[feat_cols].to_numpy(float)
        y_tr = le.transform(train_df["activity"])

        # Impute NaN before SMOTE (SMOTE cannot handle NaN)
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)

        # SMOTE: upsample minority classes to median class count
        counts = np.bincount(y_tr)
        target_n = int(np.median(counts[counts > 1]))
        sampling = {i: max(int(counts[i]), target_n)
                    for i in range(len(CLASSES))
                    if counts[i] > 0}
        k_neighbors = min(5, min(c - 1 for c in counts if c > 1))
        k_neighbors = max(1, k_neighbors)
        try:
            sm = SMOTE(sampling_strategy=sampling, k_neighbors=k_neighbors,
                       random_state=0)
            X_res, y_res = sm.fit_resample(X_tr_imp, y_tr)
        except Exception as e:
            print(f"    SMOTE fallback (no oversampling): {e}")
            X_res, y_res = X_tr_imp, y_tr

        clf = _make_xgb()
        clf.fit(X_res, y_res)

        X_te = test_df[feat_cols].to_numpy(float)
        X_te_imp = imp.transform(X_te)   # same imputer fitted on training
        pred = clf.predict(X_te_imp)

        fold_res = _metrics(
            le.inverse_transform(le.transform(test_df["activity"])),
            le.inverse_transform(pred), CLASSES)
        folds.append(fold_res)
        print(f"  fold [{r:30s}]  acc={fold_res['accuracy']:.3f}"
              f"  f1={fold_res['macro_f1']:.3f}")

    result = _agg(folds, CLASSES)
    print(f"\n  LORO mean macro-F1 : {result['mean_macro_f1']:.3f}"
          f" 짹 {result['std_macro_f1']:.3f}")
    print(f"  Pooled macro-F1   : {result['pooled_macro_f1']:.3f}")
    print(f"  Pooled accuracy   : {result['pooled_accuracy']:.3f}")
    _save_cm_png(result["confusion_total"], CLASSES,
                 "XGBoost LORO ??v5: V4 + SMOTE",
                 Path("figures") / "ablation" / "loro__xgb_v5_smote.png")
    return result


# ---------------------------------------------------------------------------
# V6 ??LSTM on feature sequences
# ---------------------------------------------------------------------------

def run_v6(df_base):
    print("\n" + "=" * 72)
    print(f" V6: LSTM on feature sequences (context={SEQ_LEN} windows = {SEQ_LEN*2} s)")
    print("=" * 72)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [ERROR] PyTorch not installed.")
        return {}

    df = _add_delta(df_base, col="hr")
    df = df.drop(columns=["samp_en"], errors="ignore")

    le = LabelEncoder().fit(CLASSES)
    feat_cols = sorted(set(df.columns) - META_COLS)
    n_feat = len(feat_cols)
    n_classes = len(CLASSES)
    print(f"  features ({n_feat}): {feat_cols}")

    # Build sequences per recording -----------------------------------------
    def _build_sequences(sub_df, feat_cols, seq_len):
        """For each window at pos t, stack [t-seq_len+1 ??t] ??shape (seq_len, n_feat)."""
        sub_df = sub_df.sort_values("t_start").reset_index(drop=True)
        X_raw = sub_df[feat_cols].to_numpy(float)
        y_raw = le.transform(sub_df["activity"].to_numpy())
        seqs, labels = [], []
        for i in range(len(X_raw)):
            lo = max(0, i - seq_len + 1)
            chunk = X_raw[lo: i + 1]
            # left-pad with zeros if shorter than seq_len
            if len(chunk) < seq_len:
                pad = np.zeros((seq_len - len(chunk), n_feat), dtype=np.float32)
                chunk = np.vstack([pad, chunk])
            seqs.append(chunk.astype(np.float32))
            labels.append(y_raw[i])
        return np.stack(seqs, axis=0), np.array(labels, dtype=np.int64)

    # LSTM model ---------------------------------------------------------------
    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, LSTM_HIDDEN, LSTM_LAYERS,
                                batch_first=True,
                                dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0)
            self.fc = nn.Linear(LSTM_HIDDEN, n_classes)

        def forward(self, x):               # x: (B, seq_len, n_feat)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])   # last time-step ??logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  torch device: {device}")

    recs = sorted(df["rec_name"].unique())
    folds = []

    for r in recs:
        test_df  = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if test_df["activity"].nunique() < 1:
            continue

        # Impute + scale on training data, transform both
        imp  = SimpleImputer(strategy="median")
        scl  = StandardScaler()
        X_tr_all = train_df[feat_cols].to_numpy(float)
        X_tr_imp = imp.fit_transform(X_tr_all)
        X_tr_scl = scl.fit_transform(X_tr_imp)
        train_df = train_df.copy()
        train_df[feat_cols] = X_tr_scl

        X_te_all = test_df[feat_cols].to_numpy(float)
        X_te_imp = imp.transform(X_te_all)
        X_te_scl = scl.transform(X_te_imp)
        test_df = test_df.copy()
        test_df[feat_cols] = X_te_scl

        # Build per-recording sequences
        train_seqs, train_labels = [], []
        for rec_name in sorted(train_df["rec_name"].unique()):
            s, l = _build_sequences(train_df[train_df.rec_name == rec_name],
                                    feat_cols, SEQ_LEN)
            train_seqs.append(s); train_labels.append(l)
        X_tr = np.concatenate(train_seqs, axis=0)
        y_tr = np.concatenate(train_labels, axis=0)

        X_te, y_te = _build_sequences(test_df, feat_cols, SEQ_LEN)

        # Class weights (inverse frequency)
        counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        counts = np.where(counts == 0, 1, counts)
        weights = torch.tensor(1.0 / counts / (1.0 / counts).sum(),
                               dtype=torch.float32).to(device)

        # DataLoader
        ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        loader = DataLoader(ds_tr, batch_size=LSTM_BATCH, shuffle=True)

        # Train
        model = LSTMClassifier().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        crit  = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for epoch in range(LSTM_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_te).to(device))
            pred = logits.argmax(dim=1).cpu().numpy()

        fold_res = _metrics(le.inverse_transform(y_te),
                            le.inverse_transform(pred), CLASSES)
        folds.append(fold_res)
        print(f"  fold [{r:30s}]  acc={fold_res['accuracy']:.3f}"
              f"  f1={fold_res['macro_f1']:.3f}")

    result = _agg(folds, CLASSES)
    print(f"\n  LORO mean macro-F1 : {result['mean_macro_f1']:.3f}"
          f" 짹 {result['std_macro_f1']:.3f}")
    print(f"  Pooled macro-F1   : {result['pooled_macro_f1']:.3f}")
    print(f"  Pooled accuracy   : {result['pooled_accuracy']:.3f}")
    _save_cm_png(result["confusion_total"], CLASSES,
                 f"LSTM LORO ??v6: seq={SEQ_LEN}횞2s context",
                 Path("figures") / "ablation" / "loro__lstm_v6_sequences.png")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" Building feature table (10 raw features incl. resp_amp + mic_rms) ...")
    print("=" * 72)
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df_raw = df_raw[df_raw["activity"].isin(CLASSES)].copy()
    raw_feat = sorted(set(df_raw.columns) - META_COLS)
    print(f"  Windows: {len(df_raw)}  {dict(df_raw['activity'].value_counts())}")
    print(f"  Raw feature columns ({len(raw_feat)}): {raw_feat}")

    results = {}
    results["v4_mic_rms"] = run_v4(df_raw)
    results["v5_smote"]   = run_v5(df_raw)
    results["v6_lstm"]    = run_v6(df_raw)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" FULL COMPARISON  (pooled metrics)")
    print("=" * 72)
    header = " / ".join(CLASSES)
    print(f"  {'version':<34s}  {'acc':>6s}  {'f1':>6s}   per-class ({header})")

    for key, res in results.items():
        if not res:
            continue
        pc = " / ".join(f"{res['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
        print(f"  {key:<34s}  {res['pooled_accuracy']:>6.3f}  "
              f"{res['pooled_macro_f1']:>6.3f}   {pc}")

    # Load prior results
    for path, name_map in [
        ("outputs/ablation_v3.json",
         [("v3_delta_hr_resp_amp", "v3 (-samp_en +delta_hr +resp_amp)")]),
        ("outputs/ablation_v1v2.json",
         [("v2_delta_hr", "v2 (-samp_en +delta_hr)")]),
        ("outputs/split_reports.json", []),
    ]:
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as fh:
            data = json.load(fh)
        if path.endswith("split_reports.json"):
            r = (data.get("with_math", {}).get("classical", {})
                     .get("loro", {}).get("xgboost", {}))
            if r:
                pc = " / ".join(f"{r['pooled_per_class_f1'][c]:.3f}" for c in CLASSES)
                print(f"  {'baseline (8 feat, samp_en)':<34s}  "
                      f"{r['pooled_accuracy']:>6.3f}  "
                      f"{r['pooled_macro_f1']:>6.3f}   {pc}")
        else:
            for key, label in name_map:
                r = data.get(key, {})
                if r:
                    pc = " / ".join(f"{r['pooled_per_class_f1'][c]:.3f}"
                                    for c in CLASSES)
                    print(f"  {label:<34s}  {r['pooled_accuracy']:>6.3f}  "
                          f"{r['pooled_macro_f1']:>6.3f}   {pc}")

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / "ablation_v4v5v6.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  -> wrote {out_path}")


if __name__ == "__main__":
    main()

