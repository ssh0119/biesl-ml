"""End-to-end pipeline: load -> preprocess -> features -> KNN with subject-wise CV.

Run from the project root:
    uv run python -m src.pipeline
"""
from __future__ import annotations

import json
import os
import time
import warnings
from collections import Counter
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from .features import preprocess_recording, windows_for_recording
from .io import list_recordings, load_recording


warnings.filterwarnings("ignore")  # filterwarnings handled per-window

DATA_DIR = "data"
OUT_DIR = "outputs"

PHASE2_FOLDER_LABELS = {
    "mental1": "mental1",
    "mental2": "mental2",
    "physical1": "physical1",
    "physical2": "physical2",
}
PHASE_CLASSES = ["rest", *PHASE2_FOLDER_LABELS.values()]


def assign_activity(phase: str, stressor: str, phase2_label: str | None = None) -> str:
    """Map a protocol phase to its training label.

    Phase 1 is always ``rest``. Phase 2 uses ``phase2_label`` from the
    recording's parent folder. The filename stressor remains available as a
    fallback for legacy callers and protocol-duration parsing. Recovery is
    excluded by returning an empty label.
    """
    if phase == "rest":
        return "rest"
    if phase == "recovery":
        return ""                   # signal: drop this window
    # phase == "stress"
    if phase2_label is not None:
        return phase2_label
    if stressor == "medi":
        return "meditation"
    if stressor == "pla":
        return "plank"
    if stressor == "math":
        return "math"
    return f"stress_{stressor}"


def phase2_label_from_path(path: str) -> str | None:
    """Return the Phase 2 class encoded by a recording's parent folder."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(path))).lower()
    return PHASE2_FOLDER_LABELS.get(folder)


# ----------------------------------------------------------------------------

def build_feature_table(data_dir: str = DATA_DIR, *, include_temp: bool = False) -> pd.DataFrame:
    """Process every recording and produce a (n_windows, n_features + meta) DataFrame.

    `include_temp` adds the optional temperature ablation features (mean, std,
    slope) alongside the eight PDF features. The PDF feature set is the
    default; temperature is opt-in.
    """
    records: list[dict] = []
    for path in list_recordings(data_dir):
        t0 = time.time()
        phase2_label = phase2_label_from_path(path)
        if phase2_label is None:
            raise ValueError(
                f"recording is not inside a supported Phase 2 label folder: {path}. "
                f"Expected one of {sorted(PHASE2_FOLDER_LABELS)}"
            )
        rec = load_recording(path)
        pp = preprocess_recording(rec)
        wins = windows_for_recording(pp, include_temp=include_temp)
        for w in wins:
            row = {
                "rec_name": w.rec_name,
                "subject": w.subject,
                "stressor": w.stressor,
                "phase": w.phase,
                "activity": assign_activity(w.phase, w.stressor, phase2_label),
                "t_start": w.t_start,
                "t_end": w.t_end,
            }
            row.update(w.features)
            records.append(row)
        print(f"  {os.path.basename(path):34s} -> {len(wins):3d} windows ({time.time()-t0:.1f}s)")
    df = pd.DataFrame(records)
    return df


# Features whose absolute values are most confounded by inter-subject
# baseline differences (diagnosis: these have the largest effect size for
# math-vs-rest but fail cross-subject because nvt's math HR overlaps mta's
# rest HR). Normalising against each recording's own rest phase removes
# this confound.
DELTA_BASE_FEATURES = ("hr", "hrv_rmssd", "ss", "rr", "resp_amp")


def add_delta_features(
    df: pd.DataFrame,
    base_features: tuple[str, ...] = DELTA_BASE_FEATURES,
    rest_label: str = "rest",
) -> pd.DataFrame:
    """Add within-recording baseline-normalised (delta) features.

    For each recording, compute the rest-phase mean and std of each base
    feature. Then for EVERY window (including rest) compute:

        delta_{feat} = (value - rest_mean) / rest_std

    Rest windows will cluster around 0; stress windows will show the
    subject-relative deviation. This removes inter-subject baseline
    differences that confuse math ↔ rest classification.

    The original absolute features are kept — the model gets both.
    """
    df = df.copy()
    for feat in base_features:
        delta_col = f"delta_{feat}"
        df[delta_col] = np.nan

        for rec_name, grp in df.groupby("rec_name"):
            rest_mask = grp["activity"] == rest_label
            rest_vals = grp.loc[rest_mask, feat].dropna()
            if len(rest_vals) < 3:
                # Not enough rest data — leave as NaN (imputer handles it)
                continue
            rest_mean = float(rest_vals.mean())
            rest_std = float(rest_vals.std())
            if rest_std < 1e-9:
                rest_std = 1.0  # prevent division by zero

            idx = grp.index
            df.loc[idx, delta_col] = (df.loc[idx, feat] - rest_mean) / rest_std

    return df


def split_xy(df: pd.DataFrame, label_col: str = "activity") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X (float), y (str), groups (subject), feature_names."""
    meta_cols = {"rec_name", "subject", "stressor", "phase", "activity", "t_start", "t_end"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy()
    groups = df["subject"].to_numpy()
    return X, y, groups, feat_cols


def make_knn_pipeline() -> Pipeline:
    """Imputer -> Scaler -> SelectKBest(MI) -> KNN."""
    return Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("scaler",   StandardScaler()),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=12)),
        ("clf",      KNeighborsClassifier()),
    ])


def make_rf_pipeline() -> Pipeline:
    """Imputer -> RandomForest. Trees don't need scaling or feature selection."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(random_state=0, n_jobs=-1)),
    ])


class SafeXGBClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper around XGBClassifier that fits LabelEncoder locally on the
    current training split, mapping classes to consecutive integers starting from 0.
    This prevents XGBoost from raising ValueError in cross-validation folds that
    lack one or more classes.
    """
    def __init__(
        self,
        n_estimators=None,
        max_depth=None,
        learning_rate=None,
        subsample=None,
        colsample_bytree=None,
        random_state=None,
        n_jobs=None,
        eval_metric=None,
        tree_method=None,
        device=None,
        class_weight="balanced",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.device = device
        self.class_weight = class_weight
        self.clf_ = None
        self.le_ = None
        self.classes_ = None

    def fit(self, X, y, **fit_params):
        self.le_ = LabelEncoder().fit(y)
        y_encoded = self.le_.transform(y)
        kwargs = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "device": self.device,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.clf_ = XGBClassifier(**kwargs)

        fit_params_use = fit_params.copy()
        if self.class_weight == "balanced" and "sample_weight" not in fit_params_use:
            from sklearn.utils.class_weight import compute_sample_weight
            fit_params_use["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_encoded)

        self.clf_.fit(X, y_encoded, **fit_params_use)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        preds = self.clf_.predict(X)
        return self.le_.inverse_transform(preds)

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)


def make_xgb_pipeline() -> Pipeline:
    """XGBoost handles NaN natively and ignores feature scale.

    Set BIESL_USE_GPU=1 to run XGBoost on CUDA (requires xgboost>=2.0 + NVIDIA GPU).
    For this dataset size (~hundreds of samples) GPU is *slower* than CPU due to
    transfer overhead; enable it once you have ~10k+ samples.
    """
    use_gpu = os.environ.get("BIESL_USE_GPU", "0") == "1"
    kwargs = dict(
        random_state=0, n_jobs=-1, eval_metric="mlogloss",
        tree_method="hist", device=("cuda" if use_gpu else "cpu"),
    )
    return Pipeline([("clf", SafeXGBClassifier(**kwargs))])


KNN_GRID = {
    "selector__k":   [10, 15, "all"],
    "clf__n_neighbors": [3, 5, 7, 11, 15, 21],
    "clf__weights":     ["uniform", "distance"],
    "clf__metric":      ["euclidean", "manhattan"],
}

RF_GRID = {
    "clf__n_estimators":  [200, 500],
    "clf__max_depth":     [None, 6, 12],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__max_features":  ["sqrt", 0.5],
}

XGB_GRID = {
    "clf__n_estimators":  [200, 500],
    "clf__max_depth":     [3, 6],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample":     [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
}

MODELS = {
    "knn":          (make_knn_pipeline, KNN_GRID, False),
    "randomforest": (make_rf_pipeline,  RF_GRID,  False),
    "xgboost":      (make_xgb_pipeline, XGB_GRID, False),   # SafeXGBClassifier handles labels internally
}


# kept for backwards compatibility (callers used to import make_pipeline / GRID)
make_pipeline = make_knn_pipeline
GRID = KNN_GRID


def _loso_one_model(
    model_name: str, build_pipe, grid: dict, needs_int_labels: bool,
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, recs: np.ndarray,
) -> dict:
    """Run LOSO evaluation for a single model. Returns aggregate dict."""
    logo = LeaveOneGroupOut()
    cm_total = np.zeros((len(PHASE_CLASSES), len(PHASE_CLASSES)), dtype=int)
    fold_reports: list[dict] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups)):
        train_subject = list(set(groups[tr_idx]))
        test_subject  = list(set(groups[te_idx]))
        print(f"  [{model_name}] fold {fold_idx + 1}: train={train_subject} test={test_subject}", end="")

        X_tr, y_tr_raw, groups_tr = X[tr_idx], y[tr_idx], recs[tr_idx]
        X_te, y_te_raw = X[te_idx], y[te_idx]

        if needs_int_labels:
            le_fold = LabelEncoder().fit(y_tr_raw)
            y_tr = le_fold.transform(y_tr_raw)
        else:
            le_fold = None
            y_tr = y_tr_raw

        n_recs_in_train = len(set(groups_tr))
        if n_recs_in_train >= 3:
            cv_splits = list(GroupKFold(n_splits=min(n_recs_in_train, 5)).split(X_tr, y_tr, groups_tr))
        else:
            from sklearn.model_selection import StratifiedKFold
            cv_splits = StratifiedKFold(n_splits=min(3, n_recs_in_train + 1),
                                        shuffle=True, random_state=0)

        gs = GridSearchCV(
            build_pipe(), grid, cv=cv_splits, scoring="f1_macro",
            n_jobs=-1, refit=True, error_score="raise",
        )
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)

        # decode back to string labels for the report
        if le_fold is not None:
            y_pred_lbl = le_fold.inverse_transform(y_pred)
        else:
            y_pred_lbl = y_pred
        y_te_lbl = y_te_raw

        acc = accuracy_score(y_te_lbl, y_pred_lbl)
        f1m = f1_score(y_te_lbl, y_pred_lbl, average="macro", zero_division=0)
        cm = confusion_matrix(y_te_lbl, y_pred_lbl, labels=PHASE_CLASSES)
        cm_total += cm
        report = classification_report(
            y_te_lbl, y_pred_lbl, labels=PHASE_CLASSES, zero_division=0, output_dict=True,
        )
        print(f"  inner CV f1m={gs.best_score_:.3f}  test acc={acc:.3f}  test f1m={f1m:.3f}")

        fold_reports.append(dict(
            fold=fold_idx + 1, train=train_subject, test=test_subject,
            inner_cv_f1m=float(gs.best_score_),
            accuracy=float(acc), macro_f1=float(f1m),
            best_params={k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                         for k, v in gs.best_params_.items()},
            confusion=cm.tolist(), report=report,
        ))

    return dict(
        model=model_name,
        folds=fold_reports,
        confusion_total=cm_total.tolist(),
        mean_accuracy=float(np.mean([f["accuracy"] for f in fold_reports])),
        mean_macro_f1=float(np.mean([f["macro_f1"] for f in fold_reports])),
    )


def make_train_val_test_split(
    df: pd.DataFrame,
    val_recordings: tuple[str, ...] = ("nvt-5-8-medi",),
    test_recordings: tuple[str, ...] = ("nvt-5-15-medi",),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recording-grouped train/val/test split.

    Default: train = 5 mta recordings (only set containing plank), val = one
    nvt recording, test = the other nvt recording. Cross-subject test set.

    Returns (train_df, val_df, test_df) with no recording overlap.
    """
    val_set  = set(val_recordings)
    test_set = set(test_recordings)
    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"val and test share recordings: {overlap}")

    train_df = df[~df["rec_name"].isin(val_set | test_set)].copy()
    val_df   = df[df["rec_name"].isin(val_set)].copy()
    test_df  = df[df["rec_name"].isin(test_set)].copy()

    if val_df.empty or test_df.empty:
        raise ValueError(
            f"empty val/test split. Available recordings: {sorted(df['rec_name'].unique())}"
        )
    return train_df, val_df, test_df


def _print_split_summary(train_df, val_df, test_df) -> None:
    print("\n--- Split summary ---")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        recs = sorted(d["rec_name"].unique())
        acts = Counter(d["activity"])
        subs = Counter(d["subject"])
        print(f"  {name:5s} n={len(d):3d}  subjects={dict(subs)}  activities={dict(acts)}")
        for r in recs:
            print(f"          - {r}")


def stratified_window_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pool all windows, stratify-random into train/val/test by activity label.

    WARNING: windows are 30 s with 50 % overlap, so adjacent windows in the
    same recording are highly correlated. This split leaks across that
    overlap and across subject baselines — scores will be optimistic relative
    to recording-grouped or subject-grouped splits.
    """
    from sklearn.model_selection import train_test_split

    rng = seed
    y = df["activity"].to_numpy()
    idx_tv, idx_te = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=y, random_state=rng,
    )
    y_tv = y[idx_tv]
    val_relative = val_size / (1.0 - test_size)
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=val_relative, stratify=y_tv, random_state=rng,
    )
    return df.iloc[idx_tr].copy(), df.iloc[idx_va].copy(), df.iloc[idx_te].copy()


def train_val_test_evaluate(
    df: pd.DataFrame,
    out_dir: str = OUT_DIR,
    models: list[str] | None = None,
    *,
    mode: str = "window_stratified",
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 0,
    val_recordings: tuple[str, ...] = ("nvt-5-8-medi",),
    test_recordings: tuple[str, ...] = ("nvt-5-15-medi",),
) -> dict:
    """Train / val / test evaluation.

    mode = "window_stratified"  -> pool all windows, stratified random split
                                    (LEAKS across overlap; optimistic).
    mode = "recording_grouped"  -> split whole recordings into train/val/test
                                    (honest, but limited by 7-recording total).

    Hyperparameter search uses a single train/val split (PredefinedSplit).
    Final scores are reported on BOTH val and the held-out test set.
    """
    from sklearn.model_selection import PredefinedSplit
    os.makedirs(out_dir, exist_ok=True)

    if mode == "window_stratified":
        train_df, val_df, test_df = stratified_window_split(
            df, val_size=val_size, test_size=test_size, seed=seed,
        )
        print("\n!! SPLIT MODE: window-stratified random - leaks across the 50%-overlap")
        print("   windows in each recording. Scores are optimistic vs. recording-grouped CV.")
    elif mode == "recording_grouped":
        train_df, val_df, test_df = make_train_val_test_split(
            df, val_recordings=val_recordings, test_recordings=test_recordings,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    _print_split_summary(train_df, val_df, test_df)

    # Build (X, y) per set using the same feature columns
    X_tr, y_tr, _, feat_cols = split_xy(train_df)
    X_va, y_va, _, _         = split_xy(val_df)
    X_te, y_te, _, _         = split_xy(test_df)

    # PredefinedSplit: -1 for train rows, 0 for val rows (means single fold)
    X_tv = np.vstack([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    test_fold = np.concatenate([np.full(len(X_tr), -1, dtype=int),
                                np.zeros(len(X_va), dtype=int)])
    ps = PredefinedSplit(test_fold)

    chosen = models or list(MODELS.keys())
    all_results: dict[str, dict] = {}
    for name in chosen:
        build_pipe, grid, needs_int = MODELS[name]
        print(f"\n>>> Model: {name}")

        if needs_int:
            le = LabelEncoder().fit(y_tv)
            y_tv_use = le.transform(y_tv)
        else:
            le = None
            y_tv_use = y_tv

        gs = GridSearchCV(
            build_pipe(), grid, cv=ps, scoring="f1_macro",
            n_jobs=-1, refit=True, error_score="raise",
        )
        gs.fit(X_tv, y_tv_use)
        best_params = gs.best_params_
        # NOTE: refit uses train+val combined. For an honest val score we
        # additionally refit on train ONLY and re-score on val.
        train_only_pipe = build_pipe()
        train_only_pipe.set_params(**best_params)
        train_only_pipe.fit(X_tr, y_tv_use[:len(X_tr)])
        y_va_pred = train_only_pipe.predict(X_va)
        y_te_pred = gs.predict(X_te)

        if le is not None:
            y_va_pred_lbl = le.inverse_transform(y_va_pred)
            y_te_pred_lbl = le.inverse_transform(y_te_pred)
        else:
            y_va_pred_lbl = y_va_pred
            y_te_pred_lbl = y_te_pred

        def _metrics(y_true_lbl, y_pred_lbl):
            return dict(
                accuracy = float(accuracy_score(y_true_lbl, y_pred_lbl)),
                macro_f1 = float(f1_score(y_true_lbl, y_pred_lbl,
                                          average="macro", zero_division=0)),
                report   = classification_report(y_true_lbl, y_pred_lbl,
                              labels=PHASE_CLASSES, zero_division=0,
                              output_dict=True),
                confusion = confusion_matrix(y_true_lbl, y_pred_lbl,
                              labels=PHASE_CLASSES).tolist(),
            )

        val_m  = _metrics(y_va, y_va_pred_lbl)
        test_m = _metrics(y_te, y_te_pred_lbl)
        print(f"  best params: {best_params}")
        print(f"  val  acc={val_m['accuracy']:.3f}  macroF1={val_m['macro_f1']:.3f}")
        print(f"  test acc={test_m['accuracy']:.3f}  macroF1={test_m['macro_f1']:.3f}")
        for c in PHASE_CLASSES:
            tr = test_m["report"][c]
            print(f"    [test] {c:10s} P={tr['precision']:.3f} R={tr['recall']:.3f} "
                  f"F1={tr['f1-score']:.3f} support={int(tr['support'])}")

        all_results[name] = dict(
            best_params={k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                         for k, v in best_params.items()},
            val=val_m, test=test_m,
        )

    out = dict(
        classes=PHASE_CLASSES, feature_names=feat_cols,
        split=dict(
            train_recordings=sorted(train_df["rec_name"].unique()),
            val_recordings=list(val_recordings),
            test_recordings=list(test_recordings),
            counts=dict(train=len(train_df), val=len(val_df), test=len(test_df)),
        ),
        results=all_results,
    )
    with open(os.path.join(out_dir, "tvt_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n========== Model comparison (recording-grouped train/val/test) ==========")
    header = " / ".join(PHASE_CLASSES)
    print(f"{'model':<14} {'val_acc':>7} {'val_F1':>7} {'test_acc':>8} {'test_F1':>7}  "
          f"test per-class F1 ({header})")
    for name, r in all_results.items():
        f1s = " / ".join(f"{r['test']['report'][c]['f1-score']:.3f}" for c in PHASE_CLASSES)
        print(f"{name:<14} {r['val']['accuracy']:>7.3f} {r['val']['macro_f1']:>7.3f} "
              f"{r['test']['accuracy']:>8.3f} {r['test']['macro_f1']:>7.3f}  {f1s}")
    return out


def loso_evaluate(df: pd.DataFrame, out_dir: str = OUT_DIR,
                  models: list[str] | None = None) -> dict:
    """LOSO evaluation across all configured models. Writes loso_results.json
    plus a per-model summary table."""
    os.makedirs(out_dir, exist_ok=True)
    X, y, groups, feat_cols = split_xy(df)
    recs = df["rec_name"].to_numpy()
    chosen = models or list(MODELS.keys())

    all_results: dict[str, dict] = {}
    for name in chosen:
        build_pipe, grid, needs_int = MODELS[name]
        print(f"\n>>> Model: {name}")
        all_results[name] = _loso_one_model(name, build_pipe, grid, needs_int,
                                            X, y, groups, recs)

    out = {
        "classes": PHASE_CLASSES,
        "feature_names": feat_cols,
        "results": all_results,
    }
    with open(os.path.join(out_dir, "loso_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # Side-by-side summary
    print("\n========== Model comparison (LOSO mean across subjects) ==========")
    header = " / ".join(PHASE_CLASSES)
    print(f"{'model':<14} {'acc':>6} {'macroF1':>8}  per-class F1 ({header})")
    for name, r in all_results.items():
        per_cls = {c: [] for c in PHASE_CLASSES}
        for f in r["folds"]:
            for c in PHASE_CLASSES:
                per_cls[c].append(f["report"][c]["f1-score"])
        f1s = " / ".join(f"{np.mean(per_cls[c]):.3f}" for c in PHASE_CLASSES)
        print(f"{name:<14} {r['mean_accuracy']:>6.3f} {r['mean_macro_f1']:>8.3f}  {f1s}")

    return out


def top_features_by_mi(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Mutual-information feature ranking on the full (imputed, scaled) dataset.

    Reported only for reference; not used during model selection.
    """
    X, y, _, feat_cols = split_xy(df)
    Ximp = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_classif(Ximp, y, random_state=0)
    ranking = pd.DataFrame({"feature": feat_cols, "MI": mi}).sort_values("MI", ascending=False)
    return ranking.head(k).reset_index(drop=True)


def fit_final_models(df: pd.DataFrame, out_dir: str = OUT_DIR) -> dict:
    """Fit each model on ALL data with subject-grouped grid search; save artifacts."""
    os.makedirs(out_dir, exist_ok=True)
    X, y, groups, feat_cols = split_xy(df)
    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(n_groups, 5))
    saved = {}
    for name, (build_pipe, grid, needs_int) in MODELS.items():
        if needs_int:
            le = LabelEncoder().fit(np.unique(y))
            y_use = le.transform(y)
        else:
            y_use = y
        gs = GridSearchCV(
            build_pipe(), grid, cv=cv.split(X, y_use, groups), scoring="f1_macro",
            n_jobs=-1, refit=True, error_score="raise",
        )
        gs.fit(X, y_use)
        path = os.path.join(out_dir, f"{name}_model.joblib")
        joblib.dump(gs.best_estimator_, path)
        saved[name] = dict(path=path, best_params=gs.best_params_)
        print(f"  {name:<14} best: {gs.best_params_}  -> {path}")
    joblib.dump({"feature_names": feat_cols, "classes": PHASE_CLASSES},
                os.path.join(out_dir, "feature_names.joblib"))
    return saved


# back-compat alias
fit_final_model = fit_final_models


def main():
    print("[1/4] Building feature table ...")
    df = build_feature_table()
    n_meta = 7
    print(f"\nFeature table: {df.shape[0]} windows x {df.shape[1] - n_meta} features")
    print("Activity     :", Counter(df["activity"]))
    print("Phase        :", Counter(df["phase"]))
    print("Subjects     :", Counter(df["subject"]))
    print("Stressors    :", Counter(df["stressor"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    feat_csv = os.path.join(OUT_DIR, "features.csv")
    df.to_csv(feat_csv, index=False)
    print(f"  -> wrote {feat_csv}")

    print("\n[2/4] Top features by mutual information:")
    print(top_features_by_mi(df, k=12).to_string(index=False))

    print("\n[3a/4] Window-stratified random train/val/test evaluation ...")
    train_val_test_evaluate(df, mode="window_stratified")

    print("\n[3b/4] LOSO evaluation (full subject hold-out, for comparison) ...")
    loso_evaluate(df)

    print("\n[4/4] Fitting final models on all data ...")
    fit_final_models(df)


if __name__ == "__main__":
    main()
