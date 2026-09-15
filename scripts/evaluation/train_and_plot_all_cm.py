#!/usr/bin/env python
"""Train KNN, Random Forest, XGBoost, and 1D-CNN on 9 features/signals with exclusions,
compute confusion matrices, and generate individual plots in the original look.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.exclusions import filter_feature_df, filter_raw_windows
from src.pipeline import build_feature_table, SafeXGBClassifier
from src.raw_windows import local_raw_windows, stack_windows, LABEL_NAMES
from src.dl_models import BiosignalCNN1D

CLASSES = ["rest", "mental1", "mental2", "physical1", "physical2"]
FINAL_FEATS = ["csi", "hr", "hrv_rmssd", "mic_rms", "resp_amp", "rr", "rrv", "sd2_sd1", "ss"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# ---------------------------------------------------------------------------
# Setup Seeds
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_set_seed(SEED)

# ---------------------------------------------------------------------------
# Dataset & Trainer for 1D-CNN
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, *, augment: bool = False):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.augment:
            shift = int(np.random.randint(-125, 126))
            x = np.roll(x, shift, axis=-1)
            n_ch = x.shape[0]
            scale = (1.0 + 0.05 * (2 * np.random.rand(n_ch, 1) - 1)).astype(np.float32)
            x = x * scale
            x = x + (0.05 * np.random.randn(*x.shape)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


def _class_weights(y: np.ndarray, n_classes: int = len(CLASSES)) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for c, n in Counter(y.tolist()).items():
        counts[c] = n
    counts = np.where(counts > 0, counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(CLASSES))), zero_division=0))


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    in_channels: int,
    epochs: int = 80,
    patience: int = 12,
    batch_size: int = 32,
    augment: bool = True,
) -> tuple[BiosignalCNN1D, dict]:
    model = BiosignalCNN1D(in_channels=in_channels, n_classes=len(CLASSES)).to(DEVICE)
    weights = _class_weights(y_train, len(CLASSES))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    train_loader = DataLoader(
        WindowDataset(X_train, y_train, augment=augment),
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(X_val, y_val, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
            tl += float(loss.item()) * len(xb)
            n += len(xb)
        sched.step()
        history["train_loss"].append(tl / max(n, 1))

        model.eval()
        vl = 0.0
        n = 0
        preds: list[int] = []
        truths: list[int] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                vl += float(loss.item()) * len(xb)
                n += len(xb)
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                truths.extend(yb.cpu().tolist())
        history["val_loss"].append(vl / max(n, 1))
        f1 = _macro_f1(np.array(truths), np.array(preds))
        history["val_macro_f1"].append(f1)

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_state = copy_best_state(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_macro_f1"] = float(best_f1)
    history["stopped_at_epoch"] = len(history["train_loss"])
    return model, history


def copy_best_state(state):
    import copy
    return copy.deepcopy(state)


@torch.no_grad()
def predict_dl(model: BiosignalCNN1D, X: np.ndarray) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        WindowDataset(X, np.zeros(len(X), dtype=np.int64), augment=False),
        batch_size=32, shuffle=False,
    )
    preds: list[int] = []
    for xb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return np.array(preds, dtype=np.int64)

# ---------------------------------------------------------------------------
# Classical Models Factory
# ---------------------------------------------------------------------------
def _make_knn() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")),
    ])


def _make_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            max_features="sqrt", random_state=0, n_jobs=-1,
        )),
    ])


def _make_xgb() -> SafeXGBClassifier:
    return SafeXGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        eval_metric="mlogloss", tree_method="hist", device="cpu",
        class_weight="balanced"
    )

# ---------------------------------------------------------------------------
# Helper: Pad Missing Classes for Folds
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helper: Save Confusion Matrix with specific "original look"
# ---------------------------------------------------------------------------
def save_original_look_cm(cm: np.ndarray, title: str, filename: str, supports: np.ndarray):
    # Row normalized %
    cm_pct = cm.astype(float)
    row_sums = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = cm_pct / np.where(row_sums == 0, 1.0, row_sums) * 100.0
    
    fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor="white")
    ax.set_facecolor("white")
    
    # original color scale ranges 0 to 100, Blues cmap
    im = ax.imshow(cm_pct, cmap="Blues", aspect="equal", vmin=0, vmax=100)
    
    # Tick marks
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    
    # labels
    class_labels_x = [c for c in CLASSES]
    class_labels_y = [f"{c}\n(n={supports[i]:,})" for i, c in enumerate(CLASSES)]
    
    ax.set_xticklabels(class_labels_x, color="black", rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels_y, color="black", fontsize=9)
    
    ax.set_xlabel("predicted", color="black", fontsize=10, labelpad=8)
    ax.set_ylabel("true", color="black", fontsize=10, labelpad=8)
    ax.set_title(title, color="black", fontsize=10, pad=12)
    
    # Draw numbers inside cells (rounded integer % to match original)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = cm_pct[i, j]
            # Match the text color threshold of original
            color = "white" if v >= 50.0 else "black"
            ax.text(
                j, i, 
                f"{int(round(v))}%", 
                ha="center", 
                va="center", 
                fontsize=9, 
                color=color
            )
            
    # Set borders/spines to match original (black outlines)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        
    ax.tick_params(colors="black", length=3)
    
    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="black")
    cb.ax.tick_params(colors="black", labelsize=8)
    cb.set_label("% of true class", color="black", fontsize=9)
    cb.outline.set_edgecolor("black")
    
    plt.tight_layout()
    
    out_dir = Path("figures/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved confusion matrix plot to -> {out_path}")

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(" 1. Classical Models LORO Cross-Validation (9 features)")
    print("=" * 80)
    df_raw = build_feature_table(include_temp=False)
    df_raw = filter_feature_df(df_raw)
    df = df_raw[df_raw["activity"].isin(CLASSES)].copy()
    
    le = LabelEncoder().fit(CLASSES)
    recs = sorted(df["rec_name"].unique())
    
    # Store predictions for classical models
    y_true_all = []
    y_preds = {m: [] for m in ["knn", "randomforest", "xgboost"]}
    
    for r in recs:
        test_df = df[df.rec_name == r]
        train_df = df[df.rec_name != r]
        if len(test_df) == 0:
            continue
            
        train_df = _pad_missing(train_df, FINAL_FEATS, CLASSES)
        y_true_all.extend(test_df["activity"].tolist())
        
        # KNN
        knn = _make_knn()
        knn.fit(train_df[FINAL_FEATS].to_numpy(float), le.transform(train_df["activity"]))
        knn_pred = knn.predict(test_df[FINAL_FEATS].to_numpy(float))
        y_preds["knn"].extend(le.inverse_transform(knn_pred))
        
        # RF
        rf = _make_rf()
        rf.fit(train_df[FINAL_FEATS].to_numpy(float), le.transform(train_df["activity"]))
        rf_pred = rf.predict(test_df[FINAL_FEATS].to_numpy(float))
        y_preds["randomforest"].extend(le.inverse_transform(rf_pred))
        
        # XGBoost
        xgb = _make_xgb()
        xgb.fit(train_df[FINAL_FEATS].to_numpy(float), train_df["activity"].to_numpy())
        xgb_pred = xgb.predict(test_df[FINAL_FEATS].to_numpy(float))
        y_preds["xgboost"].extend(xgb_pred)
        
    y_true_all = np.array(y_true_all)
    supports = np.zeros(len(CLASSES), dtype=int)
    for i, c in enumerate(CLASSES):
        supports[i] = (y_true_all == c).sum()
        
    for m in ["knn", "randomforest", "xgboost"]:
        cm = confusion_matrix(y_true_all, y_preds[m], labels=CLASSES)
        acc = accuracy_score(y_true_all, y_preds[m])
        f1 = f1_score(y_true_all, y_preds[m], average="macro", zero_division=0)
        
        model_title = {
            "knn": "KNN LORO - 9-Feature",
            "randomforest": "RandomForest LORO - 9-Feature",
            "xgboost": "XGBoost LORO - 9-Feature"
        }[m]
        
        print(f"\n{model_title}")
        print(f"  Accuracy: {acc:.4f}   Macro F1: {f1:.4f}")
        print(cm)
        
        save_original_look_cm(
            cm=cm,
            title=model_title,
            filename=f"cm_9feat_{m}.png",
            supports=supports
        )
        
    print("\nSkipping 1D-CNN LORO Cross-Validation as requested.")


if __name__ == "__main__":
    main()

