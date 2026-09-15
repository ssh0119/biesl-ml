"""Recording-grouped evaluation and reporting for Plan C."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import CLASSES, FEATURES
from .model import create_model


def _pad_missing_classes(train_df: pd.DataFrame) -> pd.DataFrame:
    missing = set(CLASSES) - set(train_df["activity"].unique())
    if not missing:
        return train_df
    rows = [
        {
            **{feature: np.nan for feature in FEATURES},
            "rec_name": f"_pad_{label}",
            "subject": "_pad",
            "stressor": "_pad",
            "phase": label,
            "activity": label,
            "t_start": 0.0,
            "t_end": 0.0,
        }
        for label in missing
    ]
    return pd.concat([train_df, pd.DataFrame(rows)], ignore_index=True)


def evaluate_loro(df: pd.DataFrame) -> dict:
    """Evaluate one held-out recording per fold and pool all predictions."""
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    folds: list[dict] = []

    for rec_name in sorted(df["rec_name"].unique()):
        test_df = df[df["rec_name"] == rec_name]
        train_df = _pad_missing_classes(df[df["rec_name"] != rec_name])

        model = create_model()
        model.fit(train_df[list(FEATURES)].to_numpy(float), train_df["activity"].to_numpy())
        y_true = test_df["activity"].to_numpy()
        y_pred = model.predict(test_df[list(FEATURES)].to_numpy(float))
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
        folds.append({
            "recording": rec_name,
            "n_windows": int(len(test_df)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1_present_classes": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        })

    y_true = np.asarray(y_true_all)
    y_pred = np.asarray(y_pred_all)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    per_f1 = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    return {
        "protocol": "leave-one-recording-out",
        "classes": list(CLASSES),
        "features": list(FEATURES),
        "n_recordings": int(df["rec_name"].nunique()),
        "n_windows": int(len(y_true)),
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(CLASSES, map(float, per_f1))),
        "per_class_precision": dict(zip(CLASSES, map(float, per_precision))),
        "per_class_recall": dict(zip(CLASSES, map(float, per_recall))),
        "confusion_matrix": cm.tolist(),
        "folds": folds,
    }


def save_evaluation(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    figure_path = output_dir / "confusion_matrix.png"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot_confusion_matrix(result, figure_path)
    return metrics_path, figure_path


def _plot_confusion_matrix(result: dict, path: Path) -> None:
    cm = np.asarray(result["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = cm / np.where(row_sums == 0, 1, row_sums) * 100.0
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{int(cm[i, j])}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if pct[i, j] >= 50 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Plan C XGBoost LORO\nAccuracy {result['pooled_accuracy']:.1%} | "
        f"Macro F1 {result['pooled_macro_f1']:.3f}"
    )
    fig.colorbar(image, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
