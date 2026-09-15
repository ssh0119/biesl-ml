"""Leakage-conscious recording-level cross-validation for the Phase 2 CNN."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import CLASSES, MAX_EPOCHS, PATIENCE
from .training import DEVICE, predict_probabilities, train_with_validation


def _phase_label(y: np.ndarray) -> int:
    non_rest = sorted(set(y.tolist()) - {0})
    if len(non_rest) != 1:
        raise ValueError(f"a recording must contain rest and exactly one Phase 2 label: {non_rest}")
    return non_rest[0]


def select_validation_recordings(recordings: np.ndarray, y: np.ndarray, test_recording: str) -> list[str]:
    """Pick one non-test recording per Phase 2 class for inner validation."""
    selected: list[str] = []
    all_recordings = sorted(set(recordings.tolist()))
    test_index = all_recordings.index(test_recording)
    candidates = set(all_recordings) - {test_recording}
    for label in range(1, len(CLASSES)):
        matches = sorted(name for name in candidates if _phase_label(y[recordings == name]) == label)
        if not matches:
            raise ValueError(f"no validation recording available for class {CLASSES[label]}")
        selected.append(matches[test_index % len(matches)])
    return selected


def _plot_confusion(cm: np.ndarray, output_path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    percent = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            color = "white" if percent[row, col] >= 55 else "black"
            ax.text(col, row, f"{percent[row, col]:.0f}%\n({cm[row, col]})", ha="center", va="center", color=color)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), [f"{name}\n(n={cm[i].sum()})" for i, name in enumerate(CLASSES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("1D-CNN LORO (Phase 2 folder labels, 250 Hz)")
    fig.colorbar(image, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_loro(
    dataset: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    test_recordings: list[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = dataset["X"], dataset["y"]
    recordings = dataset["recordings"].astype(str)
    all_recordings = sorted(set(recordings.tolist()))
    unique_recordings = test_recordings or all_recordings
    unknown = sorted(set(unique_recordings) - set(all_recordings))
    if unknown:
        raise ValueError(f"unknown test recordings: {unknown}")
    all_true: list[int] = []
    all_pred: list[int] = []
    prediction_frames: list[pd.DataFrame] = []
    folds: list[dict] = []

    print(f"device={DEVICE}, input={tuple(X.shape)}, folds={len(unique_recordings)}")
    for fold_index, test_recording in enumerate(unique_recordings, 1):
        val_recordings = select_validation_recordings(recordings, y, test_recording)
        test_mask = recordings == test_recording
        val_mask = np.isin(recordings, val_recordings)
        train_mask = ~(test_mask | val_mask)
        print(f"[{fold_index:02d}/{len(unique_recordings)}] test={test_recording} val={','.join(val_recordings)}")

        model, training = train_with_validation(
            X[train_mask], y[train_mask], X[val_mask], y[val_mask],
            n_classes=len(CLASSES), max_epochs=max_epochs, patience=patience,
        )
        probability = predict_probabilities(model, X[test_mask])
        prediction = probability.argmax(axis=1)
        true = y[test_mask]
        present = sorted(set(true.tolist()))
        fold_result = {
            "test_recording": test_recording,
            "validation_recordings": val_recordings,
            "n_train": int(train_mask.sum()),
            "n_validation": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "accuracy": float(accuracy_score(true, prediction)),
            "macro_f1_present_classes": float(f1_score(true, prediction, labels=present, average="macro", zero_division=0)),
            **training,
        }
        folds.append(fold_result)
        print(f"  accuracy={fold_result['accuracy']:.4f} f1={fold_result['macro_f1_present_classes']:.4f} best_epoch={training['best_epoch']}")

        frame = pd.DataFrame({
            "recording": recordings[test_mask],
            "anchor_seconds": dataset["anchors"][test_mask],
            "true_label": [CLASSES[index] for index in true],
            "predicted_label": [CLASSES[index] for index in prediction],
        })
        for class_index, name in enumerate(CLASSES):
            frame[f"prob_{name}"] = probability[:, class_index]
        prediction_frames.append(frame)
        all_true.extend(true.tolist())
        all_pred.extend(prediction.tolist())

    true_array = np.asarray(all_true)
    pred_array = np.asarray(all_pred)
    cm = confusion_matrix(true_array, pred_array, labels=range(len(CLASSES)))
    report = classification_report(
        true_array, pred_array, labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    result = {
        "protocol": "Leave-One-Recording-Out with four-recording inner validation",
        "device": DEVICE,
        "classes": list(CLASSES),
        "n_test_recordings": len(unique_recordings),
        "n_test_windows": len(true_array),
        "accuracy": float(accuracy_score(true_array, pred_array)),
        "macro_f1": float(f1_score(true_array, pred_array, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "folds": folds,
    }
    (output_dir / "loro_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "loro_predictions.csv", index=False)
    _plot_confusion(cm, output_dir / "confusion_matrix.png")
    return result
