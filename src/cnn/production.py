"""Fit and serialize the CNN used after cross-validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import CHANNELS, CLASSES, TARGET_FS, WINDOW_SAMPLES, WINDOW_SECONDS
from .model import Phase2CNN1D
from .training import train_fixed_epochs


def recommended_epochs(evaluation: dict) -> int:
    epochs = [fold["best_epoch"] for fold in evaluation["folds"] if fold["best_epoch"] > 0]
    return max(1, int(round(float(np.median(epochs)))))


def train_and_save(dataset: dict[str, np.ndarray], output_dir: str | Path, *, epochs: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = train_fixed_epochs(dataset["X"], dataset["y"], n_classes=len(CLASSES), epochs=epochs)
    model_path = output_dir / "cnn_model.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, model_path)
    metadata = {
        "model_class": Phase2CNN1D.__name__,
        "classes": list(CLASSES),
        "channels": list(CHANNELS),
        "sampling_rate_hz": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "epochs": epochs,
        "n_windows": len(dataset["y"]),
        "n_recordings": len(set(dataset["recordings"].astype(str).tolist())),
        "state_dict": model_path.name,
    }
    (output_dir / "cnn_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path
