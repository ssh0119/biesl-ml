"""Build and cache aligned raw windows for CNN evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import list_recordings, load_recording
from ..raw_windows import windows_from_local_recording
from .config import DOWNSAMPLE_FACTOR, WINDOW_SAMPLES


def build_raw_dataset(data_dir: str = "data") -> dict[str, np.ndarray]:
    signals: list[np.ndarray] = []
    labels: list[int] = []
    recordings: list[str] = []
    subjects: list[str] = []
    anchors: list[float] = []

    for path in list_recordings(data_dir):
        windows = windows_from_local_recording(load_recording(path), include_temp=False)
        for window in windows:
            signal = window.signal[:, ::DOWNSAMPLE_FACTOR]
            if signal.shape[-1] != WINDOW_SAMPLES:
                raise ValueError(f"unexpected signal shape {signal.shape} for {path}")
            signals.append(signal.astype(np.float32, copy=False))
            labels.append(window.label)
            recordings.append(window.rec_name)
            subjects.append(window.subject)
            anchors.append(window.t_start)
        print(f"  CNN {Path(path).name:34s} -> {len(windows):3d} windows")

    return {
        "X": np.stack(signals).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "recordings": np.asarray(recordings),
        "subjects": np.asarray(subjects),
        "anchors": np.asarray(anchors, dtype=np.float64),
    }


def save_cache(dataset: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dataset)
    return path


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}
