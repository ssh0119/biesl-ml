"""Raw windowed signals for the 1D-CNN.

Produces fixed-length float32 tensors, one per anchor on the same 2-s anchor
grid the classical features use (see `src.features`). Each window is `WINDOW_S`
seconds long (40 s) CENTERED on its anchor:
    channel 0: ECG, LP-150-Hz filtered + detrended, 250 Hz
    channel 1: Respiration, median-baseline + 0.5-s smoothing, 250 Hz
    channel 2: Microphone Shannon-energy envelope, 250 Hz
    channel 3: Skin temperature linearly upsampled to 250 Hz  (only when include_temp=True)

Per-recording robust z-score per channel.

Local-only — WESAD inputs are intentionally omitted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .features import (
    ANCHOR_STEP_S,
    MAX_FEATURE_WINDOW_S,
    WINDOW_S,
    _anchor_in_boundary,
    preprocess_recording,
)
from .io import (
    Recording,
    list_recordings,
    load_recording,
    phase_boundaries,
    resample_uniform,
)
from .pipeline import PHASE_CLASSES, assign_activity, phase2_label_from_path

TARGET_FS = 250.0
ACTIVITY_TO_LABEL = {name: i for i, name in enumerate(PHASE_CLASSES)}
LABEL_TO_ACTIVITY = {v: k for k, v in ACTIVITY_TO_LABEL.items()}
LABEL_NAMES = tuple(PHASE_CLASSES)
WINDOW_N = int(WINDOW_S * TARGET_FS)


@dataclass
class RawWindow:
    signal: np.ndarray       # (n_channels, WINDOW_N) float32
    label: int               # index into pipeline.PHASE_CLASSES
    activity: str
    source: str              # 'local'
    subject: str
    rec_name: str
    t_start: float


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Std-based z-score with ±8 sigma clip. Same as the previous run — this
    is for CNN-input scaling, not for QRS-peak emphasis."""
    x = x.astype(np.float64, copy=False)
    mean = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return (x - mean).astype(np.float32)
    z = (x - mean) / sd
    return np.clip(z, -8.0, 8.0).astype(np.float32)


def _upsample_linear(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def _align_to_length(x: np.ndarray, n_target: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad or crop so the array has length `n_target`."""
    if len(x) == n_target:
        return x.astype(np.float32, copy=False)
    if len(x) < n_target:
        pad = np.full(n_target - len(x), pad_value, dtype=np.float32)
        return np.concatenate([x, pad]).astype(np.float32)
    return x[:n_target].astype(np.float32)


def windows_from_local_recording(
    rec: Recording, *, include_temp: bool = False,
) -> list[RawWindow]:
    """Build per-window 3-channel (or 4-channel with temp) tensors."""
    pp = preprocess_recording(rec)

    # Resample everything to TARGET_FS and z-score per channel over the WHOLE
    # recording. This is the cross-device-friendly input scaling.
    ecg_250 = _robust_z(_upsample_linear(pp.ecg, pp.fs_ecg, TARGET_FS))
    resp_250 = _robust_z(_upsample_linear(pp.br, pp.fs_br, TARGET_FS))
    mic_250 = _robust_z(_upsample_linear(pp.se, pp.fs_mic, TARGET_FS))

    # ECG is the longest by sample count; clip the others to its length so
    # all channels share the same sample grid (they're aligned at t=0 by
    # construction since preprocess_recording uses resample_uniform).
    n_target = min(len(ecg_250), len(resp_250), len(mic_250))
    ecg_250 = ecg_250[:n_target]
    resp_250 = resp_250[:n_target]
    mic_250 = mic_250[:n_target]

    if include_temp:
        temp_uniform = resample_uniform(rec.channels["temp"], TARGET_FS)
        temp_250 = _align_to_length(
            _robust_z(temp_uniform.astype(np.float64)),
            n_target,
            pad_value=0.0,
        )

    phases = phase_boundaries(rec)
    phase2_label = phase2_label_from_path(rec.path)
    if phase2_label is None:
        raise ValueError(f"recording is not inside a supported Phase 2 label folder: {rec.path}")
    out: list[RawWindow] = []
    ecg_t0 = float(rec.channels["ecg"][0, 0])

    # Anchor-based windowing — same anchor grid as the classical features
    # (2-s step, 60-s longest feature window). CNN raw windows are 40 s
    # CENTERED on each anchor. We also require the classical-pipeline
    # safety margin (`MAX_FEATURE_WINDOW_S/2` from each phase boundary) so
    # the CNN training set is on the same rows as the feature table.
    half_max_classical = MAX_FEATURE_WINDOW_S / 2.0
    half_cnn = WINDOW_S / 2.0

    for phase_name, (p_start, p_end) in phases.items():
        if phase_name == "recovery":
            continue
        activity = assign_activity(phase_name, rec.stressor, phase2_label)
        if activity not in ACTIVITY_TO_LABEL:
            continue
        anchor_lo = p_start + half_max_classical
        anchor_hi = p_end - half_max_classical
        if anchor_hi < anchor_lo:
            continue
        n_anchors = int(np.floor((anchor_hi - anchor_lo) / ANCHOR_STEP_S)) + 1
        anchors = anchor_lo + ANCHOR_STEP_S * np.arange(n_anchors)

        for abs_t in anchors:
            abs_t = float(abs_t)
            if _anchor_in_boundary(abs_t):
                continue
            # 40-s CNN window centered on the anchor, in ECG-local time.
            t_local_lo = (abs_t - half_cnn) - ecg_t0
            a = int(round(t_local_lo * TARGET_FS))
            b = a + WINDOW_N
            if a < 0 or b > n_target:
                continue
            chs = [ecg_250[a:b], resp_250[a:b], mic_250[a:b]]
            if include_temp:
                chs.append(temp_250[a:b])
            signal = np.stack(chs, axis=0).astype(np.float32)
            out.append(RawWindow(
                signal=signal,
                label=ACTIVITY_TO_LABEL[activity],
                activity=activity,
                source="local",
                subject=rec.subject,
                rec_name=rec.name,
                t_start=abs_t,                # store the ANCHOR time
            ))

    return out


def local_raw_windows(data_dir: str = "data", *, include_temp: bool = False) -> list[RawWindow]:
    out: list[RawWindow] = []
    for path in list_recordings(data_dir):
        rec = load_recording(path)
        ws = windows_from_local_recording(rec, include_temp=include_temp)
        n_ch = ws[0].signal.shape[0] if ws else "?"
        print(f"  local  {os.path.basename(path):34s}  {len(ws):3d} windows  ({n_ch} ch)")
        out.extend(ws)
    return out


def stack_windows(windows: Iterable[RawWindow]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of RawWindow into (N, C, WINDOW_N) and (N,) arrays."""
    ws = list(windows)
    if not ws:
        return np.empty((0, 3, WINDOW_N), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack([w.signal for w in ws], axis=0).astype(np.float32)
    y = np.array([w.label for w in ws], dtype=np.int64)
    return X, y
