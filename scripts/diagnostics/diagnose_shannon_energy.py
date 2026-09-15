#!/usr/bin/env python
"""Diagnose Shannon energy flattening and generate a comparison plot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.io import load_recording, resample_uniform
from src.preprocess import filter_mic_pcg, shannon_energy

def main():
    print("Loading recording mta_6_3_math_8...")
    rec_path = "data/mta_6_3_math_8.txt"
    if not os.path.exists(rec_path):
        print(f"Error: {rec_path} not found.")
        return

    rec = load_recording(rec_path)
    fs = 2000.0
    mic_u = resample_uniform(rec.channels["mic"], fs).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, fs)

    # 1. Compute Global Shannon Energy (current implementation)
    se_global = shannon_energy(mic_f, smooth_ms=50.0, fs=fs)

    # 2. Compute Clipped Shannon Energy (clipping raw signal at 99th percentile)
    p99 = np.percentile(np.abs(mic_f), 99)
    mic_clipped = np.clip(mic_f, -p99, p99)
    se_clipped = shannon_energy(mic_clipped, smooth_ms=50.0, fs=fs)

    # 3. Compute Locally Normalized Shannon Energy (2.0s rolling max)
    n_win = int(2.0 * fs)
    padded = np.pad(np.abs(mic_f), n_win // 2, mode="edge")
    local_max = maximum_filter1d(padded, size=n_win)[n_win // 2 : n_win // 2 + len(mic_f)]
    local_max = np.where(local_max < 1e-12, 1e-12, local_max)
    xn = np.clip(mic_f / local_max, -1.0, 1.0)
    x2 = np.where(xn * xn < 1e-12, 1e-12, xn * xn)
    se_raw_local = -x2 * np.log(x2)
    win_smooth = int(50.0 / 1000.0 * fs)
    se_local = np.convolve(se_raw_local, np.ones(win_smooth) / win_smooth, mode="same")

    # Time vectors
    t = np.arange(len(mic_f)) / fs

    # Zoom window: 120s to 130s
    t_start, t_end = 120.0, 130.0
    mask = (t >= t_start) & (t < t_end)
    t_zoom = t[mask]
    mic_zoom = mic_f[mask]
    se_global_zoom = se_global[mask]
    se_clipped_zoom = se_clipped[mask]
    se_local_zoom = se_local[mask]

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Panel 1: Filtered PCG
    axes[0].plot(t_zoom, mic_zoom, color="#1f77b4", linewidth=1.5)
    axes[0].set_title("Filtered Microphone (PCG) Signal (120 s - 130 s)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Global Shannon Energy
    axes[1].plot(t_zoom, se_global_zoom, color="#d62728", linewidth=1.5)
    axes[1].set_title("Globally Normalized Shannon Energy (Current Code - Flattened by Outliers)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Shannon Energy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    # Force y-axis limit to show how small the values are compared to the max possible (0.368)
    axes[1].set_ylim(-0.01, 0.38)

    # Panel 3: Clipped Shannon Energy
    axes[2].plot(t_zoom, se_clipped_zoom, color="#ff7f0e", linewidth=1.5)
    axes[2].set_title("Shannon Energy after Clipping outliers to 99th Percentile", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Shannon Energy")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].set_ylim(-0.01, 0.38)

    # Panel 4: Local Shannon Energy
    axes[3].plot(t_zoom, se_local_zoom, color="#2ca02c", linewidth=1.5)
    axes[3].set_title("Locally Normalized Shannon Energy (2.0 s Sliding Window Max)", fontsize=12, fontweight="bold")
    axes[3].set_ylabel("Shannon Energy")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True, linestyle=":", alpha=0.6)
    axes[3].set_ylim(-0.01, 0.38)

    plt.tight_layout()
    
    # Save path
    artifact_dir = Path("C:/Users/User/.gemini/antigravity-ide/brain/b85347c5-51d5-44c0-a52a-0d83134fc904")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "shannon_energy_diagnose.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully generated comparison plot at: {out_path}")

if __name__ == "__main__":
    main()

