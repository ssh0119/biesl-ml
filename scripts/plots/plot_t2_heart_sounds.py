#!/usr/bin/env python
"""Plot filtered PCG, Shannon energy envelope, ECG R-peaks, and S1/S2 peak markers for data/T2.txt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.io import load_recording, resample_uniform
from src.features import preprocess_recording
from src.preprocess import classify_s1_s2_indices, filter_mic_pcg, calculate_cardiac_snr

def main():
    rec_path = "data/T2.txt"
    if not os.path.exists(rec_path):
        print(f"Error: {rec_path} not found.")
        return

    print("Loading recording data/T2.txt...")
    rec = load_recording(rec_path)
    print("Preprocessing recording...")
    pp = preprocess_recording(rec)

    # 1. Filter microphone signal
    mic_u = resample_uniform(rec.channels["mic"], pp.fs_mic).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, pp.fs_mic)

    # 2. Align ECG R-peaks to mic timeline
    r_abs_t = pp.ecg_t0 + pp.rpeaks / pp.fs_ecg
    r_se_idx = np.round((r_abs_t - pp.mic_t0) * pp.fs_mic).astype(int)

    valid_mask = (r_se_idx >= 0) & (r_se_idx < len(pp.se))
    r_se_idx = r_se_idx[valid_mask]
    r_abs_t = r_abs_t[valid_mask]

    # 3. Classify S1 and S2 peak indices (with amplitude threshold >= 100.0)
    min_amp = 100.0
    s1_idxs, s2_idxs = classify_s1_s2_indices(
        pp.se, pp.fs_mic, pp.pcg_peaks, r_se_idx,
        mic_f=mic_f, min_amp=min_amp,
    )

    s1_times = pp.mic_t0 + s1_idxs / pp.fs_mic
    s2_times = pp.mic_t0 + s2_idxs / pp.fs_mic

    snr_db = calculate_cardiac_snr(mic_f, pp.fs_mic, s1_idxs, s2_idxs)
    print(f"Detected {len(s1_idxs)} S1 peaks and {len(s2_idxs)} S2 peaks with min_amp >= {min_amp}.")
    print(f"Cardiac Sound SNR: {snr_db:.2f} dB")

    t_mic = pp.mic_t0 + np.arange(len(mic_f)) / pp.fs_mic
    total_duration = t_mic[-1] - t_mic[0]

    # Split into 20-second panels
    panel_s = 20.0
    num_panels = max(1, int(np.floor(total_duration / panel_s)))
    if total_duration - num_panels * panel_s >= 1.0:
        num_panels += 1

    fig, axes = plt.subplots(num_panels, 1, figsize=(16, 2.5 * num_panels), sharex=False)
    if num_panels == 1:
        axes = [axes]
    plt.subplots_adjust(hspace=0.45)

    ylim_se = np.percentile(pp.se, 99) * 1.4 if len(pp.se) > 0 else 1.0
    ylim_mic = np.percentile(np.abs(mic_f), 99) * 1.4 if len(mic_f) > 0 else 1.0

    # Add a horizontal dashed line for amplitude threshold = 100
    for i in range(num_panels):
        ax = axes[i]
        t_start = t_mic[0] + i * panel_s
        t_end = min(t_mic[-1], t_start + panel_s)

        mask = (t_mic >= t_start) & (t_mic < t_end)
        t_sub = t_mic[mask]
        mic_sub = mic_f[mask]
        se_sub = pp.se[mask]

        # Plot Filtered PCG
        ax.plot(t_sub, mic_sub, color="#9ecae1", alpha=0.7, label="Filtered CPS (20-200Hz)" if i == 0 else "")
        # Plot Shannon Energy Envelope
        ax.plot(t_sub, se_sub, color="#e6550d", linewidth=1.5, label="Shannon Energy Envelope" if i == 0 else "")

        # Threshold line
        ax.axhline(min_amp, color="#d95f02", linestyle=":", alpha=0.7, linewidth=1.2, label=f"Amp Threshold ({min_amp:.0f})" if i == 0 else "")
        ax.axhline(-min_amp, color="#d95f02", linestyle=":", alpha=0.7, linewidth=1.2)

        # Plot R-peaks
        r_mask = (r_abs_t >= t_start) & (r_abs_t < t_end)
        for r_t in r_abs_t[r_mask]:
            ax.axvline(r_t, color="#969696", linestyle="--", alpha=0.6, linewidth=1.2, label="ECG R-peak" if i == 0 and r_t == r_abs_t[r_mask][0] else "")

        # Highlight S1 regions and peaks
        s1_mask = (s1_times >= t_start) & (s1_times < t_end)
        for s1_t in s1_times[s1_mask]:
            ax.axvspan(s1_t - 0.05, s1_t + 0.05, color="#2ca02c", alpha=0.15)
        if np.any(s1_mask):
            s1_sub_t = s1_times[s1_mask]
            s1_sub_se = pp.se[s1_idxs[s1_mask]]
            ax.scatter(s1_sub_t, s1_sub_se, color="#2ca02c", marker="o", s=45, zorder=5, label="S1 Peak" if i == 0 else "")

        # Highlight S2 regions and peaks
        s2_mask = (s2_times >= t_start) & (s2_times < t_end)
        for s2_t in s2_times[s2_mask]:
            ax.axvspan(s2_t - 0.05, s2_t + 0.05, color="#1f77b4", alpha=0.15)
        if np.any(s2_mask):
            s2_sub_t = s2_times[s2_mask]
            s2_sub_se = pp.se[s2_idxs[s2_mask]]
            ax.scatter(s2_sub_t, s2_sub_se, color="#1f77b4", marker="s", s=40, zorder=5, label="S2 Peak" if i == 0 else "")

        ax.set_xlim(t_start, t_end)
        ax.set_ylim(-ylim_mic, max(ylim_se, ylim_mic))
        ax.set_ylabel("Amplitude")
        ax.set_title(f"T2.txt | Time: {t_start:.1f}s - {t_end:.1f}s | SNR (Amp ??{min_amp:.0f}): {snr_db:.2f} dB", fontsize=10, loc="left", pad=3)
        ax.grid(True, linestyle=":", alpha=0.6)

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        if i == num_panels - 1:
            ax.set_xlabel("Time (seconds)")

    # Save outputs
    art_dir = Path("C:/Users/User/.gemini/antigravity-ide/brain/b1b377dd-78fb-43d5-a451-4bfcc4e4171a")
    art_dir.mkdir(parents=True, exist_ok=True)
    out_art = art_dir / "t2_heart_sounds_s1_s2.png"

    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig = fig_dir / "t2_heart_sounds_s1_s2.png"

    plt.savefig(out_art, dpi=150, bbox_inches="tight")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print("Successfully generated plot:")
    print(f"  Artifact: {out_art}")
    print(f"  Workspace figure: {out_fig}")

if __name__ == "__main__":
    main()

