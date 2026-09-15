#!/usr/bin/env python
"""Plot filtered PCG, Shannon energy, and S1/S2 markers for mta_6_3_math_8 from 0 to 600 s.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.io import load_recording
from src.features import preprocess_recording

def main():
    print("Loading recording mta_6_3_math_8...")
    rec_path = "data/mta_6_3_math_8.txt"
    if not os.path.exists(rec_path):
        print(f"Error: {rec_path} not found.")
        return

    rec = load_recording(rec_path)
    print("Preprocessing recording...")
    pp = preprocess_recording(rec)

    # 1. Align R-peaks to the microphone timeline
    # rpeaks are indices into ECG at pp.fs_ecg, starting at pp.ecg_t0
    # mic signal starts at pp.mic_t0, sampled at pp.fs_mic
    r_abs_t = pp.ecg_t0 + pp.rpeaks / pp.fs_ecg
    r_se_idx = np.round((r_abs_t - pp.mic_t0) * pp.fs_mic).astype(int)
    
    # Keep only those R-peaks that fall within the bounds of the mic signal
    valid_r_mask = (r_se_idx >= 0) & (r_se_idx < len(pp.se))
    r_se_idx = r_se_idx[valid_r_mask]
    r_abs_t = r_abs_t[valid_r_mask]

    # 2. Retrieve filtered mic signal
    # Resample raw mic signal to get the filtered mic signal
    # Let's rebuild the filtered mic signal from pp
    # Note: pp.fs_mic is target fs (usually 2000 Hz)
    # The filtered mic signal was calculated inside preprocess_recording but not stored in Preprocessed.
    # Wait, Preprocessed has: se (Shannon energy), pcg_peaks.
    # Let's filter the mic signal here exactly as in preprocess_recording:
    from src.preprocess import filter_mic_pcg
    from src.io import resample_uniform
    mic_u = resample_uniform(rec.channels["mic"], pp.fs_mic).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, pp.fs_mic)

    # 3. Classify S1/S2 using Method 3 (R-peak-centered classification)
    # Apply a conservative amplitude threshold based on the Shannon energy standard deviation
    std_se = np.std(pp.se)
    threshold = 1.5 * std_se
    filtered_peaks = pp.pcg_peaks[pp.se[pp.pcg_peaks] >= threshold]
    pcg_t_filtered = pp.mic_t0 + filtered_peaks / pp.fs_mic

    s1_dict = {}
    s2_dict = {}
    for r in r_abs_t:
        in_range = pcg_t_filtered[(pcg_t_filtered >= r - 0.4) & (pcg_t_filtered <= r + 0.4)]
        if len(in_range) >= 2:
            # Interpretation B: first 2 peaks chronologically in range
            A, B = in_range[0], in_range[1]
            D = B - A
            
            if 0.35 <= D <= 0.5:
                # Diastole
                s1_dict[r] = B
                prev_rs = r_abs_t[r_abs_t < r]
                if len(prev_rs) > 0:
                    s2_dict[prev_rs[-1]] = A
            elif 0.1 <= D <= 0.3:
                # Systole
                s1_dict[r] = A
                s2_dict[r] = B

    # Collect the S1/S2 peaks for plotting
    s1_times = np.sort(list(set(s1_dict.values())))
    s2_times = np.sort(list(set(s2_dict.values())))
    
    s1_idxs = np.round((s1_times - pp.mic_t0) * pp.fs_mic).astype(int)
    s2_idxs = np.round((s2_times - pp.mic_t0) * pp.fs_mic).astype(int)
    r_times = r_abs_t

    print(f"Detected {len(s1_times)} S1 peaks and {len(s2_times)} S2 peaks using Method 3 (Interpretation B).")

    # 4. Plotting 0 to 600 s in 20 s panels (30 panels total)
    duration = 600.0
    panel_s = 20.0
    num_panels = int(duration / panel_s) # 30 panels
    
    fig, axes = plt.subplots(num_panels, 1, figsize=(18, 1.8 * num_panels), sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.45)

    # Calculate absolute time vector for mic signal
    t_mic = pp.mic_t0 + np.arange(len(mic_f)) / pp.fs_mic

    # Determine y-limits to avoid scaling issues from huge noise transients
    # Set limits based on 98th percentile of active region
    ylim_se = np.percentile(pp.se, 98) * 1.5
    ylim_mic = np.percentile(np.abs(mic_f), 98) * 1.5

    for i in range(num_panels):
        ax = axes[i]
        t_start = i * panel_s
        t_end = (i + 1) * panel_s

        # Mask data for current panel
        mask_mic = (t_mic >= t_start) & (t_mic < t_end)
        t_sub = t_mic[mask_mic]
        mic_sub = mic_f[mask_mic]
        se_sub = pp.se[mask_mic]

        # Plot filtered heart sound (PCG) in light blue
        ax.plot(t_sub, mic_sub, color="#9ecae1", alpha=0.7, label="Filtered PCG" if i == 0 else "")
        
        # Plot Shannon energy envelope in orange
        ax.plot(t_sub, se_sub, color="#e6550d", linewidth=1.5, label="Shannon Energy" if i == 0 else "")

        # Plot ECG R-peaks as vertical dashed lines
        r_mask = (r_times >= t_start) & (r_times < t_end)
        for r_t in r_times[r_mask]:
            ax.axvline(r_t, color="#969696", linestyle="--", alpha=0.5, linewidth=1, label="ECG R-peak" if i == 0 and r_t == r_times[r_mask][0] else "")

        # Plot S1 peaks as green dots
        s1_mask = (s1_times >= t_start) & (s1_times < t_end)
        s1_t_sub = s1_times[s1_mask]
        s1_se_sub = pp.se[s1_idxs[s1_mask]]
        ax.scatter(s1_t_sub, s1_se_sub, color="#2ca02c", marker="o", s=30, zorder=5, label="S1 Peak" if i == 0 else "")

        # Plot S2 peaks as blue dots
        s2_mask = (s2_times >= t_start) & (s2_times < t_end)
        s2_t_sub = s2_times[s2_mask]
        s2_se_sub = pp.se[s2_idxs[s2_mask]]
        ax.scatter(s2_t_sub, s2_se_sub, color="#1f77b4", marker="s", s=25, zorder=5, label="S2 Peak" if i == 0 else "")

        # Labels and formatting
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(-ylim_mic, max(ylim_se, ylim_mic))
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Time: {int(t_start)}s - {int(t_end)}s | Phase: {'Rest' if t_start < 300 else 'Math'}", fontsize=10, loc="left", pad=2)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        if i == num_panels - 1:
            ax.set_xlabel("Time (seconds)")

    # Save to artifacts directory
    artifact_dir = Path("C:/Users/User/.gemini/antigravity-ide/brain/b85347c5-51d5-44c0-a52a-0d83134fc904")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path_artifact = artifact_dir / "mta_6_3_math_8_heart_sounds_method3.png"
    
    # Also save to workspace figures folder for easy viewing
    out_path_workspace = Path(__file__).resolve().parents[2] / "figures" / "mta_6_3_math_8_heart_sounds_method3.png"
    
    plt.savefig(out_path_artifact, dpi=150, bbox_inches="tight")
    plt.savefig(out_path_workspace, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully saved plot to:")
    print(f"  1. Artifacts: {out_path_artifact}")
    print(f"  2. Workspace: {out_path_workspace}")

if __name__ == "__main__":
    main()

