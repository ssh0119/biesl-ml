#!/usr/bin/env python
"""Calculate Signal-to-Noise Ratio (SNR in dB) of cardiac sound (CPS/PCG).

S1 and S2 peaks are extracted from the Shannon-energy envelope of the
bandpass-filtered (20-200 Hz) microphone signal, aligned with ECG R-peaks
using Method 3 (Interpretation B).

Signal region: Windows of duration s1_win_s / s2_win_s around S1 and S2 peaks.
Noise region : All other samples of the CPS signal.
SNR (dB)     : 10 * log10(Power_signal / Power_noise)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.io import load_recording, phase_boundaries, resample_uniform
from src.preprocess import (
    calculate_cardiac_snr,
    classify_s1_s2_indices,
    filter_mic_pcg,
)
from src.features import preprocess_recording


def calculate_snr_for_file(filepath: str, s1_win_s: float = 0.10, s2_win_s: float = 0.10, min_amp: float = 0.0, method: str = "rms") -> dict:
    rec = load_recording(filepath)
    pp = preprocess_recording(rec)

    # 1. Filter microphone CPS signal
    mic_u = resample_uniform(rec.channels["mic"], pp.fs_mic).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, pp.fs_mic)

    # 2. Align ECG R-peaks to microphone timeline
    r_abs_t = pp.ecg_t0 + pp.rpeaks / pp.fs_ecg
    r_se_idx = np.round((r_abs_t - pp.mic_t0) * pp.fs_mic).astype(int)

    valid_mask = (r_se_idx >= 0) & (r_se_idx < len(pp.se))
    r_se_idx = r_se_idx[valid_mask]

    # 3. Classify S1 and S2 peak indices
    s1_idxs, s2_idxs = classify_s1_s2_indices(
        pp.se, pp.fs_mic, pp.pcg_peaks, r_se_idx,
        mic_f=mic_f, min_amp=min_amp,
    )

    # 4. Overall recording SNR
    overall_snr = calculate_cardiac_snr(
        mic_f, pp.fs_mic, s1_idxs, s2_idxs,
        s1_win_s=s1_win_s, s2_win_s=s2_win_s, method=method,
    )

    # 5. Per-phase SNR (rest, stress, recovery)
    phases = phase_boundaries(rec)
    phase_snrs = {}

    for phase_name, (t_start, t_end) in phases.items():
        i_lo = max(0, int(round((t_start - pp.mic_t0) * pp.fs_mic)))
        i_hi = min(len(mic_f), int(round((t_end - pp.mic_t0) * pp.fs_mic)))

        if i_hi <= i_lo:
            phase_snrs[phase_name] = np.nan
            continue

        mic_phase = mic_f[i_lo:i_hi]

        # S1/S2 indices within phase slice
        s1_p = s1_idxs[(s1_idxs >= i_lo) & (s1_idxs < i_hi)] - i_lo
        s2_p = s2_idxs[(s2_idxs >= i_lo) & (s2_idxs < i_hi)] - i_lo

        snr_p = calculate_cardiac_snr(
            mic_phase, pp.fs_mic, s1_p, s2_p,
            s1_win_s=s1_win_s, s2_win_s=s2_win_s, method=method,
        )
        phase_snrs[phase_name] = snr_p

    return {
        "filename": os.path.basename(filepath),
        "subject": rec.subject,
        "stressor": rec.stressor,
        "n_s1": len(s1_idxs),
        "n_s2": len(s2_idxs),
        "snr_overall_db": overall_snr,
        "snr_rest_db": phase_snrs.get("rest", np.nan),
        "snr_stress_db": phase_snrs.get("stress", np.nan),
        "snr_recovery_db": phase_snrs.get("recovery", np.nan),
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate Cardiac Sound SNR (dB)")
    parser.add_argument("--file", type=str, default=None, help="Path to specific recording file (.txt)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing recordings")
    parser.add_argument("--s1-win", type=float, default=0.10, help="S1 window duration in seconds (default 0.10s)")
    parser.add_argument("--s2-win", type=float, default=0.10, help="S2 window duration in seconds (default 0.10s)")
    parser.add_argument("--min-amp", type=float, default=100.0, help="Minimum peak amplitude threshold for S1/S2 (default 100.0)")
    parser.add_argument("--method", type=str, default="rms", choices=["rms", "voltage_squared_peak", "voltage_squared_window"], help="SNR calculation method")
    parser.add_argument("--output-csv", type=str, default="outputs/cardiac_snr_report.csv", help="CSV report path")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]

    if not files:
        print("No recording files found to process.")
        return

    print(f"Calculating Cardiac Sound SNR for {len(files)} files...")
    print(f"Method: {args.method} | S1 Window: {args.s1_win:.2f}s | S2 Window: {args.s2_win:.2f}s | Min Amp: {args.min_amp:.1f}\n")

    results = []
    for fpath in files:
        try:
            print(f"Processing: {os.path.basename(fpath)}...")
            res = calculate_snr_for_file(fpath, s1_win_s=args.s1_win, s2_win_s=args.s2_win, min_amp=args.min_amp, method=args.method)
            results.append(res)
            print(f"  -> S1/S2 pairs: {res['n_s1']}/{res['n_s2']} | SNR (overall): {res['snr_overall_db']:.2f} dB | Rest: {res['snr_rest_db']:.2f} dB | Stress: {res['snr_stress_db']:.2f} dB")
        except Exception as e:
            print(f"  -> Failed to process {fpath}: {e}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved cardiac SNR report to: {args.output_csv}")
        print("\nSummary Statistics across recordings:")
        print(df[["snr_overall_db", "snr_rest_db", "snr_stress_db", "snr_recovery_db"]].describe())


if __name__ == "__main__":
    main()

