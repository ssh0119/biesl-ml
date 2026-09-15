"""Export RR and RRV calculations from a 2-column Excel file (Time, Breathing Data).

Usage:
    python scripts/exports/export_rr_rrv_excel.py --input path/to/input.xlsx --output path/to/output.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def filter_br(x: np.ndarray, fs: float, baseline_window_s: float = 30.0, smooth_window_s: float = 0.5) -> np.ndarray:
    """Median-filter pipeline matching src.preprocess.filter_br:
    1. Subtract 30-s median filter baseline (removes drift without ringing).
    2. Apply 0.5-s median filter for smoothing/despiking.
    """
    x = x.astype(np.float64, copy=False)
    base_w = max(1, int(round(baseline_window_s * fs)))
    baseline = median_filter(x, size=base_w, mode="nearest")
    detrended = x - baseline
    smooth_w = max(1, int(round(smooth_window_s * fs)))
    return median_filter(detrended, size=smooth_w, mode="nearest")


def detect_br_peaks(br: np.ndarray, fs: float, window_s: float = 60.0, step_s: float = 30.0, prom_frac: float = 0.25, merge_dist_s: float = 0.75) -> np.ndarray:
    """Sliding-window peak detection matching src.preprocess.detect_br_peaks_sliding."""
    n = len(br)
    if n < int(fs * 2):
        return np.array([], dtype=int)
    win_n = max(int(round(window_s * fs)), 1)
    step_n = max(int(round(step_s * fs)), 1)
    dist = max(int(round(fs * 1.5)), 1)

    found: list[int] = []
    starts = list(range(0, max(n - win_n + 1, 1), step_n))
    if starts[-1] + win_n < n:
        starts.append(max(0, n - win_n))

    for s in starts:
        seg = br[s:s + win_n]
        if len(seg) < int(fs * 2):
            continue
        active = float(np.percentile(np.abs(seg - np.median(seg)), 90))
        prom = prom_frac * active if active > 1e-9 else None
        loc, _ = find_peaks(seg, distance=dist, prominence=prom)
        found.extend((loc + s).tolist())

    if not found:
        return np.array([], dtype=int)

    merge = max(int(round(merge_dist_s * fs)), 1)
    found.sort()
    out: list[int] = [found[0]]
    for p in found[1:]:
        if p - out[-1] > merge:
            out.append(p)
        elif br[p] > br[out[-1]]:
            out[-1] = p
    return np.asarray(out, dtype=int)


def calculate_rr_and_rrv(
    t_raw: np.ndarray,
    br_raw: np.ndarray,
    fs_target: float = 100.0,
    anchor_step_s: float = 5.0,
    rr_win_s: float = 40.0,
    rrv_win_s: float = 60.0,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Process raw breathing signal to compute RR and RRV over time."""
    # Filter out any NaN/Inf in input arrays
    valid_mask = np.isfinite(t_raw) & np.isfinite(br_raw)
    t_raw = t_raw[valid_mask]
    br_raw = br_raw[valid_mask]

    if len(t_raw) < 2:
        return pd.DataFrame(columns=["time_of_RR", "RR", "time_of_RRV", "RRV"])

    # Sort by time in case input is unsorted
    sort_idx = np.argsort(t_raw)
    t_raw = t_raw[sort_idx]
    br_raw = br_raw[sort_idx]

    # Resample to uniform target rate
    t0 = t_raw[0]
    t_end = t_raw[-1]
    n_samples = int(np.floor((t_end - t0) * fs_target)) + 1
    t_grid = t0 + np.arange(n_samples) / fs_target
    br_grid = np.interp(t_grid, t_raw, br_raw)

    # Filter signal
    br_filtered = filter_br(br_grid, fs_target)

    # Detect peaks
    peaks_idx = detect_br_peaks(br_filtered, fs_target)

    # Generate sliding window anchor points
    half_max_win = max(rr_win_s, rrv_win_s) / 2.0
    t_anchors = np.arange(t0 + half_max_win, t_end - half_max_win + 1e-6, anchor_step_s)

    results = []
    for t_anc in t_anchors:
        # RR: 40-s window centered at t_anc -> [t_anc - 20, t_anc + 20]
        rr_start, rr_end = t_anc - rr_win_s / 2.0, t_anc + rr_win_s / 2.0
        rr_peak_mask = (t_grid[peaks_idx] >= rr_start) & (t_grid[peaks_idx] < rr_end)
        rr_peaks = peaks_idx[rr_peak_mask]

        if len(rr_peaks) >= 2:
            rr_iv = np.diff(rr_peaks) / fs_target
            rr_iv_valid = rr_iv[(rr_iv > 1.0) & (rr_iv < 12.0)]
            rr_val = 60.0 / float(np.mean(rr_iv_valid)) if len(rr_iv_valid) >= 1 else np.nan
        else:
            rr_val = np.nan

        # RRV: 60-s window centered at t_anc -> [t_anc - 30, t_anc + 30]
        rrv_start, rrv_end = t_anc - rrv_win_s / 2.0, t_anc + rrv_win_s / 2.0
        rrv_peak_mask = (t_grid[peaks_idx] >= rrv_start) & (t_grid[peaks_idx] < rrv_end)
        rrv_peaks = peaks_idx[rrv_peak_mask]

        if len(rrv_peaks) >= 2:
            rrv_iv = np.diff(rrv_peaks) / fs_target
            rrv_iv_valid = rrv_iv[(rrv_iv > 1.0) & (rrv_iv < 12.0)]
            rrv_val = float(np.std(rrv_iv_valid[-5:], ddof=0)) if len(rrv_iv_valid) >= 2 else np.nan
        else:
            rrv_val = np.nan

        results.append({
            "time_of_RR": t_anc,
            "RR": rr_val,
            "time_of_RRV": t_anc,
            "RRV": rrv_val,
        })

    df_out = pd.DataFrame(results)
    if drop_na:
        df_out = df_out.dropna().reset_index(drop=True)
    return df_out


def load_input_file(path: Path) -> pd.DataFrame:
    """Load time and breathing data from Excel (.xlsx/.xls), CSV (.csv), TSV (.tsv), or TXT (.txt) files.
    Automatically removes any NaN or missing values at the end or anywhere in the file.
    """
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
    elif ext in (".tsv", ".txt", ".dat"):
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
    else:
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

    # Take first 2 columns, convert to numeric, and drop any rows containing NaN
    df = df.iloc[:, :2]
    df.columns = ["time", "breathing"]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def save_output_file(df: pd.DataFrame, path: Path) -> Path:
    """Export DataFrame to Excel (.xlsx) or CSV/TXT (.csv/.txt) based on output file extension."""
    ext = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure output has no residual NaNs
    df = df.dropna().reset_index(drop=True)

    if ext in (".csv", ".txt", ".tsv", ".dat"):
        sep = "\t" if ext in (".tsv", ".txt") else ","
        df.to_csv(path, index=False, sep=sep)
        return path
    else:
        if ext not in (".xlsx", ".xls"):
            path = path.with_suffix(".xlsx")
        df.to_excel(path, index=False)
        return path


def plot_peaks_overview(
    t_grid: np.ndarray,
    br_filtered: np.ndarray,
    peaks_idx: np.ndarray,
    output_path: Path | str,
    t_raw: np.ndarray | None = None,
    br_raw: np.ndarray | None = None,
    df_rr: pd.DataFrame | None = None,
    title: str = "Respiration Signal with Detected Peaks",
) -> Path:
    """Generate an overview figure showing the breathing signal, detected peaks, and RR/RRV trends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    peak_t = t_grid[peaks_idx]
    peak_y = br_filtered[peaks_idx]
    n_peaks = len(peaks_idx)

    ibis = np.diff(peak_t) if n_peaks >= 2 else np.array([])
    mean_ibi = float(np.mean(ibis)) if len(ibis) else np.nan
    mean_rr = 60.0 / mean_ibi if len(ibis) and mean_ibi > 0 else np.nan

    has_rr_data = df_rr is not None and not df_rr.empty

    if has_rr_data:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]}, sharex=True)
        ax_sig, ax_rr, ax_rrv = axes[0], axes[1], axes[2]
    else:
        fig, ax_sig = plt.subplots(1, 1, figsize=(14, 4.5))
        ax_rr, ax_rrv = None, None

    # Top Panel: Respiration Signal & Detected Peaks
    if t_raw is not None and br_raw is not None:
        ax2 = ax_sig.twinx()
        raw_z = (br_raw - np.nanmedian(br_raw)) / (np.nanstd(br_raw) or 1.0)
        ax2.plot(t_raw, raw_z, color="#a0aec0", lw=0.6, alpha=0.55, label="Raw signal (z-norm)")
        ax2.set_ylabel("Raw (z-norm)", color="#718096", fontsize=9)
        ax2.tick_params(colors="#718096", labelsize=8)
        ax2.grid(False)

    ax_sig.plot(t_grid, br_filtered, color="#1e3a8a", lw=1.2, label="Median-filtered breathing signal")
    if n_peaks > 0:
        ax_sig.plot(peak_t, peak_y, "v", color="#dc2626", markersize=6.5, markeredgecolor="#991b1b",
                    markeredgewidth=1.0, label=f"Detected peaks (n={n_peaks})")

    ax_sig.set_title(
        f"{title}\n"
        f"Total Peaks: {n_peaks}  |  Mean IBI: {mean_ibi:.2f} s  |  Overall Mean RR: {mean_rr:.1f} bpm  |  Duration: {t_grid[-1] - t_grid[0]:.1f} s",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_sig.set_ylabel("Filtered Breathing (N)", fontsize=9, fontweight="semibold")
    ax_sig.grid(True, linestyle="--", alpha=0.45)

    if t_raw is not None and br_raw is not None:
        h1, l1 = ax_sig.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_sig.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.9)
    else:
        ax_sig.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Middle Panel: Dynamic Respiration Rate (RR)
    if has_rr_data and ax_rr is not None:
        ax_rr.plot(df_rr["time_of_RR"], df_rr["RR"], color="#047857", lw=1.6, marker="o", markersize=3, label="RR (40s sliding window)")
        ax_rr.axhline(mean_rr, color="#065f46", linestyle=":", lw=1.2, label=f"Mean RR ({mean_rr:.1f} bpm)")
        ax_rr.set_ylabel("RR (bpm)", fontsize=9, fontweight="semibold")
        ax_rr.grid(True, linestyle="--", alpha=0.45)
        ax_rr.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Bottom Panel: Respiration Rate Variability (RRV)
    if has_rr_data and ax_rrv is not None:
        ax_rrv.plot(df_rr["time_of_RRV"], df_rr["RRV"], color="#d97706", lw=1.6, marker="s", markersize=3, label="RRV (60s sliding window std)")
        ax_rrv.set_ylabel("RRV (s)", fontsize=9, fontweight="semibold")
        ax_rrv.set_xlabel("Time (s)", fontsize=9, fontweight="semibold")
        ax_rrv.grid(True, linestyle="--", alpha=0.45)
        ax_rrv.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    else:
        ax_sig.set_xlabel("Time (s)", fontsize=9, fontweight="semibold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_peaks_detailed_segments(
    t_grid: np.ndarray,
    br_filtered: np.ndarray,
    peaks_idx: np.ndarray,
    output_path: Path | str,
    segment_duration_s: float = 60.0,
    title: str = "Detailed Segmented Respiration Peaks",
) -> Path:
    """Generate a multi-row segmented plot (e.g., 60s per segment) showing each individual peak clearly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0, t_end = t_grid[0], t_grid[-1]
    n_segments = int(np.ceil((t_end - t0) / segment_duration_s))

    fig, axes = plt.subplots(n_segments, 1, figsize=(14, 2.7 * n_segments), sharey=True)
    if n_segments == 1:
        axes = [axes]

    peak_t = t_grid[peaks_idx]
    peak_y = br_filtered[peaks_idx]

    global_peak_num = 1
    for i, ax in enumerate(axes):
        seg_start = t0 + i * segment_duration_s
        seg_end = min(t0 + (i + 1) * segment_duration_s, t_end)

        mask_sig = (t_grid >= seg_start) & (t_grid <= seg_end)
        mask_peaks = (peak_t >= seg_start) & (peak_t <= seg_end)

        ax.plot(t_grid[mask_sig], br_filtered[mask_sig], color="#1e3a8a", lw=1.2, label="Filtered Signal" if i == 0 else "")
        ax.axhline(0, color="#94a3b8", linestyle="--", lw=0.7, alpha=0.6)

        seg_peak_t = peak_t[mask_peaks]
        seg_peak_y = peak_y[mask_peaks]

        if len(seg_peak_t) > 0:
            ax.plot(seg_peak_t, seg_peak_y, "v", color="#dc2626", markersize=7, markeredgecolor="#991b1b",
                    markeredgewidth=1.0, label="Detected Peaks" if i == 0 else "")

            for pt, py in zip(seg_peak_t, seg_peak_y):
                ax.annotate(
                    f"P{global_peak_num}\n{pt:.1f}s",
                    xy=(pt, py),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#991b1b",
                    fontweight="bold",
                )
                global_peak_num += 1

        ax.set_xlim(seg_start, seg_end)
        ax.set_ylabel(f"{seg_start:.0f}-{seg_end:.0f}s\nBreathing (N)", fontsize=8.5, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.set_title(f"{title} (Total: {len(peaks_idx)} Peaks)", fontsize=11, fontweight="bold", pad=8)
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)", fontsize=9, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Calculate RR and RRV from breathing data in TXT, CSV, or Excel files.")
    parser.add_argument("--input", "-i", required=True, help="Path to input file (.txt, .csv, .tsv, .xlsx)")
    parser.add_argument("--output", "-o", required=True, help="Path to output file (.xlsx, .csv, .txt)")
    parser.add_argument("--step", "-s", type=float, default=5.0, help="Step size between anchor points in seconds (default: 5.0)")
    parser.add_argument("--plot", "-p", type=str, default=None, help="Optional path to save overview plot image (.png)")
    parser.add_argument("--detailed-plot", "-d", type=str, default=None, help="Optional path to save segmented detailed plot (.png)")
    parser.add_argument("--export-peaks", type=str, default=None, help="Optional path to save detected peaks table (.csv or .xlsx)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print(f"Reading input file: {input_path}...")
    df_in = load_input_file(input_path)

    if df_in.shape[0] == 0:
        print("Error: Input file contains no valid numerical data rows after dropping NaN values.")
        sys.exit(1)

    t_raw = df_in["time"].to_numpy(dtype=float)
    br_raw = df_in["breathing"].to_numpy(dtype=float)

    fs_target = 100.0
    t0 = t_raw[0]
    t_end = t_raw[-1]
    n_samples = int(np.floor((t_end - t0) * fs_target)) + 1
    t_grid = t0 + np.arange(n_samples) / fs_target
    br_grid = np.interp(t_grid, t_raw, br_raw)
    br_filtered = filter_br(br_grid, fs_target)
    peaks_idx = detect_br_peaks(br_filtered, fs_target)

    print(f"Detected {len(peaks_idx)} peaks across {t_end - t0:.1f} seconds of respiration data.")

    print(f"Calculating RR and RRV (step: {args.step}s)...")
    df_out = calculate_rr_and_rrv(t_raw, br_raw, anchor_step_s=args.step)

    saved_path = save_output_file(df_out, output_path)
    print(f"Successfully saved RR/RRV table ({len(df_out)} valid rows) to: {saved_path}")

    if args.export_peaks:
        peak_t = t_grid[peaks_idx]
        peak_filt = br_filtered[peaks_idx]
        peak_raw = br_grid[peaks_idx]
        ibis = np.concatenate([[np.nan], np.diff(peak_t)]) if len(peak_t) > 0 else []
        inst_rr = [60.0 / ibi if ibi > 0 else np.nan for ibi in ibis]

        df_peaks = pd.DataFrame({
            "peak_index": np.arange(1, len(peaks_idx) + 1),
            "time_s": peak_t,
            "filtered_amplitude": peak_filt,
            "raw_amplitude": peak_raw,
            "inter_breath_interval_s": ibis,
            "instantaneous_rr_bpm": inst_rr,
        })
        peaks_out_path = Path(args.export_peaks)
        if peaks_out_path.suffix.lower() == ".csv":
            df_peaks.to_csv(peaks_out_path, index=False)
        else:
            df_peaks.to_excel(peaks_out_path, index=False)
        print(f"Successfully saved detected peaks ({len(df_peaks)} peaks) to: {peaks_out_path}")

    if args.plot:
        plot_path = plot_peaks_overview(
            t_grid=t_grid,
            br_filtered=br_filtered,
            peaks_idx=peaks_idx,
            output_path=args.plot,
            t_raw=t_raw,
            br_raw=br_raw,
            df_rr=df_out,
            title=f"Respiration Peaks & RR/RRV: {input_path.name}",
        )
        print(f"Successfully saved overview plot to: {plot_path}")

    if args.detailed_plot:
        det_plot_path = plot_peaks_detailed_segments(
            t_grid=t_grid,
            br_filtered=br_filtered,
            peaks_idx=peaks_idx,
            output_path=args.detailed_plot,
            segment_duration_s=60.0,
            title=f"Detailed Respiration Peaks: {input_path.name}",
        )
        print(f"Successfully saved detailed segmented plot to: {det_plot_path}")


if __name__ == "__main__":
    main()

