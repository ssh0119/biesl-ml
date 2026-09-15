"""
Plot ECG signal for files in biesl-ml/data/ directory.
Plots 0 to 600s, divided into 30 panels (20s each), 1 panel per row (1 column layout).
Plots the filtered ECG signal with R-peaks marked in red.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add the biesl-ml/src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.io import load_recording, list_recordings
from src.features import preprocess_recording

def plot_ecg_panels_for_rec(rec_path: str, out_path: str) -> None:
    print(f"Loading and preprocessing {os.path.basename(rec_path)}...")
    rec = load_recording(rec_path)
    
    # We use phase_aware=True to avoid cross-phase contamination (e.g. plank noise)
    pp = preprocess_recording(rec, phase_aware=True)
    
    fs = pp.fs_ecg
    t0 = pp.ecg_t0
    sig = -pp.ecg  # Flip ECG so R-peaks deflect upwards on the plot
    t = t0 + np.arange(len(sig)) / fs
    peaks = pp.rpeaks
    
    panel_s = 20.0
    max_s = 600.0
    n_panels = int(max_s / panel_s)  # 30 panels
    
    # 1 panel 1 row layout: 30 rows, 1 column
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, n_panels * 2.2), sharex=False)
    fig.patch.set_facecolor("#0D1B2A")
    
    for p in range(n_panels):
        ax = axes[p]
        ax.set_facecolor("#112233")
        
        seg_t0 = t0 + p * panel_s
        seg_t1 = seg_t0 + panel_s
        
        # Get samples in time window
        i_lo = int(round((seg_t0 - t0) * fs))
        i_hi = int(round((seg_t1 - t0) * fs))
        
        i_lo = max(0, min(len(sig), i_lo))
        i_hi = max(0, min(len(sig), i_hi))
        
        t_seg = t[i_lo:i_hi]
        sig_seg = sig[i_lo:i_hi]
        
        if len(t_seg) == 0:
            ax.text(0.5, 0.5, "No Data in this time window", color="gray", ha="center", va="center")
            ax.set_facecolor("#1B2A4A")
            continue
            
        # Plot filtered ECG signal
        ax.plot(t_seg, sig_seg, color="#4FC3F7", lw=0.8, alpha=0.95, zorder=2)
        
        # Filter peaks falling in this segment
        peaks_in_seg = peaks[(peaks >= i_lo) & (peaks < i_hi)]
        
        # Compute local y-limits using 1st and 99th percentiles to handle outliers/noise spikes
        ylo = float(np.percentile(sig_seg, 1.0))
        yhi = float(np.percentile(sig_seg, 99.0))
        ypad = (yhi - ylo) * 0.25 if yhi > ylo else 1.0
        ylim = (ylo - ypad, yhi + ypad)
        ax.set_ylim(ylim)
        
        if len(peaks_in_seg) > 0:
            peak_t = t0 + peaks_in_seg / fs
            peak_val = sig[peaks_in_seg]
            ax.scatter(peak_t, peak_val, color="#FF1744", s=35, zorder=5, marker="v", 
                       edgecolors="white", linewidths=0.5)
            
            # Label peak count and estimated HR in the panel
            hr_est = 60.0 * len(peaks_in_seg) / panel_s
            ax.text(seg_t1 - 0.25, ylim[1] - ypad * 0.05, 
                    f"{len(peaks_in_seg)} peaks | ~{hr_est:.0f} bpm",
                    color="#FFD700", fontsize=8, ha="right", va="top",
                    fontfamily="monospace")
            
        # Customize axes and grids
        ax.set_xlim(seg_t0, seg_t1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
        ax.tick_params(colors="#B0BEC5", labelsize=8)
        ax.grid(axis="x", which="both", color="#37474F", lw=0.35, linestyle=":", alpha=0.6)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("#37474F")
            
        # Label each row with its time range
        ax.text(seg_t0 + 0.25, ylim[1] - ypad * 0.05, f"Panel {p+1}: {p*20}-{(p+1)*20} s",
                color="#90CAF9", fontsize=9, ha="left", va="top",
                fontweight="bold")
        
    fig.suptitle(f"ECG Panel Plot (20s rows, 0-600s) \u2014 {os.path.basename(rec_path)}",
                 color="#ECEFF1", fontsize=16, fontweight="bold", y=0.992)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Successfully saved panel plot to: {out_path}\n")

def matches_filter(name: str, targets: list[str]) -> bool:
    if not targets:
        return True
    name_lower = name.lower()
    for t in targets:
        t_lower = t.lower()
        if t_lower in name_lower:
            return True
        # Split by underscore/hyphen and check if all parts are in name
        parts = [p for p in t_lower.replace("-", "_").split("_") if p]
        if parts and all(p in name_lower for p in parts):
            return True
    return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "figures", "ecg_panels")
    os.makedirs(out_dir, exist_ok=True)
    
    recordings = list_recordings(data_dir)
    if not recordings:
        print(f"No recordings found in data folder: {data_dir}")
        return
        
    targets = sys.argv[1:]
    if targets:
        recordings = [r for r in recordings if matches_filter(os.path.splitext(os.path.basename(r))[0], targets)]
        print(f"Filtered to {len(recordings)} recordings matching targets: {targets}")
    else:
        print(f"Found {len(recordings)} recordings in {data_dir}")
        
    for rec_path in recordings:
        out_name = f"{os.path.splitext(os.path.basename(rec_path))[0]}_ecg_panels_20s.png"
        out_path = os.path.join(out_dir, out_name)
        try:
            plot_ecg_panels_for_rec(rec_path, out_path)
        except Exception as e:
            print(f"Error processing {rec_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

