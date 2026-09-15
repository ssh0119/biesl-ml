import os
import sys
import numpy as np

# Add the biesl-ml/src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.io import load_recording
from src.features import (
    preprocess_recording,
    _slice_peaks_by_time,
    FEATURE_WINDOWS_S,
    MAX_FEATURE_WINDOW_S,
    ANCHOR_STEP_S
)
from src.preprocess import clean_nn_intervals
from src.io import phase_boundaries

def _poincare(nn_ms: np.ndarray) -> dict[str, float]:
    nan = float("nan")
    out = {"sd1": nan, "sd2": nan}
    if len(nn_ms) < 3:
        return out
    sdnn = float(np.std(nn_ms, ddof=1))
    sdsd = float(np.std(np.diff(nn_ms), ddof=1))
    sd1 = sdsd / np.sqrt(2.0)
    var_diff = 2.0 * sdnn * sdnn - sd1 * sd1
    sd2 = float(np.sqrt(var_diff)) if var_diff > 0 else nan
    out["sd1"] = sd1
    out["sd2"] = sd2
    return out

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec_path = os.path.join(base_dir, "data", "oyj_6_6_math_6.txt")
    
    rec = load_recording(rec_path)
    pp = preprocess_recording(rec, phase_aware=True)
    phases = phase_boundaries(rec)
    
    half_max = MAX_FEATURE_WINDOW_S / 2.0
    half = {k: v / 2.0 for k, v in FEATURE_WINDOWS_S.items()}
    
    print("Time_s,RMSSD_ms,SD1_ms,Ratio")
    for phase_name, (p_start, p_end) in phases.items():
        if phase_name == "recovery":
            continue
        anchor_lo = p_start + half_max
        anchor_hi = p_end - half_max
        if anchor_hi < anchor_lo:
            continue
            
        n_anchors = int(np.floor((anchor_hi - anchor_lo) / ANCHOR_STEP_S)) + 1
        anchors = anchor_lo + ANCHOR_STEP_S * np.arange(n_anchors)
        
        for t_anchor in anchors:
            t_anchor = float(t_anchor)
            # We don't exclude here just to see the whole range
            
            # RMSSD window
            t_lo, t_hi = t_anchor - half["hrv_rmssd"], t_anchor + half["hrv_rmssd"]
            rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
            nn = clean_nn_intervals(rp, pp.fs_ecg)
            
            if len(nn) >= 3:
                diff = np.diff(nn)
                rmssd = np.sqrt(np.mean(diff ** 2))
                poin = _poincare(nn)
                sd1 = poin["sd1"]
                ratio = rmssd / sd1 if sd1 > 0 else float("nan")
                print(f"{t_anchor:.1f},{rmssd:.4f},{sd1:.4f},{ratio:.4f}")

if __name__ == "__main__":
    main()

