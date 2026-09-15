import os
import sys
import numpy as np

# Add the biesl-ml/src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.io import load_recording
from src.features import (
    preprocess_recording, 
    hr_window_feature, 
    poincare_window_features,
    _slice_peaks_by_time,
    FEATURE_WINDOWS_S,
    BOUNDARY_TIMES_S,
    BOUNDARY_BUFFER_PRE_S,
    BOUNDARY_BUFFER_POST_S,
    ANCHOR_STEP_S,
    MAX_FEATURE_WINDOW_S
)

def _window_touches_boundary(t_start: float, t_end: float) -> bool:
    for b in BOUNDARY_TIMES_S:
        if t_start <= b + BOUNDARY_BUFFER_POST_S and t_end >= b - BOUNDARY_BUFFER_PRE_S:
            return True
    return False

def _anchor_in_boundary(t_anchor: float) -> bool:
    for b in BOUNDARY_TIMES_S:
        if b - BOUNDARY_BUFFER_PRE_S <= t_anchor <= b + BOUNDARY_BUFFER_POST_S:
            return True
    return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec_path = os.path.join(base_dir, "data", "oyj_6_6_math_11.txt")
    
    if not os.path.exists(rec_path):
        print(f"Error: File not found at {rec_path}")
        return
        
    rec = load_recording(rec_path)
    pp = preprocess_recording(rec, phase_aware=True)
    
    from src.io import phase_boundaries
    phases = phase_boundaries(rec)
    
    half_max = MAX_FEATURE_WINDOW_S / 2.0
    half = {k: v / 2.0 for k, v in FEATURE_WINDOWS_S.items()}
    
    results_opt1 = []
    results_opt2 = []
    
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
            if _anchor_in_boundary(t_anchor):
                continue
                
            # HR
            hr_lo, hr_hi = t_anchor - half["hr"], t_anchor + half["hr"]
            if _window_touches_boundary(hr_lo, hr_hi):
                hr_val = float("nan")
            else:
                rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, hr_lo, hr_hi)
                hr_val = hr_window_feature(rp, pp.fs_ecg).get("hr", float("nan"))
                
            # RMSSD
            rmssd_lo, rmssd_hi = t_anchor - half["hrv_rmssd"], t_anchor + half["hrv_rmssd"]
            if _window_touches_boundary(rmssd_lo, rmssd_hi):
                rmssd_val = float("nan")
            else:
                rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, rmssd_lo, rmssd_hi)
                rmssd_val = poincare_window_features(rp, pp.fs_ecg).get("hrv_rmssd", float("nan"))
                
            results_opt1.append((t_anchor, hr_val, rmssd_val))
            
            # Option 2: Drop the entire anchor if the longest window touches the exclusion zone
            longest_lo, longest_hi = t_anchor - half_max, t_anchor + half_max
            if not _window_touches_boundary(longest_lo, longest_hi):
                results_opt2.append((t_anchor, hr_val, rmssd_val))

    print("=== OPTION 1: Set contaminated feature to NaN, keep anchor ===")
    print("Time_s,HR_bpm,RMSSD_ms")
    # Print only near transition to save output length, or print all
    # We will print all so the user gets the complete list they asked for
    for t, hr, rmssd in results_opt1:
        print(f"{t:.1f},{hr:.4f},{rmssd:.4f}")
        
    print("\n=== OPTION 2: Drop the entire anchor if longest window (60s) touches the exclusion zone ===")
    print("Time_s,HR_bpm,RMSSD_ms")
    for t, hr, rmssd in results_opt2:
        print(f"{t:.1f},{hr:.4f},{rmssd:.4f}")

if __name__ == "__main__":
    main()

