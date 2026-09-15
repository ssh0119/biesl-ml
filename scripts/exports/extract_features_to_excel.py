import os
import sys
import pandas as pd
import numpy as np

# Keep imports and default paths anchored at the repository root even though
# this utility lives under scripts/exports/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.io import load_recording, phase_boundaries
from src.features import (
    preprocess_recording,
    FEATURE_WINDOWS_S,
    MAX_FEATURE_WINDOW_S,
    ANCHOR_STEP_S,
    _slice_peaks_by_time,
    hr_window_feature,
    poincare_window_features,
    rr_window_feature,
    rrv_window_feature,
    _csi_for_window,
    resp_amp_window_feature,
    mic_rms_window_feature,
    Window
)

def get_phase_name(t):
    if t < 300.0:
        return "rest"
    elif t < 510.0:
        return "stress"
    else:
        return "recovery"

def custom_windows_for_recording(pp):
    rec = pp.rec
    half = {k: v / 2.0 for k, v in FEATURE_WINDOWS_S.items()}
    
    boundary_times = (300.0, 510.0)
    boundary_buffer_pre = 10.0
    boundary_buffer_post = 30.0
    
    def is_window_valid(t_anchor, w_size):
        t_lo = t_anchor - w_size / 2.0
        t_hi = t_anchor + w_size / 2.0
        
        # Must be within recording bounds
        if t_lo < 0.0 or t_hi > rec.duration_s:
            return False
            
        # Must not cross phase boundaries
        phase_lo = get_phase_name(t_lo)
        phase_hi = get_phase_name(t_hi)
        if phase_lo != phase_hi:
            return False
            
        # Must not touch boundary exclusion zones
        for b in boundary_times:
            if t_lo <= b + boundary_buffer_post and t_hi >= b - boundary_buffer_pre:
                return False
                
        return True

    # Generate anchors from 0.0 to end of recording stepping by ANCHOR_STEP_S
    anchor_lo = 0.0
    anchor_hi = rec.duration_s
    n_anchors = int(np.floor((anchor_hi - anchor_lo) / ANCHOR_STEP_S)) + 1
    anchors = anchor_lo + ANCHOR_STEP_S * np.arange(n_anchors)
    
    windows = []
    
    for t_anchor in anchors:
        t_anchor = float(t_anchor)
        phase_name = get_phase_name(t_anchor)
        
        feats = {}
        
        # HR (10-s window)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["hr"]):
            feats["hr"] = float("nan")
        else:
            t_lo, t_hi = t_anchor - half["hr"], t_anchor + half["hr"]
            rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
            feats.update(hr_window_feature(rp, pp.fs_ecg))
            
        # RMSSD + Poincaré (60-s window)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["hrv_rmssd"]):
            feats.update({k: float("nan") for k in ["hrv_rmssd", "sd2_sd1", "samp_en", "ss"]})
        else:
            t_lo, t_hi = t_anchor - half["hrv_rmssd"], t_anchor + half["hrv_rmssd"]
            rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
            feats.update(poincare_window_features(rp, pp.fs_ecg))
            
        # RR (40-s window)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["rr"]):
            feats["rr"] = float("nan")
        else:
            t_lo, t_hi = t_anchor - half["rr"], t_anchor + half["rr"]
            bp = _slice_peaks_by_time(pp.br_peaks, pp.fs_br, pp.br_t0, t_lo, t_hi)
            feats.update(rr_window_feature(bp, pp.fs_br))
            
        # RRV (60-s window)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["rrv"]):
            feats["rrv"] = float("nan")
        else:
            t_lo, t_hi = t_anchor - half["rrv"], t_anchor + half["rrv"]
            bp = _slice_peaks_by_time(pp.br_peaks, pp.fs_br, pp.br_t0, t_lo, t_hi)
            feats.update(rrv_window_feature(bp, pp.fs_br))
            
        # CSI (40-s window)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["csi"]):
            feats["csi"] = float("nan")
        else:
            t_lo, t_hi = t_anchor - half["csi"], t_anchor + half["csi"]
            feats.update(_csi_for_window(pp, t_lo, t_hi))
            
        # Respiratory amplitude (40-s window, BR signal peak-to-trough)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["resp_amp"]):
            feats["resp_amp"] = float("nan")
        else:
            t_lo, t_hi = t_anchor - half["resp_amp"], t_anchor + half["resp_amp"]
            i0 = max(0, int(round((t_lo - pp.br_t0) * pp.fs_br)))
            i1 = min(len(pp.br), int(round((t_hi - pp.br_t0) * pp.fs_br)))
            feats.update(resp_amp_window_feature(pp.br[i0:i1], pp.fs_br))
            
        # Microphone RMS energy (40-s window, Shannon-energy envelope)
        if not is_window_valid(t_anchor, FEATURE_WINDOWS_S["mic_rms"]):
            feats["mic_rms"] = float("nan")
        else:
            t_lo, t_hi = t_anchor - half["mic_rms"], t_anchor + half["mic_rms"]
            i0 = max(0, int(round((t_lo - pp.mic_t0) * pp.fs_mic)))
            i1 = min(len(pp.se), int(round((t_hi - pp.mic_t0) * pp.fs_mic)))
            feats.update(mic_rms_window_feature(pp.se[i0:i1]))
            
        for k, v in list(feats.items()):
            if v is None or not np.isfinite(v):
                feats[k] = float("nan")
                
        windows.append(Window(
            rec_name=rec.name, subject=rec.subject, stressor=rec.stressor,
            phase=phase_name, t_start=t_anchor, t_end=t_anchor,
            features=feats,
        ))
        
    return windows

def main():
    import argparse
    
    base_dir = ROOT
    default_input = os.path.join(base_dir, "data", "mta_5_26_pla_3'30.txt")
    
    parser = argparse.ArgumentParser(description="Extract features from a recording and export to Excel.")
    parser.add_argument("input_file", nargs="?", default=default_input, help="Path to the input text file.")
    parser.add_argument("--output", "-o", help="Path to the output Excel file. (default: biesl-ml/outputs/<input_name>_features.xlsx)")
    
    args = parser.parse_args()
    
    input_file = os.path.abspath(args.input_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if args.output:
        output_excel = os.path.abspath(args.output)
    else:
        input_basename = os.path.basename(input_file)
        name_without_ext, _ = os.path.splitext(input_basename)
        output_excel = os.path.join(base_dir, "outputs", f"{name_without_ext}_features.xlsx")
    
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    
    print(f"Loading recording from {input_file}...")
    rec = load_recording(input_file)
    print("Preprocessing recording (phase_aware=True)...")
    pp = preprocess_recording(rec, phase_aware=True)
    print("Extracting windows and features (0.0 to end)...")
    windows = custom_windows_for_recording(pp)
    
    # Extract lists of values
    times = []
    hr_raw = []
    hrv_raw = []
    rr_raw = []
    
    for w in windows:
        times.append(w.t_start)
        hr_raw.append(w.features.get("hr", np.nan))
        hrv_raw.append(w.features.get("hrv_rmssd", np.nan))
        rr_raw.append(w.features.get("rr", np.nan))
        
    # Create DataFrame with raw features (including NaNs)
    df_raw = pd.DataFrame({
        "time of HR": times,
        "HR data": hr_raw,
        "time of HRV": times,
        "HRV": hrv_raw,
        "time of RR": times,
        "RR data": rr_raw
    })
    
    saved = False
    filename_base, filename_ext = os.path.splitext(output_excel)
    counter = 0
    current_output = output_excel
    
    while not saved:
        try:
            print(f"Saving to Excel: {current_output}...")
            with pd.ExcelWriter(current_output, engine="openpyxl") as writer:
                df_raw.to_excel(writer, sheet_name="Features", index=False)
            saved = True
            output_excel = current_output
        except PermissionError:
            counter += 1
            current_output = f"{filename_base}_{counter}{filename_ext}"
            if counter > 10:
                raise
                
    print(f"Done! Saved file: {output_excel}")
    print("Statistics:")
    print(f"Total window anchors (from 0 to {times[-1]}s): {len(windows)}")
    print(f"NaN count in HR: {df_raw['HR data'].isna().sum()}")
    print(f"NaN count in HRV: {df_raw['HRV'].isna().sum()}")
    print(f"NaN count in RR: {df_raw['RR data'].isna().sum()}")

if __name__ == "__main__":
    main()
