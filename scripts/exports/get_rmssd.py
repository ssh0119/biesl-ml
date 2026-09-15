import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.io import load_recording
from src.features import preprocess_recording, windows_for_recording
from src.io import list_recordings

def main():
    base_dir = ROOT
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    recordings = list_recordings(data_dir)
    if not recordings:
        print(f"No recordings found in data directory: {data_dir}")
        return
        
    print(f"Found {len(recordings)} recordings to process.")
    for rec_path in recordings:
        rec_name = os.path.splitext(os.path.basename(rec_path))[0]
        out_csv = os.path.join(out_dir, f"{rec_name}_hr_rmssd.csv")
        
        try:
            print(f"Processing {rec_name}...")
            rec = load_recording(rec_path)
            pp = preprocess_recording(rec, phase_aware=True)
            windows = windows_for_recording(pp)
            
            with open(out_csv, "w") as f:
                f.write("Time_s,HR_bpm,RMSSD_ms\n")
                for w in windows:
                    t = w.t_start  # Anchor time in seconds
                    hr = w.features.get("hr", float("nan"))
                    rmssd = w.features.get("hrv_rmssd", float("nan"))
                    f.write(f"{t:.1f},{hr:.4f},{rmssd:.4f}\n")
            print(f"  -> Wrote CSV to {out_csv}")
        except Exception as e:
            print(f"Error processing {rec_name}: {e}")
            
    print("Done processing all files.")

if __name__ == "__main__":
    main()
