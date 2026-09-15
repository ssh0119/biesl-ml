import json
import os
import sys
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(base_dir, "outputs", "preprocessed_nn.json")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist. Running dump_preprocessed_nn.py...")
        # Run it
        sys.path.insert(0, base_dir)
        from scripts.exports import dump_preprocessed_nn
        dump_preprocessed_nn.dump()
        
    with open(json_path) as f:
        data = json.load(f)
        
    rec_name = "oyj_6_6_math_11"
    if rec_name not in data:
        print(f"Error: {rec_name} not found in json.")
        return
        
    phases = data[rec_name]["phases"]
    print(f"Phases available for {rec_name}: {list(phases.keys())}")
    
    for act, blk in phases.items():
        nn = np.asarray(blk["nn_ms"], dtype=float)
        if len(nn) < 3:
            print(f"Phase {act} has too few NN intervals.")
            continue
            
        # Compute RMSSD of this phase
        diff = np.diff(nn)
        rmssd = np.sqrt(np.mean(diff ** 2))
        
        # Compute SDSD
        sdsd = np.std(diff, ddof=1)
        
        # Compute SD1 and SD2
        sdnn = np.std(nn, ddof=1)
        sd1 = sdsd / np.sqrt(2.0)
        var_diff = 2.0 * sdnn * sdnn - sd1 * sd1
        sd2 = np.sqrt(var_diff) if var_diff > 0 else float("nan")
        
        print(f"\nPhase: {act}")
        print(f"  Number of NN intervals: {len(nn)}")
        print(f"  Mean NN: {np.mean(nn):.2f} ms")
        print(f"  SDNN: {sdnn:.4f} ms")
        print(f"  SDSD: {sdsd:.4f} ms")
        print(f"  RMSSD: {rmssd:.4f} ms")
        print(f"  SD1: {sd1:.4f} ms")
        print(f"  RMSSD / sqrt(2): {rmssd / np.sqrt(2.0):.4f} ms")
        print(f"  SD1 * sqrt(2): {sd1 * np.sqrt(2.0):.4f} ms (should match SDSD)")
        print(f"  SD2: {sd2:.4f} ms")

if __name__ == "__main__":
    main()

