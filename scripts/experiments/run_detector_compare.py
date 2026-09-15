#!/usr/bin/env python
"""Run scripts/evaluation/run_split_reports.py for each BR peak detector.

For each detector ??{global, sliding, neurokit} and each class config
{without_math, with_math}, train + evaluate KNN/RF/XGBoost + 1D-CNN under
LORO and 70:15:15 random split, and save:

  outputs/split_reports_<detector>.json
  figures/{cfg}/confusion_<detector>/*.png

The classical pipeline depends on the BR detector through `rr`/`rrv`; the
1D-CNN reads the filtered BR waveform so its inputs are identical across
detectors (so we still run it inside each detector loop for a clean,
single-source-of-truth output JSON, accepting the small redundant cost).
"""
from __future__ import annotations

import os
import shutil
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DETECTORS = ("global", "sliding", "neurokit")


def run_for_detector(detector: str) -> None:
    print(f"\n{'#'*72}\n# detector: {detector}\n{'#'*72}")
    env = {**os.environ, "BR_PEAK_METHOD": detector}
    # The run script writes to outputs/split_reports.json and to
    # figures/{with,without}_math/confusion/. Move both aside after.
    subprocess.run(["uv", "run", "python", "scripts/evaluation/run_split_reports.py"],
                   cwd=str(ROOT), check=True, env=env)
    # Move outputs/split_reports.json -> outputs/split_reports_<detector>.json
    src_json = ROOT / "outputs" / "split_reports.json"
    dst_json = ROOT / "outputs" / f"split_reports_{detector}.json"
    if src_json.exists():
        shutil.move(src_json, dst_json)
        print(f"  -> {dst_json}")
    # Move figures/{cfg}/confusion -> figures/{cfg}/confusion_<detector>
    for cfg in ("without_math", "with_math"):
        src = ROOT / "figures" / cfg / "confusion"
        dst = ROOT / "figures" / cfg / f"confusion_{detector}"
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.move(src, dst)
            print(f"  -> {dst}")


def main() -> None:
    for d in DETECTORS:
        run_for_detector(d)
    print("\nDone. Per-detector outputs:")
    for d in DETECTORS:
        print(f"  outputs/split_reports_{d}.json")


if __name__ == "__main__":
    main()

