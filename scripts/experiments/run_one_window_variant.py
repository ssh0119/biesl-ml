#!/usr/bin/env python
"""Run classical + 1D-CNN pipelines for ONE window configuration and write
all four JSONs into a given out_dir. Honors the WINDOW_S/OVERLAP defined in
src/features.py at import time, so the caller should edit src/features.py
*before* invoking this script (subprocess restart picks up the new values).

Usage:
    uv run python scripts/experiments/run_one_window_variant.py outputs/win40_ov20
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: run_one_window_variant.py <out_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = sys.argv[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # report the captured constants ??sanity check for the caller
    from src.features import OVERLAP, WINDOW_S
    from src.raw_windows import WINDOW_N
    print(f"[variant] WINDOW_S={WINDOW_S}  OVERLAP={OVERLAP}  "
          f"(raw_windows WINDOW_N={WINDOW_N})  out_dir={out_dir}")

    from src.local_eval import (
        run_local_loro_eval,
        run_random_split_eval as classical_random_split_eval,
    )
    from src.dl_train import (
        run_local_eval as dl_loro_eval,
        run_random_split_eval as dl_random_split_eval,
    )

    print(f"\n=== {out_dir}: Classical LORO ===")
    run_local_loro_eval(out_dir=out_dir)
    print(f"\n=== {out_dir}: Classical random split ===")
    classical_random_split_eval(out_dir=out_dir)
    print(f"\n=== {out_dir}: 1D-CNN LORO ===")
    dl_loro_eval(out_dir=out_dir)
    print(f"\n=== {out_dir}: 1D-CNN random split ===")
    dl_random_split_eval(out_dir=out_dir)

    print(f"\n[variant] done. Wrote JSONs to {out_dir}/")


if __name__ == "__main__":
    main()

