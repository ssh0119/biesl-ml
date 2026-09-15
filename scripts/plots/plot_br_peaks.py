#!/usr/bin/env python
"""Plot median-filtered breathing (BR) signal + detected peaks for every recording.

For each recording in data/:
  * resample BR to the working rate (100 Hz)
  * filter with the median-filter chain (src.preprocess.filter_br)
  * detect breath peaks (src.preprocess.detect_br_peaks)
  * plot raw-vs-filtered + peaks, with the 5-/10-min protocol boundaries marked

Saves one PNG per recording under figures/br/ and prints a per-recording
summary line (breath count, mean breath rate) to stdout.

Usage:
    uv run python scripts/plots/plot_br_peaks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from src.io import list_recordings, load_recording, phase_boundaries, resample_uniform  # noqa: E402
from src.preprocess import (  # noqa: E402
    BR_BASELINE_WINDOW_S,
    BR_SMOOTH_WINDOW_S,
    detect_br_peaks_neurokit,
    filter_br,
)

FIG_DIR = Path("figures") / "br"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BR_FS = 100.0


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("'", "_").replace("(", "_").replace(")", "_")


def plot_recording(path: str) -> dict:
    rec = load_recording(path)
    br_t0 = float(rec.channels["br"][0, 0])
    raw = resample_uniform(rec.channels["br"], BR_FS).astype(np.float64)
    filt = filter_br(raw, BR_FS)
    peaks = detect_br_peaks_neurokit(filt, BR_FS)

    t = br_t0 + np.arange(len(filt)) / BR_FS
    dur = len(filt) / BR_FS

    # breath rate from accepted peaks
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / BR_FS
        ibis = ibis[(ibis > 1.0) & (ibis < 12.0)]
        mean_rr = 60.0 / np.mean(ibis) if len(ibis) else float("nan")
    else:
        mean_rr = float("nan")

    fig, ax = plt.subplots(figsize=(16, 3.4))

    # raw resampled BR on a light twin axis (z-scored for shape comparison)
    ax2 = ax.twinx()
    raw_z = (raw - np.median(raw)) / (np.std(raw) or 1.0)
    ax2.plot(t, raw_z, color="0.80", lw=0.5, label="raw BR (z, resampled)")
    ax2.set_yticks([])

    ax.plot(t, filt, color="C0", lw=0.8, label="median-filtered BR")
    if len(peaks):
        ax.plot(t[peaks], filt[peaks], "rv", ms=5, label=f"breath peaks (n={len(peaks)})")

    # protocol boundaries
    for b, lbl in ((300.0, "5 min"), (600.0, "10 min")):
        if br_t0 <= b <= br_t0 + dur:
            ax.axvline(b, color="k", ls="--", lw=0.7, alpha=0.5)
            ax.text(b, ax.get_ylim()[1], f" {lbl}", va="top", ha="left", fontsize=7, color="k")

    ax.set_title(
        f"{rec.name}   |   stressor={rec.stressor}   ecg={rec.ecg_polarity}   "
        f"breaths={len(peaks)}   mean RR={mean_rr:.1f} bpm   "
        f"(median win: base {BR_BASELINE_WINDOW_S}s / smooth {BR_SMOOTH_WINDOW_S}s)",
        fontsize=9,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("BR (filtered)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, ncol=3)
    plt.tight_layout()

    out = FIG_DIR / f"{_slug(rec.name)}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)

    return dict(name=rec.name, slug=_slug(rec.name), stressor=rec.stressor,
                ecg=rec.ecg_polarity, n_peaks=len(peaks), mean_rr=mean_rr,
                duration_s=dur)


def main() -> None:
    files = list_recordings("data")
    print(f"Plotting BR peaks for {len(files)} recordings "
          f"(median filter: base {BR_BASELINE_WINDOW_S}s / smooth {BR_SMOOTH_WINDOW_S}s)\n")
    rows = []
    for p in files:
        try:
            r = plot_recording(p)
            rows.append(r)
            print(f"  {r['name']:34s}  {r['stressor']:5s}  "
                  f"breaths={r['n_peaks']:3d}  RR={r['mean_rr']:.1f} bpm  -> {r['slug']}.png")
        except Exception as e:
            print(f"  !! {p}: {e}", file=sys.stderr)

    import json
    (FIG_DIR / "_index.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} PNGs under {FIG_DIR}/")


if __name__ == "__main__":
    main()

