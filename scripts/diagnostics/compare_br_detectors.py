#!/usr/bin/env python
"""Compare three BR peak detectors on every recording, split into phases.

For each recording in data/:
  * filter BR via the median-filter chain (src.preprocess.filter_br)
  * run THREE peak detectors:
      - "global"   : the current detector with a single global p90 prominence
                     floor (src.preprocess.detect_br_peaks).
      - "sliding"  : sliding-window detector with a LOCAL p90 floor per
                     window (src.preprocess.detect_br_peaks_sliding).
      - "neurokit" : neurokit2's nk.rsp_peaks (biosppy method).
  * split each detector's peaks into the three protocol phases
    (rest 0-5 min, stress 5 - 5+stressor min, recovery to end of recording)
    and tabulate breath count + mean RR (bpm) per phase.
  * plot the median-filtered BR signal with each method's peaks on a stacked
    3-row panel, with phase boundaries marked.

Outputs:
  figures/br_compare/<rec>.png           one comparison figure per recording
  outputs/br_detector_compare.json       per-recording, per-phase counts/rates

Usage:
    uv run python scripts/diagnostics/compare_br_detectors.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

from src.io import load_recording, list_recordings, phase_boundaries, resample_uniform   # noqa: E402
from src.preprocess import (              # noqa: E402
    BR_BASELINE_WINDOW_S, BR_SMOOTH_WINDOW_S,
    detect_br_peaks, detect_br_peaks_sliding, detect_br_peaks_neurokit,
    filter_br,
)

warnings.filterwarnings("ignore")
FIG_DIR = Path("figures") / "br_compare"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BR_FS = 100.0
METHODS = ("global", "sliding", "neurokit")
METHOD_COLORS = {"global": "C3", "sliding": "C1", "neurokit": "C2"}


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("'", "_").replace("(", "_").replace(")", "_")


def _detect(method: str, filt: np.ndarray) -> np.ndarray:
    if method == "global":
        peaks, _ = detect_br_peaks(filt, BR_FS)
        return peaks
    if method == "sliding":
        return detect_br_peaks_sliding(filt, BR_FS)
    if method == "neurokit":
        return detect_br_peaks_neurokit(filt, BR_FS)
    raise ValueError(method)


def _per_phase_stats(peaks_idx: np.ndarray, fs: float, phases: dict, t0: float) -> dict:
    """Return {phase: {n_peaks, rate_bpm}}.

    `t0` is the absolute time of sample 0 (the BR channel's first timestamp).
    """
    out = {}
    if len(peaks_idx) == 0:
        return {ph: dict(n_peaks=0, rate_bpm=float("nan")) for ph in phases}
    t_peaks = t0 + peaks_idx / fs                       # absolute seconds
    for ph, (a, b) in phases.items():
        mask = (t_peaks >= a) & (t_peaks < b)
        n = int(mask.sum())
        if n >= 2:
            sub = t_peaks[mask]
            iv = np.diff(sub)
            iv = iv[(iv > 1.0) & (iv < 12.0)]
            rate = float(60.0 / np.mean(iv)) if len(iv) else float("nan")
        else:
            rate = float("nan")
        out[ph] = dict(n_peaks=n, rate_bpm=rate, phase_s=float(b - a))
    return out


def _plot_compare(rec_name: str, t: np.ndarray, filt: np.ndarray,
                  results: dict, phases: dict, out_path: Path,
                  ecg_polarity: str, stressor: str) -> None:
    """Per-recording figure: 3 rows (detectors) 횞 3 cols (rest / stress /
    recovery). Each subplot auto-scales its own y-axis so low-amplitude rest
    breaths aren't crushed by the high-amplitude stress phase.
    """
    phase_order = ["rest", "stress", "recovery"]
    n_rows, n_cols = len(METHODS), len(phase_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7),
                             gridspec_kw=dict(wspace=0.18, hspace=0.45))
    if n_rows == 1:
        axes = np.array([axes])

    for r, method in enumerate(METHODS):
        peaks_all = results[method]["peaks"]
        per = results[method]["per_phase"]
        # absolute time of each detected peak
        t_peaks = t[peaks_all] if len(peaks_all) else np.array([])

        for c, ph in enumerate(phase_order):
            ax = axes[r, c]
            a, b = phases[ph]
            mask_t = (t >= a) & (t < b)
            if not mask_t.any():
                ax.set_axis_off()
                continue
            tt = t[mask_t]
            ss = filt[mask_t]
            ax.plot(tt, ss, color="0.30", lw=0.7)
            if t_peaks.size:
                pmask = (t_peaks >= a) & (t_peaks < b)
                if pmask.any():
                    pt = t_peaks[pmask]
                    pv = filt[peaks_all][pmask]
                    ax.plot(pt, pv, "v", color=METHOD_COLORS[method], ms=4)

            # Robust y-limits per panel: clip out the single huge boundary
            # spikes (motion artefacts at 300 s / end-of-stress) that would
            # otherwise force the y-range to 짹5e6 and crush the breathing
            # waves. Use 1st-99th percentile with a small padding.
            if ss.size:
                lo, hi = np.percentile(ss, [1.0, 99.0])
                pad = max(0.10 * max(abs(lo), abs(hi)), 1e-9)
                ax.set_ylim(lo - pad, hi + pad)

            # subtitle: per-phase count + rate
            rate = per[ph]["rate_bpm"]
            n = per[ph]["n_peaks"]
            rate_s = f"{rate:.1f}/min" if rate == rate else "n/a"
            ax.set_title(f"{method} 쨌 {ph}   n={n}   RR={rate_s}", fontsize=8)
            ax.grid(alpha=0.3, lw=0.4)
            if r == n_rows - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel("BR (filt)", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{rec_name}   stressor={stressor}   ecg={ecg_polarity}   "
        f"(median win: base {BR_BASELINE_WINDOW_S}s / smooth {BR_SMOOTH_WINDOW_S}s)",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def analyze_recording(path: str) -> dict:
    rec = load_recording(path)
    raw = resample_uniform(rec.channels["br"], BR_FS).astype(np.float64)
    filt = filter_br(raw, BR_FS)
    t0 = float(rec.channels["br"][0, 0])
    t = t0 + np.arange(len(filt)) / BR_FS
    phases = phase_boundaries(rec)

    results = {}
    for m in METHODS:
        peaks = _detect(m, filt)
        results[m] = dict(
            peaks=peaks,
            per_phase=_per_phase_stats(peaks, BR_FS, phases, t0),
            n_total=int(len(peaks)),
        )

    out_path = FIG_DIR / f"{_slug(rec.name)}.png"
    _plot_compare(rec.name, t, filt, results, phases, out_path,
                  rec.ecg_polarity, rec.stressor)

    return dict(
        name=rec.name, slug=_slug(rec.name), stressor=rec.stressor,
        ecg_polarity=rec.ecg_polarity, duration_s=float(rec.duration_s),
        phases={ph: dict(start=float(a), end=float(b)) for ph, (a, b) in phases.items()},
        per_method={m: dict(n_total=results[m]["n_total"],
                            per_phase=results[m]["per_phase"]) for m in METHODS},
    )


def main() -> None:
    files = list_recordings("data")
    print(f"Comparing 3 detectors on {len(files)} recordings\n")
    all_rows = []
    for p in files:
        try:
            row = analyze_recording(p)
            all_rows.append(row)
            g = row["per_method"]["global"]["n_total"]
            s = row["per_method"]["sliding"]["n_total"]
            k = row["per_method"]["neurokit"]["n_total"]
            print(f"  {row['name']:34s}  global={g:3d}  sliding={s:3d}  neurokit={k:3d}")
        except Exception as e:
            print(f"  !! {p}: {e}", file=sys.stderr)
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/br_detector_compare.json", "w") as fh:
        json.dump(all_rows, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    print(f"\nSaved {len(all_rows)} comparison plots under {FIG_DIR}/")
    print(f"Wrote outputs/br_detector_compare.json")


if __name__ == "__main__":
    main()

