"""Per-window feature extraction.

Eight features per window — no others:
    csi         : S2/S1 ratio of Shannon-energy peaks paired to ECG R-peaks
    hr          : 60_000 / mean(NN_ms)
    hrv_rmssd   : root-mean-square of successive NN differences (ms)
    sd2_sd1     : Poincaré SD2 / SD1   (higher → more sympathetic dominance)
    samp_en     : Sample Entropy (via neurokit2, unitless)
    ss          : Stress Score = 1000 / SD2  (Naranjo / Kubios)
    rr          : 60 / mean(breath interval, s)        — breaths per minute
    rrv         : std of the last 5 breath intervals (s)

SD1 and SD2 are NOT in the feature set — SD1 ≈ RMSSD / √2 (redundant
with RMSSD) and SS ≡ 1000 / SD2 (redundant with SD2). They're still
computed internally to derive `sd2_sd1`, `ss`.

LF / HF / LF-HF were removed earlier: they need a 2-5 min window for a
stable Welch spectrum and were unreliable at 40 s.

Anchor-based windowing — see `windows_for_recording`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .io import Recording, channel_fs, phase_boundaries, resample_uniform
from .preprocess import (
    calculate_cardiac_snr,
    classify_s1_s2,
    classify_s1_s2_indices,
    clean_nn_intervals,
    detect_br_peaks,
    detect_br_peaks_neurokit,
    detect_br_peaks_sliding,
    detect_ecg_rpeaks,
    detect_ecg_rpeaks_per_phase,
    detect_pcg_peaks,
    filter_br,
    filter_ecg,
    filter_mic_pcg,
    shannon_energy,
)


# All rows share a common 2-s anchor step. Each feature is then computed over
# its OWN window length, centered on the anchor — different physiological
# quantities stabilise at different timescales (HR converges in ~10 s;
# Poincaré / RMSSD / RRV need ~60 s).
ANCHOR_STEP_S = 2.0

# Per-feature window lengths (seconds), centered on the anchor.
# SD1 ≈ RMSSD/√2 and SS = 1000/SD2, so SD1 and SD2 are intentionally NOT
# in the training-feature set — they would be redundant with RMSSD / SS.
# They're still computed internally to derive SD2/SD1 and SD1·SD2 (the
# Poincaré ellipse area, modulo π).
FEATURE_WINDOWS_S: dict[str, float] = {
    "csi":         40.0,
    "hr":          10.0,
    "hrv_rmssd":   60.0,
    "sd2_sd1":     60.0,
    "samp_en":     60.0,
    "ss":          60.0,
    "rr":          40.0,
    "rrv":         60.0,
    "resp_amp":    40.0,  # peak-to-trough amplitude of filtered respiration signal
    "mic_rms":     40.0,  # RMS energy of Shannon-energy microphone envelope
}
MAX_FEATURE_WINDOW_S: float = max(FEATURE_WINDOWS_S.values())

# Legacy aliases (some downstream code, the CNN, and reports still reference
# a single window length — keep it pointed at the longest feature window so
# raw-window extents don't lose information vs the classical features.
WINDOW_S = 40.0          # the 1D-CNN raw-window length (centered on anchor)
OVERLAP = 1.0 - ANCHOR_STEP_S / WINDOW_S      # implied for legacy callers

FEATURE_NAMES = (
    "csi",
    "hr",
    "hrv_rmssd",
    "sd2_sd1",
    "samp_en",
    "ss",
    "rr",
    "rrv",
    "resp_amp",  # peak-to-trough amplitude of filtered respiration (arbitrary units)
    "mic_rms",  # RMS energy of Shannon-energy microphone envelope (arbitrary units)
)

# Optional temperature features — NOT in features.pdf, kept only because
# legacy ablation runners may still reference them. New code should NOT
# include them (the project no longer uses temperature).
TEMP_FEATURE_NAMES = (
    "temp_mean_C",
    "temp_std_C",
    "temp_slope_Cps",
)

# Skip windows whose extent touches the rest → stress transition at 300 s.
# Patient reports asymmetric discomfort: 10 s of overlap *before* the cue and
# 30 s of overlap *after*. The 600 s (stress → recovery) boundary is not
# explicitly skipped because the recovery phase is already dropped from the
# dataset (see assign_activity in src.pipeline).
BOUNDARY_TIMES_S = (300.0,)
BOUNDARY_BUFFER_PRE_S = 10.0
BOUNDARY_BUFFER_POST_S = 30.0


# BR peak detector selection (swap-able for ablations). Allowed values:
# "neurokit" (default), "sliding", "global". Set the module-level variable
# or pass via the BR_PEAK_METHOD env var (read once on import) — used by
# scripts/evaluation/run_split_reports.py and related evaluation tools.
import os as _os
BR_PEAK_METHOD: str = _os.environ.get("BR_PEAK_METHOD", "sliding")


def _window_touches_boundary(t_start: float, t_end: float) -> bool:
    """True iff the window [t_start, t_end] overlaps the asymmetric exclusion
    zone around any protocol boundary: [b − BOUNDARY_BUFFER_PRE_S,
    b + BOUNDARY_BUFFER_POST_S].
    """
    for b in BOUNDARY_TIMES_S:
        if t_start <= b + BOUNDARY_BUFFER_POST_S and t_end >= b - BOUNDARY_BUFFER_PRE_S:
            return True
    return False


def _anchor_in_boundary(t_anchor: float) -> bool:
    """True iff anchor time `t_anchor` lies inside the asymmetric exclusion
    zone around any protocol boundary.
    """
    for b in BOUNDARY_TIMES_S:
        if b - BOUNDARY_BUFFER_PRE_S <= t_anchor <= b + BOUNDARY_BUFFER_POST_S:
            return True
    return False


@dataclass
class Window:
    rec_name: str
    subject: str
    stressor: str
    phase: str               # rest | stress | recovery (label)
    t_start: float
    t_end: float
    features: dict[str, float]


# ---- HRV time / frequency / non-linear helpers -----------------------------

def _hr_from_nn(nn_ms: np.ndarray) -> float:
    if len(nn_ms) == 0:
        return float("nan")
    mean_nn = float(np.mean(nn_ms))
    if mean_nn <= 0:
        return float("nan")
    return 60_000.0 / mean_nn


def _rmssd(nn_ms: np.ndarray) -> float:
    if len(nn_ms) < 2:
        return float("nan")
    d = np.diff(nn_ms)
    return float(np.sqrt(np.mean(d * d)))


def _poincare(nn_ms: np.ndarray) -> dict[str, float]:
    """Poincaré non-linear HRV features.

    Always computed (returned in the dict, even though SD1 / SD2 themselves
    are *not* in the training feature set):

        sd1       : SD of NN points perpendicular to identity line
                    = √(½·Var(diff(NN))) = SDSD / √2                  (ms)
        sd2       : SD of NN points along identity line
                    = √(2·SDNN² − SD1²)                                (ms)

    Derived features (these ARE in the training feature set):

        sd2_sd1   : SD2 / SD1   (higher → more sympathetic dominance)
        sd1_x_sd2 : SD1 · SD2   (Poincaré ellipse area / π, ms²)
        ss        : 1000 / SD2  (Naranjo / Kubios stress score)

    NaN where the input is too short or numerically ill-conditioned.
    """
    nan = float("nan")
    out = {"sd1": nan, "sd2": nan, "sd2_sd1": nan, "sd1_x_sd2": nan, "ss": nan}
    if len(nn_ms) < 3:
        return out
    sdnn = float(np.std(nn_ms, ddof=1))
    sdsd = float(np.std(np.diff(nn_ms), ddof=1))
    sd1 = sdsd / np.sqrt(2.0)
    var_diff = 2.0 * sdnn * sdnn - sd1 * sd1
    sd2 = float(np.sqrt(var_diff)) if var_diff > 0 else nan
    out["sd1"] = float(sd1)
    out["sd2"] = sd2
    if np.isfinite(sd2) and sd2 > 0 and sd1 > 0:
        out["sd2_sd1"] = float(sd2 / sd1)
        out["sd1_x_sd2"] = float(sd1 * sd2)
        out["ss"] = float(1000.0 / sd2)
    return out


# ---- per-window feature extractors ----------------------------------------

def hr_window_feature(rpeaks_idx_window: np.ndarray, fs_ecg: float) -> dict[str, float]:
    """Mean HR (bpm) from R-peaks falling in the short HR window.

    HR is fast-converging (a 10-s window covers ~12 beats at 70 bpm) so the
    Poincaré / RMSSD pieces, which need 60 s, are computed separately below.
    """
    nn = clean_nn_intervals(rpeaks_idx_window, fs_ecg)
    if len(nn) < 2:
        return {"hr": float("nan")}
    return {"hr": _hr_from_nn(nn)}


def poincare_window_features(rpeaks_idx_window: np.ndarray, fs_ecg: float) -> dict[str, float]:
    """RMSSD + SD2/SD1 + Sample Entropy + SS from R-peaks falling in the long
    Poincaré window (60 s). SD1 and SD2 are derived internally but not
    returned (they would be redundant with RMSSD and SS in training).
    All NaN if fewer than 3 NN intervals survive cleaning.
    """
    keys = ("hrv_rmssd", "sd2_sd1", "samp_en", "ss")
    nn = clean_nn_intervals(rpeaks_idx_window, fs_ecg)
    if len(nn) < 3:
        return {k: float("nan") for k in keys}
    poin = _poincare(nn)

    # Compute Sample Entropy using neurokit2
    samp_en = float("nan")
    try:
        # Monkey patch np.trapz to prevent NumPy 2.0+ incompatibility in older neurokit2 versions
        if not hasattr(np, "trapz"):
            np.trapz = getattr(np, "trapezoid", None)
        import neurokit2 as nk
        peaks = nk.intervals_to_peaks(nn)
        hrv_nl = nk.hrv_nonlinear(peaks, sampling_rate=1000)
        if hasattr(hrv_nl, "loc"):
            samp_val = hrv_nl["HRV_SampEn"].iloc[0]
        else:
            samp_val = hrv_nl["HRV_SampEn"]
        samp_en = float(samp_val)
    except Exception:
        pass

    return dict(
        hrv_rmssd=_rmssd(nn),
        sd2_sd1=poin["sd2_sd1"],
        samp_en=samp_en,
        ss=poin["ss"],
    )


def _breath_intervals(peaks_idx: np.ndarray, fs_br: float) -> np.ndarray:
    if len(peaks_idx) < 2:
        return np.empty(0, dtype=float)
    iv = np.diff(peaks_idx) / fs_br
    return iv[(iv > 1.0) & (iv < 12.0)]


def rr_window_feature(peaks_idx_in_window: np.ndarray, fs_br: float) -> dict[str, float]:
    """RR (breath rate, bpm) from the 40-s BR-peak window."""
    iv = _breath_intervals(peaks_idx_in_window, fs_br)
    if len(iv) < 1:
        return {"rr": float("nan")}
    return {"rr": 60.0 / float(np.mean(iv))}


def rrv_window_feature(peaks_idx_in_window: np.ndarray, fs_br: float) -> dict[str, float]:
    """RRV (std of the last 5 breath intervals, seconds) from the 60-s BR
    window. Returns NaN if fewer than 2 valid intervals are present.
    """
    iv = _breath_intervals(peaks_idx_in_window, fs_br)
    if len(iv) < 2:
        return {"rrv": float("nan")}
    return {"rrv": float(np.std(iv[-5:], ddof=0))}


# Legacy combined helper retained for any external callers (deprecated).
def rr_window_features(peaks_idx_in_window: np.ndarray, fs_br: float) -> dict[str, float]:
    out = {}
    out.update(rr_window_feature(peaks_idx_in_window, fs_br))
    out.update(rrv_window_feature(peaks_idx_in_window, fs_br))
    return out


def resp_amp_window_feature(br_slice: np.ndarray, fs_br: float) -> dict[str, float]:
    """Respiratory amplitude: mean peak-to-trough amplitude of the filtered BR signal.

    Detects peaks (inhalation peaks) and valleys (exhalation troughs) in the
    provided BR signal slice using scipy.signal.find_peaks with a minimum
    separation of 1 second (handles up to 60 breaths/min) and an adaptive
    prominence threshold proportional to local variance to suppress noise.

    Returns the mean of detected peaks minus the mean of detected valleys.
    Returns NaN if fewer than 1 peak or 1 valley is found.
    """
    from scipy.signal import find_peaks as _find_peaks
    nan = float("nan")
    if len(br_slice) < int(fs_br * 2.0):
        return {"resp_amp": nan}
    min_dist = max(1, int(fs_br * 1.0))  # minimum 1 s between peaks

    # Adaptive prominence floor (0.25 * standard deviation of the slice)
    std_val = np.std(br_slice)
    prom = 0.25 * std_val if std_val > 1e-6 else None

    peaks, _   = _find_peaks( br_slice, distance=min_dist, prominence=prom)
    valleys, _ = _find_peaks(-br_slice, distance=min_dist, prominence=prom)

    # Relax to 1 peak / 1 valley to handle slow respiration windows without losing data
    if len(peaks) < 1 or len(valleys) < 1:
        return {"resp_amp": nan}
    amp = float(np.mean(br_slice[peaks])) - float(np.mean(br_slice[valleys]))
    return {"resp_amp": amp if np.isfinite(amp) and amp >= 0 else nan}


def mic_rms_window_feature(se_slice: np.ndarray) -> dict[str, float]:
    """RMS energy of the Shannon-energy microphone envelope over the window.

    Physical exertion (plank) causes deep breathing sounds, muscle strain, and
    fabric noise that raise the acoustic energy envelope well above the quiet
    levels seen during mental stress (math) or rest.

    Returns NaN if the slice is empty.
    """
    if len(se_slice) == 0:
        return {"mic_rms": float("nan")}
    return {"mic_rms": float(np.sqrt(np.mean(se_slice ** 2)))}


def csi_window_features(s1_amps: np.ndarray, s2_amps: np.ndarray) -> dict[str, float]:
    """CSI = mean(S2 amplitude) / mean(S1 amplitude) over the window."""
    if len(s1_amps) == 0 or len(s2_amps) == 0:
        return dict(csi=float("nan"))
    s1_mean = float(np.mean(s1_amps))
    s2_mean = float(np.mean(s2_amps))
    csi = float(s2_mean / s1_mean) if s1_mean > 0 else float("nan")
    return dict(csi=csi)


def temp_features_window(temp_ch: np.ndarray, t_start: float, t_end: float) -> dict[str, float]:
    """Optional skin-temperature features for the ablation comparison.

    NOT part of features.pdf. Returns mean / std / linear-fit slope of the
    raw 2xN (time, value) temperature channel slice in [t_start, t_end).
    """
    t, v = temp_ch[0], temp_ch[1]
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 2:
        return {k: float("nan") for k in TEMP_FEATURE_NAMES}
    tt = t[mask]
    vv = v[mask]
    out: dict[str, float] = {
        "temp_mean_C": float(np.mean(vv)),
        "temp_std_C":  float(np.std(vv)),
        "temp_slope_Cps": float(np.polyfit(tt - tt[0], vv, 1)[0]) if len(tt) >= 3 else float("nan"),
    }
    return out


# ---- recording-level preprocessing ----------------------------------------

@dataclass
class Preprocessed:
    rec: Recording
    fs_ecg: float
    ecg: np.ndarray
    ecg_t0: float
    rpeaks: np.ndarray
    fs_br: float
    br: np.ndarray
    br_t0: float
    br_peaks: np.ndarray
    fs_mic: float
    se: np.ndarray              # Shannon-energy envelope at fs_mic
    pcg_peaks: np.ndarray       # indices into `se`
    mic_t0: float


def preprocess_recording(
    rec: Recording,
    ecg_fs_target: float = 500.0,
    br_fs_target: float = 100.0,
    mic_fs_target: float = 2000.0,
    *,
    phase_aware: bool = False,
    rpeak_method: str = "neurokit",
) -> Preprocessed:
    """Resample to common grids and run the PDF's per-modality pipelines.

    `phase_aware`: when True, run R-peak detection independently per protocol
    phase (rest / stress / recovery). Isolates adaptive detector state so
    plank-noise can't contaminate adjacent phases.
    `rpeak_method`: forwarded to neurokit2.ecg_peaks (default pantompkins1985).
    """
    # ECG @ 500 Hz (matches local native; gives good QRS resolution)
    ecg_t0 = float(rec.channels["ecg"][0, 0])
    ecg_u = resample_uniform(rec.channels["ecg"], ecg_fs_target).astype(np.float64)
    ecg_f = filter_ecg(ecg_u, ecg_fs_target)
    if phase_aware:
        phases = phase_boundaries(rec)
        rpeaks = detect_ecg_rpeaks_per_phase(
            ecg_f, ecg_fs_target, ecg_t0, phases, method=rpeak_method,
        )
    else:
        rpeaks = detect_ecg_rpeaks(
            ecg_f, ecg_fs_target, polarity=rec.ecg_polarity, method=rpeak_method,
        )

    # BR @ 100 Hz (≥ 4× the Cheby II stopband edge of 1 Hz, comfortable margin)
    br_t0 = float(rec.channels["br"][0, 0])
    br_u = resample_uniform(rec.channels["br"], br_fs_target).astype(np.float64)
    br_f = filter_br(br_u, br_fs_target)
    # BR peak detector is selectable via BR_PEAK_METHOD (module-level / env
    # var) so we can ablate global vs sliding vs neurokit on the classical
    # pipeline. neurokit wins the head-to-head — see
    # br-detector-comparison.md — and is the default.
    if BR_PEAK_METHOD == "global":
        br_peaks, _ = detect_br_peaks(br_f, br_fs_target)
    elif BR_PEAK_METHOD == "sliding":
        br_peaks = detect_br_peaks_sliding(br_f, br_fs_target)
    else:
        br_peaks = detect_br_peaks_neurokit(br_f, br_fs_target)

    # Mic @ 2 kHz (matches native rate, Nyquist 1 kHz ≫ 200 Hz PCG band)
    mic_t0 = float(rec.channels["mic"][0, 0])
    mic_u = resample_uniform(rec.channels["mic"], mic_fs_target).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, mic_fs_target)
    se = shannon_energy(mic_f, smooth_ms=50.0, fs=mic_fs_target)
    pcg_peaks = detect_pcg_peaks(se, mic_fs_target)

    return Preprocessed(
        rec=rec,
        fs_ecg=ecg_fs_target, ecg=ecg_f, ecg_t0=ecg_t0, rpeaks=rpeaks,
        fs_br=br_fs_target, br=br_f, br_t0=br_t0, br_peaks=br_peaks,
        fs_mic=mic_fs_target, se=se, pcg_peaks=pcg_peaks, mic_t0=mic_t0,
    )


# ---- windowing ------------------------------------------------------------

def _slice_peaks_by_time(peaks_idx: np.ndarray, fs: float, t0: float,
                         t_start: float, t_end: float) -> np.ndarray:
    """Peak indices whose absolute time falls in [t_start, t_end)."""
    if len(peaks_idx) == 0:
        return peaks_idx
    t = t0 + peaks_idx / fs
    mask = (t >= t_start) & (t < t_end)
    return peaks_idx[mask]


def _csi_for_window(pp: Preprocessed, t_lo: float, t_hi: float) -> dict[str, float]:
    """Compute CSI over absolute time window [t_lo, t_hi]: slice Shannon-energy
    envelope + paired-S1/S2 detection against R-peaks falling in the window.
    """
    rp_window = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
    mic_t0 = pp.mic_t0
    def t_to_idx(t_abs: float) -> int:
        return int(round((t_abs - mic_t0) * pp.fs_mic))
    i0_mic = max(0, t_to_idx(t_lo))
    i1_mic = min(len(pp.se), t_to_idx(t_hi))
    if i1_mic <= i0_mic:
        return {"csi": float("nan")}
    se_slice = pp.se[i0_mic:i1_mic]
    pcg_slice = pp.pcg_peaks[(pp.pcg_peaks >= i0_mic) & (pp.pcg_peaks < i1_mic)] - i0_mic
    r_abs_t = pp.ecg_t0 + rp_window / pp.fs_ecg
    r_se_idx = np.round((r_abs_t - t_lo) * pp.fs_mic).astype(int)
    r_se_idx = r_se_idx[(r_se_idx >= 0) & (r_se_idx < len(se_slice))]
    s1_amps, s2_amps = classify_s1_s2(se_slice, pp.fs_mic, pcg_slice, r_se_idx)
    return csi_window_features(s1_amps, s2_amps)


def cardiac_snr_for_window(
    pp: Preprocessed,
    t_lo: float,
    t_hi: float,
    *,
    s1_win_s: float = 0.10,
    s2_win_s: float = 0.10,
) -> float:
    """Compute cardiac sound Signal-to-Noise Ratio (SNR in dB) over absolute time window [t_lo, t_hi].

    S1 and S2 peaks are extracted from the Shannon-energy envelope. The S1/S2 sound component windows
    are defined as 'signal', and the remaining window samples are defined as 'noise'.
    """
    rp_window = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
    mic_t0 = pp.mic_t0
    def t_to_idx(t_abs: float) -> int:
        return int(round((t_abs - mic_t0) * pp.fs_mic))
    i0_mic = max(0, t_to_idx(t_lo))
    i1_mic = min(len(pp.se), t_to_idx(t_hi))
    if i1_mic <= i0_mic:
        return float("nan")

    se_slice = pp.se[i0_mic:i1_mic]
    pcg_slice = pp.pcg_peaks[(pp.pcg_peaks >= i0_mic) & (pp.pcg_peaks < i1_mic)] - i0_mic
    r_abs_t = pp.ecg_t0 + rp_window / pp.fs_ecg
    r_se_idx = np.round((r_abs_t - t_lo) * pp.fs_mic).astype(int)
    r_se_idx = r_se_idx[(r_se_idx >= 0) & (r_se_idx < len(se_slice))]

    s1_idxs, s2_idxs = classify_s1_s2_indices(se_slice, pp.fs_mic, pcg_slice, r_se_idx)
    if len(s1_idxs) == 0 and len(s2_idxs) == 0:
        return float("nan")

    mic_u = resample_uniform(pp.rec.channels["mic"], pp.fs_mic).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, pp.fs_mic)
    mic_f_slice = mic_f[i0_mic:i1_mic]

    return calculate_cardiac_snr(
        mic_f_slice, pp.fs_mic, s1_idxs, s2_idxs,
        s1_win_s=s1_win_s, s2_win_s=s2_win_s,
    )


def windows_for_recording(
    pp: Preprocessed,
    *,
    include_temp: bool = False,
) -> list[Window]:
    """Anchor-based per-feature windowing.

    Anchors step every `ANCHOR_STEP_S` seconds across each non-recovery
    phase. At each anchor `t`, every feature is computed over its own window
    `[t − W/2, t + W/2]` (W from `FEATURE_WINDOWS_S`).

    An anchor is dropped if:
      * the asymmetric protocol-boundary buffer covers it
        (`_anchor_in_boundary` — defaults to [290 s, 330 s] around the rest →
        stress transition), OR
      * the **longest** feature window centered on the anchor would extend
        outside the phase the anchor sits in (no cross-phase mixing).
    """
    rec = pp.rec
    phases = phase_boundaries(rec)
    half_max = MAX_FEATURE_WINDOW_S / 2.0
    half: dict[str, float] = {k: v / 2.0 for k, v in FEATURE_WINDOWS_S.items()}

    windows: list[Window] = []

    for phase_name, (p_start, p_end) in phases.items():
        if phase_name == "recovery":
            continue
        anchor_lo = p_start + half_max
        anchor_hi = p_end - half_max
        if anchor_hi < anchor_lo:
            continue

        # iterate anchors on a clean 2-s grid
        n_anchors = int(np.floor((anchor_hi - anchor_lo) / ANCHOR_STEP_S)) + 1
        anchors = anchor_lo + ANCHOR_STEP_S * np.arange(n_anchors)

        for t_anchor in anchors:
            t_anchor = float(t_anchor)
            if _anchor_in_boundary(t_anchor):
                continue

            feats: dict[str, float] = {}

            # HR (10-s window)
            t_lo, t_hi = t_anchor - half["hr"], t_anchor + half["hr"]
            if _window_touches_boundary(t_lo, t_hi):
                feats["hr"] = float("nan")
            else:
                rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
                feats.update(hr_window_feature(rp, pp.fs_ecg))

            # RMSSD + Poincaré (60-s window)
            t_lo, t_hi = t_anchor - half["hrv_rmssd"], t_anchor + half["hrv_rmssd"]
            if _window_touches_boundary(t_lo, t_hi):
                feats.update({k: float("nan") for k in ["hrv_rmssd", "sd2_sd1", "samp_en", "ss"]})
            else:
                rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t_lo, t_hi)
                feats.update(poincare_window_features(rp, pp.fs_ecg))

            # RR (40-s window)
            t_lo, t_hi = t_anchor - half["rr"], t_anchor + half["rr"]
            if _window_touches_boundary(t_lo, t_hi):
                feats["rr"] = float("nan")
            else:
                bp = _slice_peaks_by_time(pp.br_peaks, pp.fs_br, pp.br_t0, t_lo, t_hi)
                feats.update(rr_window_feature(bp, pp.fs_br))

            # RRV (60-s window)
            t_lo, t_hi = t_anchor - half["rrv"], t_anchor + half["rrv"]
            if _window_touches_boundary(t_lo, t_hi):
                feats["rrv"] = float("nan")
            else:
                bp = _slice_peaks_by_time(pp.br_peaks, pp.fs_br, pp.br_t0, t_lo, t_hi)
                feats.update(rrv_window_feature(bp, pp.fs_br))

            # CSI (40-s window)
            t_lo, t_hi = t_anchor - half["csi"], t_anchor + half["csi"]
            if _window_touches_boundary(t_lo, t_hi):
                feats["csi"] = float("nan")
            else:
                feats.update(_csi_for_window(pp, t_lo, t_hi))

            # Respiratory amplitude (40-s window, BR signal peak-to-trough)
            t_lo, t_hi = t_anchor - half["resp_amp"], t_anchor + half["resp_amp"]
            if _window_touches_boundary(t_lo, t_hi):
                feats["resp_amp"] = float("nan")
            else:
                i0 = max(0, int(round((t_lo - pp.br_t0) * pp.fs_br)))
                i1 = min(len(pp.br), int(round((t_hi - pp.br_t0) * pp.fs_br)))
                feats.update(resp_amp_window_feature(pp.br[i0:i1], pp.fs_br))

            # Microphone RMS energy (40-s window, Shannon-energy envelope)
            t_lo, t_hi = t_anchor - half["mic_rms"], t_anchor + half["mic_rms"]
            if _window_touches_boundary(t_lo, t_hi):
                feats["mic_rms"] = float("nan")
            else:
                i0 = max(0, int(round((t_lo - pp.mic_t0) * pp.fs_mic)))
                i1 = min(len(pp.se), int(round((t_hi - pp.mic_t0) * pp.fs_mic)))
                feats.update(mic_rms_window_feature(pp.se[i0:i1]))

            # Optional temperature ablation features over the longest window
            if include_temp:
                t_lo, t_hi = t_anchor - half_max, t_anchor + half_max
                if _window_touches_boundary(t_lo, t_hi):
                    feats.update({k: float("nan") for k in TEMP_FEATURE_NAMES})
                else:
                    feats.update(temp_features_window(rec.channels["temp"], t_lo, t_hi))

            for k, v in list(feats.items()):
                if v is None or not np.isfinite(v):
                    feats[k] = float("nan")

            keep = list(FEATURE_NAMES) + (list(TEMP_FEATURE_NAMES) if include_temp else [])
            feats = {k: feats.get(k, float("nan")) for k in keep}

            # t_start / t_end are kept anchor-equal so downstream tools
            # treating the row as a point sample don't accidentally double
            # the window length.
            windows.append(Window(
                rec_name=rec.name, subject=rec.subject, stressor=rec.stressor,
                phase=phase_name, t_start=t_anchor, t_end=t_anchor,
                features=feats,
            ))

    return windows
