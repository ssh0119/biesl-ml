"""Signal preprocessing per features.pdf.

ECG  : wavelet denoise (sym4, soft-threshold) keeping the 5-45 Hz band,
       then Pan-Tompkins R-peak detection (peaks are negative in the raw
       local signal; pre-flipped before detection).
BR   : detrend -> 0.5 s moving-average -> 4th-order Cheby II low-pass
       (stopband edge 1 Hz) -> 2nd-order Butterworth high-pass at 0.12 Hz.
       Peak detection uses the PDF's slope-based algorithm with an adaptive
       1/3-of-recent-amplitude threshold.
Mic  : bandpass 20-200 Hz (PCG band), then Shannon-energy envelope
       SE = -x^2 log(x^2). S1/S2 peaks classified per cardiac cycle using
       the ECG R-peak timing.
NN   : intervals shorter than 300 ms or longer than 1500 ms are rejected;
       intervals that deviate more than 20 % from the median of the
       surrounding ~10 beats are also rejected; rejects are cubic-spline
       interpolated.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy import signal
from scipy.interpolate import CubicSpline


# ---- ECG --------------------------------------------------------------------

ECG_WAVELET = "sym4"
ECG_BAND_LO_HZ = 5.0
ECG_BAND_HI_HZ = 45.0


def _detail_levels_in_band(fs: float, max_level: int,
                            lo: float, hi: float) -> set[int]:
    """Detail levels (1..max_level) whose dyadic passband intersects [lo, hi].
    Level L covers approximately [fs / 2^(L+1), fs / 2^L]."""
    keep: set[int] = set()
    for L in range(1, max_level + 1):
        f_hi = fs / (2 ** L)
        f_lo = fs / (2 ** (L + 1))
        if f_hi >= lo and f_lo <= hi:
            keep.add(L)
    return keep


def filter_ecg(x: np.ndarray, fs: float, mains: float | None = None) -> np.ndarray:
    """Wavelet denoise + 5-45 Hz band-keep. Replaces the previous
    Butterworth 1-150 Hz + mains-notch chain.

    Steps:
      1. Multi-resolution DWT with `ECG_WAVELET` (sym4) deep enough that
         the deepest approximation band falls fully *below* ECG_BAND_LO_HZ
         (so it contains only DC / baseline wander).
      2. Donoho universal soft-threshold on the detail coefficients of
         every level whose passband intersects [5, 45] Hz (denoise).
      3. Zero the detail coefficients of every other level **and** the
         deepest approximation — replaces the old HP 1 Hz, LP 150 Hz,
         and mains notch all at once (60 Hz lives in level D2 which is
         fully zeroed).
      4. Inverse DWT.

    `mains` is accepted for backwards-compatibility and ignored (mains
    rejection comes free from band-keep).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()

    safe_depth = pywt.dwt_max_level(len(x), ECG_WAVELET)
    # Choose depth so that A_max covers [0, fs / 2^(max_level+1)) which
    # sits strictly below ECG_BAND_LO_HZ. The +1 inside ceil enforces a
    # safe margin against rounding at the boundary.
    needed_depth = int(np.ceil(np.log2(fs / ECG_BAND_LO_HZ)))
    max_level = max(1, min(safe_depth, needed_depth))

    coeffs = pywt.wavedec(x, ECG_WAVELET, level=max_level, mode="symmetric")
    cA, *cDs = coeffs                          # cDs in order D_max ... D_1
    keep = _detail_levels_in_band(fs, max_level, ECG_BAND_LO_HZ, ECG_BAND_HI_HZ)

    # Donoho universal threshold derived from the finest-scale detail:
    #   λ = σ √(2 ln N),   σ = MAD(D_1) / 0.6745
    d1 = cDs[-1]
    sigma = float(np.median(np.abs(d1 - np.median(d1)))) / 0.6745
    lam = sigma * np.sqrt(2.0 * np.log(max(len(x), 2)))

    new_cDs: list[np.ndarray] = []
    for i, cD in enumerate(cDs):
        level = max_level - i                  # D_max, D_{max-1}, …, D_1
        if level in keep:
            new_cDs.append(pywt.threshold(cD, lam, mode="soft"))
        else:
            new_cDs.append(np.zeros_like(cD))

    new_cA = np.zeros_like(cA)                 # always drop the approximation

    y = pywt.waverec([new_cA] + new_cDs, ECG_WAVELET, mode="symmetric")
    return y[: len(x)]


def filter_ecg_for_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins 5-15 Hz prefilter (kept for callers that want it explicit)."""
    sos = signal.butter(4, [5.0, 15.0], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def detect_ecg_rpeaks(ecg: np.ndarray, fs: float,
                      polarity: str = "negative",
                      method: str = "pantompkins1985") -> np.ndarray:
    """R-peak detection — ORIGINAL negative-polarity algorithm.

    Assumes the device's R-peaks deflect NEGATIVE: the signal is flipped
    before detection so R-peaks appear as upward maxima for the detector.
    Detected indices are then snapped to the local maximum of the flipped
    signal within ±60 ms (Pan-Tompkins reports peak indices at the end of
    its 150 ms integration window, which lags the true apex).

    `polarity` is accepted for API compatibility but **NOT honoured here**:
    this function applies the negative-polarity pipeline to every signal so
    you can directly compare against `posiECG` recordings (which will look
    visibly broken — that is the diagnostic).

    `method` is forwarded to `neurokit2.ecg_peaks(method=...)`. The default
    "pantompkins1985" matches the historical chain; "neurokit" is the
    library's own default (more robust on noisy short-window data).
    """
    import neurokit2 as nk
    del polarity  # intentionally ignored — always run negative-polarity pipeline

    flipped = -ecg
    cleaned = nk.ecg_clean(flipped, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(peaks) > 3:
        # iterative=False: kubios's iterative mode recomputes outlier thresholds
        # over the *whole* recording, which gets poisoned by plank-phase motion
        # artefact and then wrongly discards legitimate post-plank R-peaks.
        _, info2 = nk.signal_fixpeaks(peaks, sampling_rate=fs, method="kubios",
                                      iterative=False)
        if isinstance(info2, dict):
            peaks = np.asarray(info2.get("clean", peaks), dtype=int)
        else:
            peaks = np.asarray(info2, dtype=int)

    # Snap to the local maximum of the flipped signal (~true R-wave apex).
    half = int(round(0.060 * fs))
    snapped: list[int] = []
    n = len(flipped)
    for p in peaks:
        lo = max(0, p - half)
        hi = min(n, p + half + 1)
        snapped.append(lo + int(np.argmax(flipped[lo:hi])))
    return np.asarray(snapped, dtype=int)


def detect_ecg_rpeaks_per_phase(
    ecg: np.ndarray, fs: float, t0: float,
    phases: dict[str, tuple[float, float]],
    *,
    method: str = "pantompkins1985",
) -> np.ndarray:
    """Phase-aware R-peak detection.

    Slice the ECG by absolute-time `phases` dict (`{name: (t_start, t_end)}`),
    run `detect_ecg_rpeaks` independently on each slice, offset the indices
    back into the whole-signal frame, and concatenate. This isolates each
    phase's detector state (`ecg_clean` adaptive baseline + Pan-Tompkins
    adaptive threshold + kubios outlier stats) so high-noise phases (e.g.
    plank) cannot poison detection in adjacent phases (e.g. recovery).

    Returns a sorted, de-duplicated array of R-peak indices into `ecg`.
    """
    n = len(ecg)
    out: list[int] = []
    seen: set[int] = set()
    for _, (s, e) in phases.items():
        i_lo = max(0, int(round((s - t0) * fs)))
        i_hi = min(n, int(round((e - t0) * fs)))
        if i_hi - i_lo < int(round(2.0 * fs)):  # < 2 s — skip
            continue
        sub = ecg[i_lo:i_hi]
        sub_peaks = detect_ecg_rpeaks(sub, fs, method=method)
        for p in sub_peaks:
            idx = int(p + i_lo)
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                out.append(idx)
    out.sort()
    return np.asarray(out, dtype=int)


def clean_nn_intervals(
    rpeaks_idx: np.ndarray, fs: float,
    *, lo_ms: float = 300.0, hi_ms: float = 1500.0,
    median_window: int = 10, median_tol: float = 0.20,
) -> np.ndarray:
    """Return the cleaned NN series in milliseconds.

    Per features.pdf step 4: reject NN < `lo_ms` or > `hi_ms`; reject any NN
    that differs from the median of the surrounding `median_window` beats by
    more than `median_tol` (20 %). Rejected NNs are cubic-spline interpolated.
    """
    if len(rpeaks_idx) < 3:
        return np.array([], dtype=float)

    nn_ms = np.diff(rpeaks_idx) / fs * 1000.0
    n = len(nn_ms)
    valid = (nn_ms >= lo_ms) & (nn_ms <= hi_ms)

    # Median-deviation rejection (sliding window of `median_window` NN, excluding self).
    half = median_window // 2
    for i in range(n):
        if not valid[i]:
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        nearby = np.concatenate([nn_ms[lo:i], nn_ms[i + 1:hi]])
        nearby = nearby[(nearby >= lo_ms) & (nearby <= hi_ms)]
        if len(nearby) == 0:
            continue
        med = float(np.median(nearby))
        if med > 0 and abs(nn_ms[i] - med) / med > median_tol:
            valid[i] = False

    if valid.sum() < 4:
        return nn_ms[valid].astype(float)

    if valid.all():
        return nn_ms.astype(float)

    # Cubic-spline interpolation across the rejected positions.
    idx = np.arange(n, dtype=float)
    try:
        cs = CubicSpline(idx[valid], nn_ms[valid], extrapolate=True)
        out = nn_ms.astype(float).copy()
        out[~valid] = cs(idx[~valid])
        # clip the interpolated values back into the physiologic range
        out = np.clip(out, lo_ms, hi_ms)
        return out
    except Exception:
        # fall back to keeping only the valid intervals
        return nn_ms[valid].astype(float)


# ---- BR (respiration) ------------------------------------------------------

# Default median-filter window sizes (seconds). Both are adaptable — pass
# different values to filter_br() to retune for slower/faster breathing.
BR_BASELINE_WINDOW_S = 30.0    # long median → slow-drift baseline (subtracted)
BR_SMOOTH_WINDOW_S = 0.5       # short median → despike / smooth


def filter_br(
    x: np.ndarray, fs: float,
    *,
    smooth_window_s: float = BR_SMOOTH_WINDOW_S,
    baseline_window_s: float = BR_BASELINE_WINDOW_S,
) -> np.ndarray:
    """BR analysis via median filtering, with an adaptable window.

    Replaces the previous moving-average + Chebyshev-II + Butterworth chain.
    Two median-filter passes:

      1. **baseline removal** — `baseline = median_filter(x, baseline_window_s)`
         captures slow respiratory-baseline wander; subtracting it removes
         drift *without ringing*. A linear high-pass filter rings around the
         step transients produced by body motion at the phase boundaries; a
         median filter does not, which is the whole point of switching.
      2. **despike / smooth** — a short `median_filter(smooth_window_s)`
         removes residual motion spikes while preserving the breath waveform
         shape (median filters are edge-preserving, unlike a moving average).

    Both window lengths are parameters; the defaults (8 s baseline, 0.5 s
    smooth) suit ~6–20 breaths/min at the 100 Hz working rate. Widen
    `baseline_window_s` for very slow breathing; widen `smooth_window_s` for
    noisier mic-coupled respiration belts.
    """
    from scipy.ndimage import median_filter

    x = x.astype(np.float64, copy=False)
    base_w = max(1, int(round(baseline_window_s * fs)))
    baseline = median_filter(x, size=base_w, mode="nearest")
    detrended = x - baseline
    smooth_w = max(1, int(round(smooth_window_s * fs)))
    return median_filter(detrended, size=smooth_w, mode="nearest")


def detect_br_peaks(br: np.ndarray, fs: float,
                    clip_mad: float = 5.0,
                    prom_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """BR peak detector following features.pdf steps 4-5.

    Implementation note: the PDF specifies a 250 ms minimum rising/falling
    slope, but applied verbatim it over-rejects on real data because of
    micro-variations in the filtered breathing signal. We use
    `scipy.signal.find_peaks` with a 1.5 s minimum inter-peak distance as the
    slope surrogate (a breath cycle ≥ 1.5 s implies rising + falling segments
    each ≥ 250 ms in any reasonable filtered signal), then apply the PDF's
    adaptive 1/3 × mean-of-last-eight-amplitudes threshold on AC = peak
    height above the V1-V2 baseline midline.

    Robustness fix: before peak finding we clip the signal to ±clip_mad·MAD
    (median absolute deviation, scaled by 1.4826 to match σ for Gaussian
    data). Without this, a single motion-artefact transient at a phase
    boundary becomes the first accepted peak, sets the adaptive threshold to
    ~10⁵, and rejects every subsequent real breath. Setting clip_mad=0
    disables clipping.

    Returns (peak_indices, peak_amplitudes).
    """
    n = len(br)
    if n < int(fs * 2):
        return np.array([], dtype=int), np.array([], dtype=float)

    if clip_mad > 0:
        med = float(np.median(br))
        mad = float(np.median(np.abs(br - med)))
        sigma_robust = 1.4826 * mad
        if sigma_robust > 1e-12:
            limit = clip_mad * sigma_robust
            br = np.clip(br, med - limit, med + limit)

    # Global prominence floor: a fraction of the *active* breathing amplitude
    # (90th percentile of |signal|, robust to the flat rest/recovery stretches
    # that dominate the median). This suppresses the dense noise peaks the
    # median-filtered signal still carries where breathing is shallow — those
    # peaks would otherwise inflate the breath rate to ~25/min. prom_frac=0
    # disables the floor.
    min_dist = max(int(round(fs * 1.5)), 1)
    if prom_frac > 0:
        active = float(np.percentile(np.abs(br - np.median(br)), 90))
        prominence = prom_frac * active if active > 1e-9 else None
    else:
        prominence = None
    cand, _ = signal.find_peaks(br, distance=min_dist, prominence=prominence)
    if len(cand) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    accepted: list[int] = []
    amps: list[float] = []
    for i, p in enumerate(cand):
        # V1 = local minimum between previous candidate and p
        left = cand[i - 1] if i > 0 else max(0, p - int(fs * 4))
        right = cand[i + 1] if i < len(cand) - 1 else min(n, p + int(fs * 4))
        v1_val = float(np.min(br[left:p])) if p > left else float(br[p])
        v2_val = float(np.min(br[p + 1:right])) if (p + 1) < right else float(br[p])

        ac = float(br[p]) - 0.5 * (v1_val + v2_val)
        if ac <= 0:
            continue

        threshold = (float(np.mean(amps[-8:])) / 3.0) if len(amps) > 0 else 0.0
        if ac < threshold:
            continue

        accepted.append(int(p))
        amps.append(ac)

    return np.asarray(accepted, dtype=int), np.asarray(amps, dtype=float)


def detect_br_peaks_sliding(
    br: np.ndarray, fs: float,
    *,
    window_s: float = 60.0,
    step_s: float = 30.0,
    prom_frac: float = 0.25,
    merge_dist_s: float = 0.75,
) -> np.ndarray:
    """Sliding-window BR peak detector with a LOCAL prominence floor.

    For each `window_s`-long window (stepped by `step_s`) we recompute the
    prominence floor as `prom_frac × p90(|x − median(x)|)` within that window.
    This is the key difference vs `detect_br_peaks`: the floor adapts to the
    LOCAL breathing amplitude, so the detector neither under-fires in
    low-SNR rest stretches nor over-fires when the active-phase amplitude
    dominates a global percentile.

    Peaks found in overlapping windows are de-duplicated by merging any pair
    within `merge_dist_s` seconds (keeping the larger-amplitude one).
    """
    n = len(br)
    if n < int(fs * 2):
        return np.array([], dtype=int)
    win_n = max(int(round(window_s * fs)), 1)
    step_n = max(int(round(step_s * fs)), 1)
    dist = max(int(round(fs * 1.5)), 1)

    found: list[int] = []
    starts = list(range(0, max(n - win_n + 1, 1), step_n))
    if starts[-1] + win_n < n:                       # tail window
        starts.append(max(0, n - win_n))
    for s in starts:
        seg = br[s:s + win_n]
        if len(seg) < int(fs * 2):
            continue
        active = float(np.percentile(np.abs(seg - np.median(seg)), 90))
        prom = prom_frac * active if active > 1e-9 else None
        loc, _ = signal.find_peaks(seg, distance=dist, prominence=prom)
        found.extend((loc + s).tolist())
    if not found:
        return np.array([], dtype=int)

    # de-duplicate: any two peaks within `merge_dist_s` collapse into the larger.
    merge = max(int(round(merge_dist_s * fs)), 1)
    found.sort()
    out: list[int] = [found[0]]
    for p in found[1:]:
        if p - out[-1] > merge:
            out.append(p)
        elif br[p] > br[out[-1]]:
            out[-1] = p
    return np.asarray(out, dtype=int)


def detect_br_peaks_neurokit(
    br: np.ndarray, fs: float, *, method: str = "biosppy",
) -> np.ndarray:
    """neurokit2's respiration peak detector.

    `method` is forwarded to `neurokit2.rsp_peaks`. "biosppy" is the default
    BioSPPy implementation (zero-crossing on the derivative + amplitude
    threshold) and is robust across a wide range of breathing rates.
    "khodadad2018" is specifically optimized for respiration belts.
    """
    import neurokit2 as nk
    if len(br) < int(fs * 2):
        return np.array([], dtype=int)
    try:
        _, info = nk.rsp_peaks(br, sampling_rate=int(round(fs)), method=method)
    except Exception:
        return np.array([], dtype=int)
    peaks = np.asarray(info.get("RSP_Peaks", []), dtype=int)
    return peaks


# ---- Microphone / PCG ------------------------------------------------------

def filter_mic_pcg(x: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 20-200 Hz for heart sound extraction (features.pdf step 1 of CSI)."""
    nyq = 0.5 * fs
    high = min(200.0, nyq * 0.95)
    sos = signal.butter(4, [20.0, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def shannon_energy(
    x: np.ndarray, smooth_ms: float = 50.0, fs: float = 2000.0,
    *,
    clip_mad: float = 15.0,
) -> np.ndarray:
    """Shannon-energy envelope: SE = -x^2 log(x^2), with smoothing.

    PDF wrote "S = C log(C)"; the standard PCG envelope (Liang et al. 1997)
    that yields the clear S1/S2 peaks the figure shows is the Shannon energy
    of the normalized signal, -x^2 log(x^2). We follow the standard.

    `clip_mad`: if > 0, clips the signal to ±clip_mad * robust_sigma before
    normalization. This prevents motion artifacts or noise spikes from flattening
    the envelope of normal heartbeats.
    """
    x = x.astype(np.float64, copy=False)
    if clip_mad > 0:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sigma_robust = 1.4826 * mad
        if sigma_robust > 1e-12:
            limit = clip_mad * sigma_robust
            x = np.clip(x, med - limit, med + limit)

    m = float(np.max(np.abs(x)))
    if m < 1e-12:
        return np.zeros_like(x)
    xn = x / m
    x2 = xn * xn
    x2 = np.where(x2 < 1e-12, 1e-12, x2)
    se = -x2 * np.log(x2)
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    if win > 1:
        se = np.convolve(se, np.ones(win) / win, mode="same")
    return se


def detect_pcg_peaks(se: np.ndarray, fs: float) -> np.ndarray:
    """All cardiac-sound peaks in the Shannon-energy envelope."""
    # heart sounds are ~10/sec at most -> min distance 80 ms; prominence
    # adapted to the signal scale.
    std = float(np.std(se))
    if std < 1e-12:
        return np.array([], dtype=int)
    min_dist = max(int(round(0.08 * fs)), 1)
    peaks, _ = signal.find_peaks(se, distance=min_dist, prominence=0.2 * std)
    return peaks


def classify_s1_s2_indices(
    se: np.ndarray, se_fs: float, pcg_peaks: np.ndarray,
    rpeaks_se_idx: np.ndarray,
    *,
    mic_f: np.ndarray | None = None,
    min_amp: float = 0.0,
    envelope_frac: float = 0.20,
    search_win_s: float = 0.80,
    min_dist_s: float = 0.15,
    s1_window_s: tuple[float, float] = (0.0, 0.20),
    s2_window_s: tuple[float, float] = (0.20, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    """For each R-peak, search in `search_win_s` (0.8s) window after the R-peak.

    Rules:
    1. Amplitude threshold: 20% of outer envelope (or min_amp if explicitly given).
    2. Minimum S1-S2 distance: S1 and S2 must be at least `min_dist_s` (0.15s) apart.

    Returns (s1_indices, s2_indices) in chronological order.
    """
    del s1_window_s, s2_window_s  # API compatibility
    if len(pcg_peaks) == 0 or len(rpeaks_se_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    win_idx = int(round(search_win_s * se_fs))
    half_win = int(round(0.05 * se_fs))
    min_dist_samples = int(round(min_dist_s * se_fs))

    if mic_f is not None:
        if min_amp > 0:
            threshold = min_amp
        elif envelope_frac > 0:
            from scipy.signal import hilbert
            outer_env = np.abs(hilbert(mic_f))
            threshold = envelope_frac * float(np.percentile(outer_env, 98))
        else:
            threshold = 0.0
    else:
        threshold = 0.0

    s1_idxs: list[int] = []
    s2_idxs: list[int] = []

    for r in rpeaks_se_idx:
        in_range = pcg_peaks[(pcg_peaks >= r) & (pcg_peaks <= r + win_idx)]

        if mic_f is not None and threshold > 0:
            valid_peaks = []
            for p in in_range:
                lo = max(0, p - half_win)
                hi = min(len(mic_f), p + half_win + 1)
                amp = float(np.max(np.abs(mic_f[lo:hi]))) if len(mic_f[lo:hi]) > 0 else 0.0
                if amp >= threshold:
                    valid_peaks.append(p)
            peaks_to_use = valid_peaks
        else:
            peaks_to_use = in_range.tolist()

        if len(peaks_to_use) >= 2:
            p1 = int(peaks_to_use[0])
            for p2_cand in peaks_to_use[1:]:
                if (p2_cand - p1) >= min_dist_samples:
                    s1_idxs.append(p1)
                    s2_idxs.append(int(p2_cand))
                    break

    return np.asarray(s1_idxs, dtype=int), np.asarray(s2_idxs, dtype=int)


def classify_s1_s2(
    se: np.ndarray, se_fs: float, pcg_peaks: np.ndarray,
    rpeaks_se_idx: np.ndarray,
    *,
    mic_f: np.ndarray | None = None,
    min_amp: float = 0.0,
    envelope_frac: float = 0.20,
    search_win_s: float = 0.80,
    min_dist_s: float = 0.15,
    s1_window_s: tuple[float, float] = (0.0, 0.20),
    s2_window_s: tuple[float, float] = (0.20, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    """For each R-peak, search 0.8s window after R peak. First peak is S1, second peak (>= 0.15s) is S2.

    Returns (s1_amplitudes, s2_amplitudes) in Shannon-energy units.
    """
    s1_idxs, s2_idxs = classify_s1_s2_indices(
        se, se_fs, pcg_peaks, rpeaks_se_idx,
        mic_f=mic_f, min_amp=min_amp, envelope_frac=envelope_frac,
        search_win_s=search_win_s, min_dist_s=min_dist_s,
        s1_window_s=s1_window_s, s2_window_s=s2_window_s,
    )
    if len(s1_idxs) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return se[s1_idxs], se[s2_idxs]


def calculate_cardiac_snr(
    signal_data: np.ndarray,
    fs: float,
    s1_idxs: np.ndarray,
    s2_idxs: np.ndarray,
    *,
    s1_win_s: float = 0.10,
    s2_win_s: float = 0.10,
    method: str = "rms",
) -> float:
    """Calculate Signal-to-Noise Ratio (SNR in dB) of cardiac sound (CPS/PCG).

    Parameters
    ----------
    signal_data : np.ndarray
        Filtered cardiac sound (CPS) signal array.
    fs : float
        Sampling rate of the signal (Hz).
    s1_idxs : np.ndarray
        Sample indices of detected S1 peaks.
    s2_idxs : np.ndarray
        Sample indices of detected S2 peaks.
    s1_win_s : float
        Window length (seconds) centered on each S1 peak (default 0.10s -> ±50ms).
    s2_win_s : float
        Window length (seconds) centered on each S2 peak (default 0.10s -> ±50ms).
    method : str
        - "rms" (default): Standard RMS power SNR = 10 log10( mean(x_sig^2) / mean(x_noise^2) )
        - "voltage_squared_peak": Power = (mean(V_peaks))^2 / (mean(|V_noise|))^2
        - "voltage_squared_window": Power = (mean(|V_sig_window|))^2 / (mean(|V_noise|))^2

    Returns
    -------
    float
        SNR in dB. Returns NaN if signal or noise regions are empty or zero-power.
    """
    n = len(signal_data)
    if n == 0 or (len(s1_idxs) == 0 and len(s2_idxs) == 0):
        return float("nan")

    signal_mask = np.zeros(n, dtype=bool)

    half_s1 = int(round((s1_win_s / 2.0) * fs))
    half_s2 = int(round((s2_win_s / 2.0) * fs))

    s1_amps = []
    s2_amps = []

    for idx in s1_idxs:
        lo = max(0, idx - half_s1)
        hi = min(n, idx + half_s1 + 1)
        signal_mask[lo:hi] = True
        if len(signal_data[lo:hi]) > 0:
            s1_amps.append(np.max(np.abs(signal_data[lo:hi])))

    for idx in s2_idxs:
        lo = max(0, idx - half_s2)
        hi = min(n, idx + half_s2 + 1)
        signal_mask[lo:hi] = True
        if len(signal_data[lo:hi]) > 0:
            s2_amps.append(np.max(np.abs(signal_data[lo:hi])))

    n_sig = int(signal_mask.sum())
    n_noise = n - n_sig

    if n_sig == 0 or n_noise == 0:
        return float("nan")

    sig_samples = signal_data[signal_mask]
    noise_samples = signal_data[~signal_mask]

    v_noise_avg = float(np.mean(np.abs(noise_samples)))
    if v_noise_avg <= 0:
        return float("nan")

    if method in ("voltage_squared_peak", "voltage_peak"):
        all_peaks = s1_amps + s2_amps
        if not all_peaks:
            return float("nan")
        v_sig_avg = float(np.mean(all_peaks))
        p_sig = v_sig_avg ** 2
        p_noise = v_noise_avg ** 2
    elif method in ("voltage_squared_window", "voltage_window"):
        v_sig_avg = float(np.mean(np.abs(sig_samples)))
        p_sig = v_sig_avg ** 2
        p_noise = v_noise_avg ** 2
    else:  # "rms"
        p_sig = float(np.mean(sig_samples ** 2))
        p_noise = float(np.mean(noise_samples ** 2))

    if p_sig <= 0 or p_noise <= 0:
        return float("nan")

    snr_db = 10.0 * np.log10(p_sig / p_noise)
    return float(snr_db)


# ---- back-compat shims (kept so old imports don't break callers we missed) -

def filter_cps_cardiac(x: np.ndarray, fs: float) -> np.ndarray:  # pragma: no cover
    """Deprecated — features.pdf replaces the CPS cardiac-band branch with
    Shannon-energy CSI. Returns a 0.8-3 Hz bandpass for any straggling callers."""
    sos = signal.butter(4, [0.8, 3.0], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def filter_cps_resp(x: np.ndarray, fs: float) -> np.ndarray:  # pragma: no cover
    """Deprecated — see filter_cps_cardiac. Returns a 0.1-0.5 Hz bandpass."""
    sos = signal.butter(4, [0.1, 0.5], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


# ---- diagnostics -----------------------------------------------------------

def sanity_report(name: str, fs: float, x: np.ndarray, *, flat_window_s: float = 2.0) -> dict:
    n = len(x)
    n_nan = int(np.isnan(x).sum())
    dur = n / fs
    win = max(5, int(flat_window_s * fs))
    if n >= win:
        s = np.convolve(np.abs(np.diff(x, prepend=x[0])), np.ones(win) / win, mode="same")
        flat_frac = float((s < 1e-6).mean())
    else:
        flat_frac = 0.0
    rng = (x.max() - x.min()) if n else 0.0
    sat_frac = 0.0
    if rng > 0:
        lo_thr = x.min() + 0.001 * rng
        hi_thr = x.max() - 0.001 * rng
        sat_frac = float(((x <= lo_thr) | (x >= hi_thr)).mean())
    return dict(name=name, n=n, fs=fs, duration_s=dur,
                n_nan=n_nan, flat_frac=flat_frac, sat_frac=sat_frac)
