"""Load raw biesl recordings + parse filename metadata."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CHANNEL_INDICES = {
    "mic":  (0, 1),
    "br":   (2, 3),
    "ecg":  (4, 5),
    "temp": (6, 7),
}

# A `_posiECG` suffix in the filename flags recordings whose ECG R-peaks
# deflect POSITIVE (instead of the default negative for this device). The
# regex captures it as its own group so it doesn't pollute the plank token.
FNAME_RE = re.compile(
    r"^(?P<subject>[A-Za-z]+\d*)[-_](?P<month>\d+)[-_](?P<day>\d+)[-_](?P<stressor>medi|math|pla)"
    # Trailing parameter token: plank duration ("1'26", "2") for pla, or a
    # math task spec ("8_12"). Non-greedy and allows underscores so math's
    # double-number survives; the negative lookahead keeps it from eating the
    # posiECG flag.
    r"(?:[-_](?P<plank>(?!posiECG)[^\.\(][^\.\(]*?))?"
    r"(?P<polarity>_posiECG)?"
    r"(?:\s*\((?P<rep>\d+)\))?\.(?:csv|txt)$"
)


@dataclass
class Recording:
    path: str
    subject: str            # mta, nvt, ...
    date: str               # "5-17"
    stressor: str           # medi | math | pla
    plank_seconds: float | None  # only for pla; parsed from filename
    rep: int | None         # for "(1)" duplicates
    channels: dict[str, np.ndarray]   # name -> (time, value) stack 2xN
    ecg_polarity: str = "negative"   # "negative" (default) or "positive" (posiECG)

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def duration_s(self) -> float:
        return max(ch[0, -1] for ch in self.channels.values())


def parse_plank_duration(token: str | None) -> float | None:
    """Parse plank duration token from filename.

    Examples:
        "1'26" -> 86.0  (1 min 26 s)
        "2"    -> 120.0
        None   -> None
    """
    if token is None:
        return None
    token = token.strip()
    if "'" in token:
        mins, secs = token.split("'")
        return float(mins) * 60 + float(secs)
    # bare number = minutes
    return float(token) * 60


def parse_filename(path: str) -> tuple[str, str, str, float | None, int | None, str]:
    name = os.path.basename(path)
    m = FNAME_RE.match(name)
    if not m:
        # Fallback parsing for non-standard filenames (e.g. "pre med 1.txt", "af med 1.txt")
        clean_name = os.path.splitext(name)[0]
        tokens = re.split(r'[-_\s]+', clean_name)
        subject = tokens[0] if len(tokens) > 0 else "unknown"
        stressor = "medi"  # default fallback
        for t in tokens:
            t_low = t.lower()
            if t_low in ("medi", "med"):
                stressor = "medi"
            elif t_low == "math":
                stressor = "math"
            elif t_low == "pla":
                stressor = "pla"

        rep = None
        if len(tokens) > 1:
            last_token = tokens[-1]
            if last_token.isdigit():
                rep = int(last_token)
            elif len(last_token) > 1 and last_token[:-1].isdigit():
                # E.g. "1a" -> rep = 1
                rep = int(last_token[:-1])

        date = "unknown"
        plank = None
        polarity = "negative"
        return subject, date, stressor, plank, rep, polarity

    subject = m.group("subject")
    date = f"{int(m.group('month'))}-{int(m.group('day'))}"
    stressor = m.group("stressor")
    # The trailing token is a plank DURATION only for pla; for math it is a
    # task spec ("8_12") that we don't convert to seconds.
    plank = parse_plank_duration(m.group("plank")) if stressor == "pla" else None
    rep = int(m.group("rep")) if m.group("rep") else None
    polarity = "positive" if m.group("polarity") else "negative"
    return subject, date, stressor, plank, rep, polarity


def load_recording(path: str) -> Recording:
    """Read a tab-separated recording by column INDEX (header names vary).

    Returns 4 channels as 2xN arrays of (time, value), NaNs dropped per channel.
    """
    df = pd.read_csv(path, sep="\t", header=0)
    if df.shape[1] < 8:
        raise ValueError(f"{path}: expected at least 8 columns, got {df.shape[1]}")

    channels: dict[str, np.ndarray] = {}
    for name, (ti, vi) in CHANNEL_INDICES.items():
        sub = df.iloc[:, [ti, vi]].dropna()
        t = sub.iloc[:, 0].to_numpy(dtype=np.float64)
        v = sub.iloc[:, 1].to_numpy(dtype=np.float64)
        # ensure strictly increasing time
        order = np.argsort(t, kind="mergesort")
        t = t[order]
        v = v[order]
        channels[name] = np.vstack([t, v])

    subject, date, stressor, plank, rep, polarity = parse_filename(path)
    return Recording(
        path=path,
        subject=subject,
        date=date,
        stressor=stressor,
        plank_seconds=plank,
        rep=rep,
        channels=channels,
        ecg_polarity=polarity,
    )


def list_recordings(data_dir: str) -> list[str]:
    """Return sorted CSV / TXT recording paths under `data_dir` recursively.

    Recordings dropped here as raw text per the features.pdf workflow are
    .txt; legacy / pre-cleaned CSVs are .csv. Both are accepted. Directories
    whose names start with an underscore are excluded so archived or rejected
    recordings do not re-enter the training set.
    """
    paths: list[str] = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        paths.extend(
            os.path.join(root, filename)
            for filename in files
            if filename.lower().endswith((".csv", ".txt"))
            and not filename.startswith("_")
        )
    return sorted(paths)


def phase_boundaries(rec: Recording, rest_s: float = 300.0) -> dict[str, tuple[float, float]]:
    """Return (start, end) in seconds for the three phases of a recording.

    rest:  [0, rest_s)
    stress: [rest_s, rest_s + stress_dur)
    recovery: [rest_s + stress_dur, end]

    stress_dur defaults to 300 s (medi/math) or the parsed plank duration (pla).
    Anything beyond the documented recovery window is included in recovery.
    """
    end = rec.duration_s
    if rec.stressor == "pla":
        if rec.plank_seconds is None:
            raise ValueError(f"missing plank duration for pla recording: {rec.name}")
        stress_dur = rec.plank_seconds
    else:
        stress_dur = 300.0

    rest_end = rest_s
    stress_end = rest_s + stress_dur
    if stress_end > end:
        # protocol overran; clip
        stress_end = end
    return {
        "rest":     (0.0, rest_end),
        "stress":   (rest_end, stress_end),
        "recovery": (stress_end, end),
    }


def channel_fs(ch: np.ndarray) -> float:
    """Median sampling rate of a 2xN (time, value) array."""
    t = ch[0]
    return float(1.0 / np.median(np.diff(t)))


def resample_uniform(ch: np.ndarray, fs_target: float) -> np.ndarray:
    """Resample a (time, value) channel onto a uniform grid at fs_target Hz.

    Returns 1D values; the grid starts at t[0] and is at 1/fs_target spacing.
    """
    t, v = ch[0], ch[1]
    n = int(np.floor((t[-1] - t[0]) * fs_target)) + 1
    grid = t[0] + np.arange(n) / fs_target
    return np.interp(grid, t, v)
