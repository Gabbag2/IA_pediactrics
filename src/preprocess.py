"""Preprocessing: filtering, window iteration, state labelling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import mne
import numpy as np


STATE_INTERICTAL = 0
STATE_PREICTAL = 1
STATE_ICTAL = 2
STATE_EXCLUDE = -1  # post-ictal buffer, within 1 h of any seizure etc.

STATE_NAMES = {
    STATE_INTERICTAL: "interictal",
    STATE_PREICTAL: "pre-ictal",
    STATE_ICTAL: "ictal",
    STATE_EXCLUDE: "excluded",
}


def bandpass_notch(
    raw: mne.io.BaseRaw,
    l_freq: float = 0.6,
    h_freq: float = 80.0,
    notch_hz: float = 60.0,
) -> mne.io.BaseRaw:
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR", n_jobs=1)
    raw.notch_filter(freqs=[notch_hz], verbose="ERROR", n_jobs=1)
    return raw


@dataclass
class Window:
    file: str
    start_s: float
    end_s: float
    data: np.ndarray        # shape (n_ch, n_samples)
    state: int              # interictal/preictal/ictal/exclude
    t_to_onset: float       # seconds to nearest seizure onset (positive=before)


def _state_for_window(
    mid_s: float,
    start_s: float,
    end_s: float,
    seizures: list[tuple[float, float]],
    pre_ictal_s: float,
    post_ictal_s: float,
    interictal_guard_s: float,
) -> tuple[int, float]:
    """Return (state_code, t_to_onset) for a window centred at ``mid_s``."""
    # Determine ictal overlap
    for onset, offset in seizures:
        if end_s >= onset and start_s <= offset:
            return STATE_ICTAL, onset - mid_s

    # Pre-ictal windows: wholly within [onset - pre_ictal_s, onset)
    for onset, _ in seizures:
        if onset - pre_ictal_s <= start_s and end_s <= onset:
            return STATE_PREICTAL, onset - mid_s

    # Post-ictal buffer → exclude
    for _, offset in seizures:
        if offset < mid_s <= offset + post_ictal_s:
            return STATE_EXCLUDE, 0.0

    # Interictal guard: must be ≥ interictal_guard_s from any seizure boundary
    if seizures:
        nearest = min(
            min(abs(mid_s - onset), abs(mid_s - offset))
            for onset, offset in seizures
        )
        if nearest < interictal_guard_s:
            return STATE_EXCLUDE, 0.0

    # Otherwise interictal
    if seizures:
        t_to = min(onset - mid_s for onset, _ in seizures
                   if onset - mid_s > 0) if any(onset > mid_s for onset, _ in seizures) else np.inf
    else:
        t_to = np.inf
    return STATE_INTERICTAL, float(t_to) if np.isfinite(t_to) else np.nan


def iter_windows(
    raw: mne.io.BaseRaw,
    file_name: str,
    seizures: list[tuple[float, float]],
    window_s: float = 5.0,
    step_s: float = 2.5,
    pre_ictal_s: float = 300.0,
    post_ictal_s: float = 1800.0,
    interictal_guard_s: float = 3600.0,
) -> Iterator[Window]:
    """Yield labelled sliding windows from one recording.

    Skips windows labelled STATE_EXCLUDE (post-ictal buffer / guard zone).
    """
    sfreq = float(raw.info["sfreq"])
    n_samples_win = int(round(window_s * sfreq))
    step_samples = int(round(step_s * sfreq))
    total = raw.n_times
    data = raw.get_data()  # (n_ch, n_times) in volts

    for start in range(0, total - n_samples_win + 1, step_samples):
        end = start + n_samples_win
        start_s = start / sfreq
        end_s = end / sfreq
        mid_s = 0.5 * (start_s + end_s)
        state, t_to = _state_for_window(
            mid_s, start_s, end_s, seizures,
            pre_ictal_s, post_ictal_s, interictal_guard_s,
        )
        if state == STATE_EXCLUDE:
            continue
        yield Window(
            file=file_name,
            start_s=start_s,
            end_s=end_s,
            data=data[:, start:end].astype(np.float32),
            state=state,
            t_to_onset=float(t_to),
        )
