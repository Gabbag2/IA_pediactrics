"""CHB-MIT I/O: parse summary files and load EDF recordings."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import mne
import numpy as np
import pandas as pd


_FILE_RE = re.compile(r"^File Name:\s*(\S+)", re.MULTILINE)
_START_RE = re.compile(r"^File Start Time:\s*([\d:]+)", re.MULTILINE)
_SEIZ_COUNT_RE = re.compile(r"^Number of Seizures in File:\s*(\d+)", re.MULTILINE)
_SEIZ_START_RE = re.compile(
    r"^Seizure\s*(?:\d+\s*)?Start Time:\s*(\d+)\s*seconds", re.MULTILINE)
_SEIZ_END_RE = re.compile(
    r"^Seizure\s*(?:\d+\s*)?End Time:\s*(\d+)\s*seconds", re.MULTILINE)


@dataclass
class Seizure:
    patient: str
    file: str
    onset_s: float
    offset_s: float

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


def parse_summary(patient_dir: str | Path) -> pd.DataFrame:
    """Parse chbXX-summary.txt → DataFrame with one row per (file, seizure).

    Columns: patient, file, n_seizures, onset_s, offset_s (NaN when no seizure).
    """
    patient_dir = Path(patient_dir)
    patient = patient_dir.name
    summary = patient_dir / f"{patient}-summary.txt"
    if not summary.exists():
        raise FileNotFoundError(summary)
    text = summary.read_text()

    rows: list[dict] = []
    for block in re.split(r"\n(?=File Name:)", text):
        m = _FILE_RE.search(block)
        if not m:
            continue
        fname = m.group(1).strip()
        n_seiz_m = _SEIZ_COUNT_RE.search(block)
        n_seiz = int(n_seiz_m.group(1)) if n_seiz_m else 0
        starts = [int(x) for x in _SEIZ_START_RE.findall(block)]
        ends = [int(x) for x in _SEIZ_END_RE.findall(block)]
        if n_seiz == 0 or not starts:
            rows.append(dict(patient=patient, file=fname, n_seizures=0,
                             onset_s=np.nan, offset_s=np.nan))
        else:
            for s, e in zip(starts, ends):
                rows.append(dict(patient=patient, file=fname, n_seizures=n_seiz,
                                 onset_s=float(s), offset_s=float(e)))
    return pd.DataFrame(rows)


def list_patients(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("chb"))


def load_recording(
    edf_path: str | Path,
    channels: Sequence[str],
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Load an EDF and restrict to requested bipolar channels.

    Handles cases where CHB-MIT files list a channel under variants like
    ``T8-P8-0`` / ``T8-P8-1``: we map the first occurrence of the base name.
    Raises ``ValueError`` if any requested channel is missing.
    """
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")

    present = list(raw.ch_names)
    rename: dict[str, str] = {}
    picks: list[str] = []
    for ch in channels:
        if ch in present:
            picks.append(ch)
            continue
        # Fall back to the first variant (e.g. "T8-P8-0" → "T8-P8")
        variant = next((p for p in present if p.split("-")[:2] == ch.split("-")
                        and p not in rename.values()), None)
        if variant is None:
            raise ValueError(f"Channel {ch} missing in {edf_path.name} (have: {present})")
        rename[variant] = ch
        picks.append(ch)

    if rename:
        raw.rename_channels(rename)
    raw.pick(picks)
    raw.reorder_channels(list(channels))
    return raw
