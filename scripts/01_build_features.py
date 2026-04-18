"""Build per-window feature caches for one or more patients.

Example:
    python scripts/01_build_features.py --patients chb01,chb03,chb05,chb08,chb10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from _bootstrap import setup

REPO = setup()

from src.dataset import PatientCache  # noqa: E402
from src.features import FeatureLayout, extract_window  # noqa: E402
from src.io_chbmit import load_recording, parse_summary  # noqa: E402
from src.preprocess import bandpass_notch, iter_windows  # noqa: E402


def build_layout(cfg: dict) -> FeatureLayout:
    return FeatureLayout(
        channels=tuple(cfg["channels_common"]),
        bands=tuple(cfg["bands"].keys()),
        plv_pairs=tuple((a, b) for a, b in cfg["homologous_pairs"]),
        plv_bands=tuple(cfg["plv_bands"]),
    )


def process_patient(patient: str, cfg: dict, max_files: int | None) -> PatientCache:
    data_root = Path(cfg["data_root"])
    patient_dir = data_root / patient
    summary = parse_summary(patient_dir)

    # group seizures by file
    seizures_by_file: dict[str, list[tuple[float, float]]] = {}
    for _, row in summary.iterrows():
        key = row["file"]
        if np.isnan(row["onset_s"]):
            seizures_by_file.setdefault(key, [])
        else:
            seizures_by_file.setdefault(key, []).append(
                (float(row["onset_s"]), float(row["offset_s"]))
            )

    layout = build_layout(cfg)
    bands_def = {k: tuple(v) for k, v in cfg["bands"].items()}

    files_processed: list[str] = []
    Xs, ys, ts, fids = [], [], [], []
    file_id = 0

    files = sorted(seizures_by_file.keys())
    if max_files is not None:
        # Prefer keeping all seizure-containing files, pad with interictal.
        seiz_files = [f for f in files if seizures_by_file[f]]
        non_seiz = [f for f in files if not seizures_by_file[f]]
        files = (seiz_files + non_seiz)[:max_files]

    for fname in files:
        fpath = patient_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fpath.name} missing")
            continue
        t0 = time.time()
        raw = load_recording(fpath, layout.channels, preload=True)
        bandpass_notch(raw,
                       l_freq=cfg["preprocessing"]["highpass_hz"],
                       h_freq=cfg["preprocessing"]["lowpass_hz"],
                       notch_hz=cfg["preprocessing"]["notch_hz"])

        for win in iter_windows(
            raw, file_name=fname,
            seizures=seizures_by_file[fname],
            window_s=cfg["window"]["length_s"],
            step_s=cfg["window"]["step_s"],
            pre_ictal_s=cfg["labels"]["pre_ictal_s"],
            post_ictal_s=cfg["labels"]["post_ictal_s"],
            interictal_guard_s=cfg["labels"]["interictal_guard_s"],
        ):
            v = extract_window(win.data, raw.info["sfreq"], layout, bands_def)
            Xs.append(v)
            ys.append(win.state)
            ts.append(win.t_to_onset)
            fids.append(file_id)
        files_processed.append(fname)
        print(f"  [{patient}] {fname} done ({time.time() - t0:.1f}s, "
              f"cumulative windows={len(Xs)})")
        file_id += 1

    X = np.stack(Xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int8)
    t_to = np.array(ts, dtype=np.float32)
    fid = np.array(fids, dtype=np.int32)
    return PatientCache(
        patient=patient,
        X=X, y_state=y, t_to_onset=t_to, file_id=fid,
        files=np.array(files_processed),
        feature_names=np.array(layout.feature_names()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    ap.add_argument("--patients", required=True,
                    help="Comma-separated list, e.g. chb01,chb03")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap files per patient (for smoke tests)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.out_dir or (REPO / cfg["cache_dir"] / "features"))
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    for patient in patients:
        print(f"== {patient} ==")
        t0 = time.time()
        cache = process_patient(patient, cfg, args.max_files)
        out = out_dir / f"{patient}.npz"
        cache.save(out)
        print(f"  [{patient}] saved → {out} shape X={cache.X.shape} "
              f"states={np.bincount(cache.y_state.astype(int), minlength=3).tolist()} "
              f"({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
