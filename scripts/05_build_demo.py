"""Pre-compute a replay bundle for the live-demo dashboard.

Given a patient + seizure index, this script pulls:
  * the full training embedding as the "ghost cloud" backdrop
  * per-frame latent Z, risk / velocity / variance / criticality / TTS
  * per-frame (channel × band) Jacobian attribution
  * the raw EEG slice spanning [onset - pre_ictal_s, offset + post_s]

and saves a single ``DemoBundle`` .npz that the dashboard replays.

Example:
    python scripts/05_build_demo.py \\
        --patient chb01 --seizure-idx 0 \\
        --model cache/models/pooled_chb01_chb03_chb05_chb08_chb10.pt \\
        --embedding cache/models/pooled_chb01_chb03_chb05_chb08_chb10_embedding.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import setup

REPO = setup()

from src.dataset import PatientCache, zscore_per_patient  # noqa: E402
from src.demo_bundle import (  # noqa: E402
    DemoBundle, per_frame_channel_band_attribution, save_bundle,
)
from src.features import FeatureLayout  # noqa: E402
from src.io_chbmit import load_recording, parse_summary  # noqa: E402
from src.preprocess import (  # noqa: E402
    STATE_EXCLUDE, bandpass_notch, _state_for_window,
)
from src.risk import fit_risk_model, trajectory_metrics  # noqa: E402


def _build_layout(cfg: dict) -> FeatureLayout:
    return FeatureLayout(
        channels=tuple(cfg["channels_common"]),
        bands=tuple(cfg["bands"].keys()),
        plv_pairs=tuple((a, b) for a, b in cfg["homologous_pairs"]),
        plv_bands=tuple(cfg["plv_bands"]),
    )


def _reconstruct_mid_s(
    cache: PatientCache, patient_dir: Path, cfg: dict,
) -> np.ndarray:
    """Recover the absolute ``mid_s`` (within each file) for every cache frame.

    The feature cache preserves windows in the order yielded by ``iter_windows``,
    minus the ones labelled ``STATE_EXCLUDE``. Re-running the same labelling
    logic lets us re-derive ``mid_s`` without re-filtering the EDF, by counting
    the kept windows in the same order.
    """
    summary = parse_summary(patient_dir)
    seizures_by_file: dict[str, list[tuple[float, float]]] = {}
    for _, row in summary.iterrows():
        key = row["file"]
        if np.isnan(row["onset_s"]):
            seizures_by_file.setdefault(key, [])
        else:
            seizures_by_file.setdefault(key, []).append(
                (float(row["onset_s"]), float(row["offset_s"]))
            )

    win_s = cfg["window"]["length_s"]
    step_s = cfg["window"]["step_s"]
    pre_s = cfg["labels"]["pre_ictal_s"]
    post_s = cfg["labels"]["post_ictal_s"]
    guard_s = cfg["labels"]["interictal_guard_s"]

    mid = np.full(len(cache.X), np.nan, dtype=np.float64)

    import mne
    for fid, fname in enumerate(cache.files):
        mask = cache.file_id == fid
        n_expected = int(mask.sum())
        if n_expected == 0:
            continue
        raw = mne.io.read_raw_edf(patient_dir / str(fname),
                                  preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        total = raw.n_times
        n_samples_win = int(round(win_s * sfreq))
        step_samples = int(round(step_s * sfreq))

        sz = seizures_by_file.get(str(fname), [])
        kept: list[float] = []
        for start in range(0, total - n_samples_win + 1, step_samples):
            end = start + n_samples_win
            s0 = start / sfreq
            s1 = end / sfreq
            m = 0.5 * (s0 + s1)
            state, _ = _state_for_window(m, s0, s1, sz, pre_s, post_s, guard_s)
            if state == STATE_EXCLUDE:
                continue
            kept.append(m)
        if len(kept) != n_expected:
            raise RuntimeError(
                f"[mid_s] {fname}: expected {n_expected} windows, recomputed {len(kept)}"
            )
        mid[mask] = np.array(kept, dtype=np.float64)

    if np.isnan(mid).any():
        raise RuntimeError("mid_s reconstruction left NaNs")
    return mid


def _load_estimator(ckpt: Path, cfg: dict, n_features: int, ref_n: int) -> object:
    import cebra

    from src.model import _pick_device
    device = _pick_device("cuda_if_available")

    tcfg = cfg["training"]
    est = cebra.CEBRA(
        model_architecture="offset1-model",
        output_dimension=tcfg["latent_dim"],
        time_offsets=tcfg["time_offset"],
        conditional=tcfg["conditional"],
        temperature=tcfg["temperature"],
        batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"],
        max_iterations=1,
        hybrid=True,
        num_hidden_units=128,
        device=device,
    )
    rng = np.random.default_rng(0)
    dummy_X = rng.standard_normal((max(64, ref_n), n_features)).astype(np.float32)
    dummy_y = np.linspace(0, 2, dummy_X.shape[0], dtype=np.float32)
    est.fit(dummy_X, dummy_y)
    state = torch.load(ckpt, map_location="cpu")
    est.model_.load_state_dict(state["model_state_dict"])
    return est


def _slice_eeg(
    patient_dir: Path, file_name: str, cfg: dict, channels: list[str],
    t0_s: float, t1_s: float,
) -> tuple[np.ndarray, float]:
    """Load + filter the specified EDF and slice to [t0_s, t1_s] seconds."""
    raw = load_recording(patient_dir / file_name, channels, preload=True)
    bandpass_notch(
        raw,
        l_freq=cfg["preprocessing"]["highpass_hz"],
        h_freq=cfg["preprocessing"]["lowpass_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
    )
    sfreq = float(raw.info["sfreq"])
    i0 = max(0, int(round(t0_s * sfreq)))
    i1 = min(raw.n_times, int(round(t1_s * sfreq)))
    data = raw.get_data()[:, i0:i1].astype(np.float32) * 1e6  # → µV
    return data, sfreq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    ap.add_argument("--patient", required=True)
    ap.add_argument("--seizure-idx", type=int, default=0,
                    help="Index of the seizure within parse_summary rows (seizure rows only)")
    ap.add_argument("--model", required=True, help=".pt checkpoint")
    ap.add_argument("--embedding", required=True,
                    help="Training embedding .npz from 02_train_xcebra.py")
    ap.add_argument("--post-onset-s", type=float, default=120.0,
                    help="Seconds after onset to include in the replay (capped at seizure offset + this)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data_root"])
    patient_dir = data_root / args.patient
    cache_path = REPO / cfg["cache_dir"] / "features" / f"{args.patient}.npz"
    cache = PatientCache.load(cache_path)
    layout = _build_layout(cfg)

    # Identify the target seizure
    summary = parse_summary(patient_dir)
    seiz_rows = summary.dropna(subset=["onset_s"]).reset_index(drop=True)
    if args.seizure_idx >= len(seiz_rows):
        raise ValueError(
            f"seizure-idx={args.seizure_idx} out of range (patient has {len(seiz_rows)} seizures)"
        )
    row = seiz_rows.iloc[args.seizure_idx]
    fname = str(row["file"])
    onset_s = float(row["onset_s"])
    offset_s = float(row["offset_s"])
    print(f"== {args.patient} seizure {args.seizure_idx}: "
          f"{fname} onset={onset_s:.1f}s offset={offset_s:.1f}s "
          f"duration={offset_s - onset_s:.1f}s ==")

    # Find target file id in the cache
    files = [str(f) for f in cache.files]
    if fname not in files:
        raise RuntimeError(f"{fname} not in feature cache files {files}")
    target_fid = files.index(fname)

    # Reconstruct mid_s for every frame
    print("  reconstructing window mid_s …")
    mid_s_all = _reconstruct_mid_s(cache, patient_dir, cfg)

    # Replay-window mask: frames in the target file whose mid_s lies in
    # [onset - pre_ictal_s, offset] (ictal frames included end-to-end).
    pre_s = float(cfg["labels"]["pre_ictal_s"])
    lo = onset_s - pre_s
    hi = offset_s  # cache has no post-offset frames (post_ictal_s excludes them)
    mask = (cache.file_id == target_fid) & (mid_s_all >= lo) & (mid_s_all <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError("No replay frames found — pre_ictal_s or onset mismatch?")
    print(f"  replay frames: {idx.size}  (t span = "
          f"{mid_s_all[idx[0]] - onset_s:+.1f}s → {mid_s_all[idx[-1]] - onset_s:+.1f}s)")

    # Z-score per-patient (same as training) and slice replay frames
    Zx = zscore_per_patient(cache)            # (N, F) standardised features
    X_replay = Zx[idx]
    frame_t = (mid_s_all[idx] - onset_s).astype(np.float32)
    frame_state = cache.y_state[idx].astype(np.int8)

    # Load estimator, compute frame_Z
    print("  loading estimator + embedding …")
    est = _load_estimator(Path(args.model), cfg, n_features=layout.total,
                          ref_n=min(2048, cache.X.shape[0]))
    frame_Z = est.transform(X_replay.astype(np.float32)).astype(np.float32)

    # Ghost cloud & centroids from the training embedding (full pooled set)
    emb = np.load(args.embedding, allow_pickle=False)
    ghost_Z = emb["Z"].astype(np.float32)
    ghost_state = emb["y_state"].astype(np.int8)

    behav_dims = int(cfg["training"]["behavior_dims"])
    rm = fit_risk_model(ghost_Z, ghost_state, dims=behav_dims,
                        variance_window=20)
    print(f"  risk model: vel_scale={rm.velocity_scale:.4f} "
          f"var_scale={rm.variance_scale:.4f}")

    # Per-frame metrics
    step_s = float(cfg["window"]["step_s"])
    metrics = trajectory_metrics(frame_Z, rm, window=20, step_s=step_s)

    # Per-frame (ch × band) Jacobian attribution
    print("  computing per-frame attribution …")
    attr_device = next(est.model_.parameters()).device
    frame_ch_band = per_frame_channel_band_attribution(
        est.model_, X_replay, layout.n_channels, len(layout.bands),
        behavior_dims=behav_dims, device=attr_device,
    )

    # Raw EEG slice for the scrolling trace
    print("  slicing raw EEG …")
    eeg_t0 = max(0.0, onset_s - pre_s)
    eeg_t1 = offset_s + float(args.post_onset_s)
    eeg, eeg_sfreq = _slice_eeg(
        patient_dir, fname, cfg, list(layout.channels), eeg_t0, eeg_t1,
    )

    bundle = DemoBundle(
        patient=args.patient,
        seizure_idx=args.seizure_idx,
        onset_s=onset_s - eeg_t0,       # re-zero so dashboard EEG time aligns with frames
        offset_s=offset_s - eeg_t0,
        ghost_Z=ghost_Z,
        ghost_state=ghost_state,
        frame_t=frame_t,
        frame_Z=frame_Z,
        frame_risk=metrics["risk"].astype(np.float32),
        frame_velocity=metrics["velocity"].astype(np.float32),
        frame_variance=metrics["variance"].astype(np.float32),
        frame_criticality=metrics["criticality"].astype(np.float32),
        frame_tts=metrics["tts"].astype(np.float32),
        frame_state=frame_state,
        frame_ch_band=frame_ch_band.astype(np.float32),
        eeg=eeg,
        eeg_sfreq=eeg_sfreq,
        eeg_channels=list(layout.channels),
        centroid_interictal=rm.centroid_interictal.astype(np.float32),
        centroid_preictal=rm.centroid_preictal.astype(np.float32),
        centroid_ictal=rm.centroid_ictal.astype(np.float32),
        bands=list(layout.bands),
        channels=list(layout.channels),
        feature_names=list(layout.feature_names()),
        window_s=float(cfg["window"]["length_s"]),
        step_s=step_s,
    )

    out_dir = REPO / cfg["cache_dir"] / "demo"
    out_path = Path(args.out) if args.out else out_dir / f"{args.patient}_s{args.seizure_idx}.npz"
    save_bundle(bundle, out_path)
    print(f"  bundle → {out_path}")
    print(f"    frames: {len(frame_t)}   EEG: {eeg.shape}  sfreq={eeg_sfreq}")


if __name__ == "__main__":
    main()
