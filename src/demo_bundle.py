"""Pre-compute everything the dashboard needs for one seizure of one patient.

The bundle contains:
* the *full training embedding* (the "ghost cloud" backdrop)
* the *replay window*: 30 min pre-ictal → 2 min post-onset, densely sampled
* for each replay frame: feature vector, latent z, attribution heatmap,
  risk score, drift velocity, local variance, TTS estimate, raw EEG slice
* state centroids so the dashboard can render semi-transparent reference spheres
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class DemoBundle:
    patient: str
    seizure_idx: int
    onset_s: float
    offset_s: float

    # backdrop — training embedding
    ghost_Z: np.ndarray                  # (N_ghost, latent_dim)
    ghost_state: np.ndarray              # (N_ghost,) int8

    # replay frames
    frame_t: np.ndarray                  # (T,) seconds, 0 at seizure onset
    frame_Z: np.ndarray                  # (T, latent_dim)
    frame_risk: np.ndarray               # (T,) 0-100
    frame_velocity: np.ndarray           # (T,)
    frame_variance: np.ndarray           # (T,)
    frame_criticality: np.ndarray        # (T,)
    frame_tts: np.ndarray                # (T,) minutes (NaN when indeterminate)
    frame_state: np.ndarray              # (T,) int8
    frame_ch_band: np.ndarray            # (T, n_ch, n_bands) per-frame attribution

    # raw EEG slice (for the scrolling trace panel)
    eeg: np.ndarray                      # (n_ch, n_samples) float32, µV
    eeg_sfreq: float
    eeg_channels: list[str]

    # reference geometry
    centroid_interictal: np.ndarray
    centroid_preictal: np.ndarray
    centroid_ictal: np.ndarray

    # metadata
    bands: list[str]
    channels: list[str]
    feature_names: list[str]
    window_s: float
    step_s: float


def save_bundle(b: DemoBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        patient=np.array(b.patient),
        seizure_idx=np.array(b.seizure_idx),
        onset_s=np.array(b.onset_s),
        offset_s=np.array(b.offset_s),
        ghost_Z=b.ghost_Z,
        ghost_state=b.ghost_state,
        frame_t=b.frame_t,
        frame_Z=b.frame_Z,
        frame_risk=b.frame_risk,
        frame_velocity=b.frame_velocity,
        frame_variance=b.frame_variance,
        frame_criticality=b.frame_criticality,
        frame_tts=b.frame_tts,
        frame_state=b.frame_state,
        frame_ch_band=b.frame_ch_band,
        eeg=b.eeg,
        eeg_sfreq=np.array(b.eeg_sfreq),
        eeg_channels=np.array(b.eeg_channels),
        centroid_interictal=b.centroid_interictal,
        centroid_preictal=b.centroid_preictal,
        centroid_ictal=b.centroid_ictal,
        bands=np.array(b.bands),
        channels=np.array(b.channels),
        feature_names=np.array(b.feature_names),
        window_s=np.array(b.window_s),
        step_s=np.array(b.step_s),
    )


def load_bundle(path: str | Path) -> DemoBundle:
    z = np.load(path, allow_pickle=False)
    return DemoBundle(
        patient=str(z["patient"].item()),
        seizure_idx=int(z["seizure_idx"].item()),
        onset_s=float(z["onset_s"].item()),
        offset_s=float(z["offset_s"].item()),
        ghost_Z=z["ghost_Z"], ghost_state=z["ghost_state"],
        frame_t=z["frame_t"], frame_Z=z["frame_Z"],
        frame_risk=z["frame_risk"],
        frame_velocity=z["frame_velocity"],
        frame_variance=z["frame_variance"],
        frame_criticality=z["frame_criticality"],
        frame_tts=z["frame_tts"],
        frame_state=z["frame_state"],
        frame_ch_band=z["frame_ch_band"],
        eeg=z["eeg"], eeg_sfreq=float(z["eeg_sfreq"].item()),
        eeg_channels=[str(c) for c in z["eeg_channels"]],
        centroid_interictal=z["centroid_interictal"],
        centroid_preictal=z["centroid_preictal"],
        centroid_ictal=z["centroid_ictal"],
        bands=[str(c) for c in z["bands"]],
        channels=[str(c) for c in z["channels"]],
        feature_names=[str(c) for c in z["feature_names"]],
        window_s=float(z["window_s"].item()),
        step_s=float(z["step_s"].item()),
    )


def per_frame_channel_band_attribution(
    model: torch.nn.Module,
    X_frames: np.ndarray,
    n_channels: int,
    n_bands: int,
    behavior_dims: int,
    device: torch.device,
) -> np.ndarray:
    """Compute the (channel × band) attribution block for *each* frame.

    For each frame x, we compute |∂z_b / ∂x| over behaviour dims b=0..behavior_dims-1,
    average over behaviour dims, and reshape the first n_channels*n_bands
    entries (the band-power block) into a (n_ch, n_bands) heatmap.
    """
    model.eval()
    X_t = torch.as_tensor(X_frames, dtype=torch.float32, device=device)
    n, d = X_t.shape
    bp_block = n_channels * n_bands
    out = np.zeros((n, n_channels, n_bands), dtype=np.float32)

    for i in range(n):
        x = X_t[i: i + 1].clone().requires_grad_(True)
        y = model(x).reshape(1, -1)
        # sum over behaviour dims
        accum = torch.zeros(d, device=device)
        for j in range(min(behavior_dims, y.shape[1])):
            grad = torch.autograd.grad(
                y[0, j], x, retain_graph=(j < behavior_dims - 1),
                create_graph=False,
            )[0]
            accum = accum + grad[0].abs()
        accum = accum / behavior_dims
        bp_vec = accum[:bp_block].detach().cpu().numpy()
        out[i] = bp_vec.reshape(n_channels, n_bands)
    return out
