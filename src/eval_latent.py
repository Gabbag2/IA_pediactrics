"""Latent-trajectory metrics: drift velocity, local variance, centroid
distance, local dimensionality.

Each metric takes a 2-D embedding array ``Z`` of shape (n_time, latent_dim)
and returns a 1-D series aligned to the same time axis.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def drift_velocity(Z: np.ndarray, dt: float = 1.0) -> np.ndarray:
    v = np.linalg.norm(np.diff(Z, axis=0), axis=1) / max(dt, 1e-9)
    return np.concatenate([[v[0]], v]) if len(v) else v


def local_variance(Z: np.ndarray, window: int = 20) -> np.ndarray:
    out = np.zeros(len(Z), dtype=np.float32)
    for i in range(len(Z)):
        lo = max(0, i - window // 2)
        hi = min(len(Z), i + window // 2 + 1)
        out[i] = Z[lo:hi].var(axis=0).sum()
    return out


def centroid_distance(Z: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Z - centroid[None], axis=1)


def local_dimensionality(Z: np.ndarray, k: int = 20) -> np.ndarray:
    """Participation-ratio-based local dimensionality estimator.

    For each point, compute the eigenvalues of the local kNN covariance and
    return ``(Σλ)² / Σλ²`` — a differentiable dimensionality proxy.
    """
    if len(Z) <= k + 1:
        return np.full(len(Z), Z.shape[1], dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nbrs.kneighbors(Z)
    out = np.empty(len(Z), dtype=np.float32)
    for i, ids in enumerate(idx):
        loc = Z[ids[1:]] - Z[ids[1:]].mean(axis=0, keepdims=True)
        cov = loc.T @ loc / max(1, len(loc) - 1)
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 0, None)
        s = evals.sum()
        out[i] = (s * s) / (np.square(evals).sum() + 1e-12) if s > 0 else 0.0
    return out


def align_to_onset(
    Z: np.ndarray,
    t_to_onset: np.ndarray,
    pre_horizon_s: float = 300.0,
    step_s: float = 2.5,
    post_horizon_s: float = 60.0,
) -> dict:
    """Bin windows by time-to-onset and return per-bin mean/std trajectories.

    Windows that are not in the pre-ictal horizon are ignored.
    """
    bins = np.arange(-pre_horizon_s, post_horizon_s + step_s, step_s)
    mask = (-t_to_onset >= bins[0]) & (-t_to_onset <= bins[-1])  # t=-onset measures "time since window start toward onset"
    Z = Z[mask]
    times = -t_to_onset[mask]  # negative = before onset
    digitized = np.digitize(times, bins) - 1
    means = np.full((len(bins) - 1, Z.shape[1]), np.nan, dtype=np.float32)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for b in range(len(bins) - 1):
        sel = digitized == b
        if sel.any():
            means[b] = Z[sel].mean(axis=0)
            counts[b] = int(sel.sum())
    return dict(bin_edges=bins, means=means, counts=counts)
