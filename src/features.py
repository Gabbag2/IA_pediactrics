"""Per-window feature extraction: band power, sample entropy, PLV, graph stats."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

import antropy as ant
import networkx as nx


@dataclass(frozen=True)
class FeatureLayout:
    """Describe the feature vector layout so attribution can be projected back."""

    channels: tuple[str, ...]
    bands: tuple[str, ...]                 # band power bands (5)
    plv_pairs: tuple[tuple[str, str], ...]
    plv_bands: tuple[str, ...]             # bands used for PLV (typically 3)
    graph_stats: tuple[str, ...] = (
        "mean_strength", "clustering", "modularity", "spectral_radius",
    )

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_bandpower(self) -> int:
        return self.n_channels * len(self.bands)

    @property
    def n_entropy(self) -> int:
        return self.n_channels

    @property
    def n_plv(self) -> int:
        return len(self.plv_pairs) * len(self.plv_bands)

    @property
    def n_graph(self) -> int:
        return len(self.graph_stats)

    @property
    def total(self) -> int:
        return self.n_bandpower + self.n_entropy + self.n_plv + self.n_graph

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for ch in self.channels:
            for b in self.bands:
                names.append(f"bp__{ch}__{b}")
        for ch in self.channels:
            names.append(f"sampent__{ch}")
        for a, b in self.plv_pairs:
            for band in self.plv_bands:
                names.append(f"plv__{a}__{b}__{band}")
        for g in self.graph_stats:
            names.append(f"graph__{g}")
        return names

    def slices(self) -> dict[str, slice]:
        off = 0
        out: dict[str, slice] = {}
        out["bandpower"] = slice(off, off + self.n_bandpower); off = out["bandpower"].stop
        out["entropy"] = slice(off, off + self.n_entropy); off = out["entropy"].stop
        out["plv"] = slice(off, off + self.n_plv); off = out["plv"].stop
        out["graph"] = slice(off, off + self.n_graph); off = out["graph"].stop
        return out


def relative_band_powers(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Return relative band powers per channel, shape (n_ch, n_bands)."""
    # Welch PSD over the window (works well for 5s * 256Hz = 1280 samples)
    nperseg = min(x.shape[-1], int(sfreq * 2))
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, axis=-1)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x compat
    total = trapz(psd, freqs, axis=-1) + 1e-12
    out = np.empty((x.shape[0], len(bands)), dtype=np.float32)
    for j, (name, (lo, hi)) in enumerate(bands.items()):
        idx = (freqs >= lo) & (freqs <= hi)
        out[:, j] = trapz(psd[..., idx], freqs[idx], axis=-1) / total
    return out


def sample_entropy_per_channel(x: np.ndarray) -> np.ndarray:
    """Sample entropy per channel (m=2, r=0.2*std). Shape (n_ch,)."""
    out = np.empty(x.shape[0], dtype=np.float32)
    for i, ch in enumerate(x):
        try:
            out[i] = ant.sample_entropy(ch, order=2)
        except Exception:
            out[i] = np.nan
    return out


def _butter_bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * sfreq
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(4, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x, axis=-1)


def plv_matrix(x: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Phase-locking value between every pair of channels in ``x``.

    Returns a symmetric (n_ch, n_ch) matrix with zeros on the diagonal.
    """
    filt = _butter_bandpass(x, sfreq, band[0], band[1])
    phase = np.angle(hilbert(filt, axis=-1))
    n_ch = x.shape[0]
    plv = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            d = phase[i] - phase[j]
            plv[i, j] = plv[j, i] = np.abs(np.exp(1j * d).mean())
    return plv


def plv_pair_features(
    x: np.ndarray,
    channels: Sequence[str],
    sfreq: float,
    pairs: Sequence[tuple[str, str]],
    bands: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PLV for homologous channel pairs across selected bands.

    Returns (flat_features, full_plv_matrices) where the flat vector stacks
    (pair × band) in row-major order.
    """
    ch_idx = {c: i for i, c in enumerate(channels)}
    out = np.empty(len(pairs) * len(bands), dtype=np.float32)
    mats: dict[str, np.ndarray] = {}
    for bi, (bname, (lo, hi)) in enumerate(bands.items()):
        m = plv_matrix(x, sfreq, (lo, hi))
        mats[bname] = m
        for pi, (a, b) in enumerate(pairs):
            out[pi * len(bands) + bi] = m[ch_idx[a], ch_idx[b]]
    return out, mats


def graph_summaries(plv_mat: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Graph-theoretic summaries on a PLV adjacency: strength, clustering,
    modularity, spectral radius. Returns shape (4,)."""
    n = plv_mat.shape[0]
    adj = plv_mat.copy()
    np.fill_diagonal(adj, 0.0)
    mean_strength = adj.sum(axis=1).mean() / max(n - 1, 1)

    # Thresholded binary graph for community detection
    binary = (adj >= threshold).astype(float)
    G = nx.from_numpy_array(binary)
    try:
        clustering = float(nx.average_clustering(G))
    except Exception:
        clustering = 0.0
    try:
        communities = nx.community.louvain_communities(G, seed=0)
        modularity = float(nx.community.modularity(G, communities))
    except Exception:
        modularity = 0.0
    try:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvalsh(adj))))
    except Exception:
        spectral_radius = 0.0
    return np.array([mean_strength, clustering, modularity, spectral_radius],
                    dtype=np.float32)


def extract_window(
    x: np.ndarray,
    sfreq: float,
    layout: FeatureLayout,
    bands_def: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Compute the full feature vector for a single (n_ch, n_samples) window."""
    bp = relative_band_powers(x, sfreq, bands_def).ravel()
    se = sample_entropy_per_channel(x)
    plv_bands = {b: bands_def[b] for b in layout.plv_bands}
    plv_feat, plv_mats = plv_pair_features(
        x, layout.channels, sfreq, layout.plv_pairs, plv_bands
    )
    # Graph summaries from the broadband (alpha) PLV matrix as a compact summary.
    # Alpha is stable and well-studied for interictal network state.
    graph_feat = graph_summaries(plv_mats[layout.plv_bands[0]])

    vec = np.concatenate([bp, se, plv_feat, graph_feat]).astype(np.float32)
    assert vec.shape[0] == layout.total, (vec.shape[0], layout.total)
    return vec
