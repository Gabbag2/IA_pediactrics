"""Figure helpers — latent scatter, attribution heatmaps, trajectory metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .preprocess import STATE_NAMES


STATE_COLORS = {
    0: "#6baed6",  # interictal
    1: "#fd8d3c",  # pre-ictal
    2: "#e31a1c",  # ictal
}


def plot_latent_3d(
    Z: np.ndarray,
    y_state: np.ndarray,
    patient_id: np.ndarray,
    patients: Sequence[str],
    path: str | Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d proj)

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    for state, color in STATE_COLORS.items():
        mask = y_state == state
        ax1.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=2, alpha=0.4,
                    c=color, label=STATE_NAMES[state])
    ax1.set_title("Latent (dims 0–2) by state")
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(122, projection="3d")
    cmap = plt.colormaps.get_cmap("tab20")
    for i, name in enumerate(patients):
        mask = patient_id == i
        ax2.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=2, alpha=0.4,
                    c=[cmap(i % 20)], label=name)
    ax2.set_title("Latent (dims 0–2) by patient")
    ax2.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attribution_heatmap(
    ch_band: np.ndarray,
    channels: Sequence[str],
    bands: Sequence[str],
    title: str,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.imshow(ch_band, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|∂z_behav / ∂x|")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trajectory_metrics(
    bins: np.ndarray,
    metrics: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    keys = list(metrics.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 2.2 * len(keys)),
                             sharex=True)
    if len(keys) == 1:
        axes = [axes]
    centers = 0.5 * (bins[:-1] + bins[1:])
    for ax, k in zip(axes, keys):
        ax.plot(centers, metrics[k], color="#e31a1c")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_ylabel(k)
    axes[-1].set_xlabel("time to onset (s, negative = before)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


