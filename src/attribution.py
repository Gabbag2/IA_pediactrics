"""Jacobian attribution per patient.

We compute ``J(x) = ∂f/∂x`` on pre-ictal windows, aggregate over samples, and
project the feature-space rows back onto the (channel × band) layout for
heatmap display and hemisphere scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

import cebra


@dataclass
class AttributionResult:
    jacobian_mean_abs: np.ndarray       # (latent_dim, n_features)
    behav_importance: np.ndarray        # (n_features,) — reduced over behaviour dims
    ch_band_map: np.ndarray             # (n_channels, n_bands) — channel × band heat
    channel_importance: np.ndarray      # (n_channels,) — summed across bands
    entropy_importance: np.ndarray      # (n_channels,)
    plv_importance: np.ndarray          # (n_plv_pairs, n_plv_bands)
    graph_importance: np.ndarray        # (n_graph_stats,)


def _torch_jacobian(model: torch.nn.Module, X: np.ndarray,
                    device: torch.device, batch: int = 64) -> np.ndarray:
    """Compute mean-abs Jacobian ``|∂y/∂x|`` averaged over rows of ``X``.

    Returns (output_dim, input_dim).
    """
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    n, d = X_t.shape
    probe = model(X_t[:1])
    out_dim = probe.reshape(1, -1).shape[1]
    accum = torch.zeros(out_dim, d, device=device)

    for start in range(0, n, batch):
        x = X_t[start:start + batch].clone().requires_grad_(True)
        y = model(x)
        # If the model returns (B, T, C) or (B, C), flatten extra dims.
        y = y.reshape(y.shape[0], -1)
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(
                y[:, j].sum(), x, retain_graph=(j < y.shape[1] - 1),
                create_graph=False,
            )[0]
            accum[j] += grad.abs().sum(dim=0)
    return (accum / n).detach().cpu().numpy()


def _project_to_channel_band(
    vec: np.ndarray, n_channels: int, n_bands: int
) -> np.ndarray:
    bp = vec[: n_channels * n_bands]
    return bp.reshape(n_channels, n_bands)


def attribution_for_patient(
    est: cebra.CEBRA,
    X_patient: np.ndarray,
    feature_layout,
    behavior_dims: int = 4,
    device: str | None = None,
) -> AttributionResult:
    """Compute per-patient attribution over a set of feature vectors.

    Parameters
    ----------
    est : trained CEBRA sklearn estimator
    X_patient : (N, F) feature matrix (already z-scored)
    feature_layout : FeatureLayout describing how features map back to (ch, band)
    behavior_dims : dimensions of the behaviour subspace to reduce over
    """
    if device is None:
        torch_device = next(est.model_.parameters()).device
    else:
        torch_device = torch.device(device)

    J = _torch_jacobian(est.model_, X_patient, torch_device)  # (out, F)
    behav_rows = J[:behavior_dims]
    behav_vec = behav_rows.mean(axis=0)

    sl = feature_layout.slices()
    ch_band = _project_to_channel_band(
        behav_vec[sl["bandpower"]],
        feature_layout.n_channels, len(feature_layout.bands),
    )
    entropy_imp = behav_vec[sl["entropy"]]
    plv_imp = behav_vec[sl["plv"]].reshape(len(feature_layout.plv_pairs),
                                            len(feature_layout.plv_bands))
    graph_imp = behav_vec[sl["graph"]]
    channel_importance = ch_band.sum(axis=1) + entropy_imp

    return AttributionResult(
        jacobian_mean_abs=J,
        behav_importance=behav_vec,
        ch_band_map=ch_band,
        channel_importance=channel_importance,
        entropy_importance=entropy_imp,
        plv_importance=plv_imp,
        graph_importance=graph_imp,
    )
