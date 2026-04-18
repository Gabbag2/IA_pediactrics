"""Clinician-facing derived metrics from the xCEBRA latent trajectory.

- ``compute_centroids``: interictal / pre-ictal / ictal means in latent space
- ``risk_score``: 0–100 "seizure risk", weighted combo of centroid proximity,
  drift velocity, and local variance
- ``criticality_index``: local variance — proxy for critical slowing down
- ``time_to_seizure``: naïve extrapolation of the drift toward the pre-ictal
  centroid (falls back to ``None`` when the drift is off-axis)

These metrics are deliberately simple and transparent — they are scaled to
reference distributions collected from the training embedding so gauges
interpolate between ``0`` (interictal-like) and ``1`` (ictal-like).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RiskModel:
    """Parameters fit once on the training embedding, then reused per-frame."""

    centroid_interictal: np.ndarray
    centroid_preictal: np.ndarray
    centroid_ictal: np.ndarray
    velocity_scale: float      # 95th percentile of interictal |Δz|
    variance_scale: float      # 95th percentile of interictal local variance
    weights: tuple[float, float, float] = (0.55, 0.25, 0.20)
    dims: int = 4              # first `dims` are behaviour-contrastive


def compute_centroids(Z: np.ndarray, y: np.ndarray, dims: int = 4) -> dict[str, np.ndarray]:
    out = {}
    for s, name in [(0, "interictal"), (1, "preictal"), (2, "ictal")]:
        mask = y == s
        if mask.sum() == 0:
            out[name] = Z[:, :dims].mean(axis=0)
        else:
            out[name] = Z[mask][:, :dims].mean(axis=0)
    return out


def fit_risk_model(Z: np.ndarray, y: np.ndarray, dims: int = 4,
                   variance_window: int = 20) -> RiskModel:
    c = compute_centroids(Z, y, dims=dims)
    Zb = Z[:, :dims]
    interictal_mask = y == 0
    # velocity scale from interictal drift
    if interictal_mask.sum() > 2:
        vel = np.linalg.norm(np.diff(Zb[interictal_mask], axis=0), axis=1)
        vel_scale = float(np.percentile(vel, 95)) + 1e-6
    else:
        vel_scale = 1.0
    # variance scale from rolling local variance on interictal
    var_series = _rolling_local_variance(Zb, variance_window)
    if interictal_mask.sum() > variance_window:
        var_scale = float(np.percentile(var_series[interictal_mask], 95)) + 1e-6
    else:
        var_scale = 1.0
    return RiskModel(
        centroid_interictal=c["interictal"],
        centroid_preictal=c["preictal"],
        centroid_ictal=c["ictal"],
        velocity_scale=vel_scale,
        variance_scale=var_scale,
        dims=dims,
    )


def _rolling_local_variance(Z: np.ndarray, window: int) -> np.ndarray:
    """Trace of the covariance matrix over a rolling window around each point."""
    n = len(Z)
    out = np.zeros(n, dtype=np.float32)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = Z[lo:hi].var(axis=0).sum()
    return out


def risk_score(z: np.ndarray, drift_vel: float, local_var: float,
               rm: RiskModel) -> float:
    """Return a 0–100 risk score for a single frame.

    The three ingredients:
    1. **Proximity**: how close the point sits to the pre-ictal centroid vs.
       the interictal centroid, in the behaviour subspace.
    2. **Drift**: step-wise velocity, scaled against the 95-pct interictal
       velocity so that a "normal" step yields ~0 and a sudden jump ~1.
    3. **Variance**: rolling local variance, similarly scaled.
    """
    zb = np.asarray(z[: rm.dims])
    d_int = float(np.linalg.norm(zb - rm.centroid_interictal))
    d_pre = float(np.linalg.norm(zb - rm.centroid_preictal))
    proximity = d_int / (d_int + d_pre + 1e-9)        # 0..1 — 1 = all-pre-ictal
    vel = float(np.clip(drift_vel / rm.velocity_scale, 0.0, 2.0)) / 2.0
    var = float(np.clip(local_var / rm.variance_scale, 0.0, 2.0)) / 2.0
    w0, w1, w2 = rm.weights
    raw = w0 * proximity + w1 * vel + w2 * var
    return float(np.clip(raw, 0.0, 1.0) * 100.0)


def criticality_index(Z_hist: np.ndarray, window: int = 20, rm: Optional[RiskModel] = None) -> float:
    """Trace of covariance over the last ``window`` frames, scaled if rm given."""
    if len(Z_hist) == 0:
        return 0.0
    tail = Z_hist[-window:]
    var = float(tail.var(axis=0).sum())
    if rm is not None and rm.variance_scale > 0:
        return float(np.clip(var / rm.variance_scale, 0.0, 5.0))
    return var


def time_to_seizure(
    z_curr: np.ndarray, z_prev: np.ndarray, rm: RiskModel,
    step_s: float,
) -> Optional[float]:
    """Estimate minutes until the point reaches the pre-ictal centroid.

    Projects the current drift vector onto the vector from the current point to
    the pre-ictal centroid and divides distance by that component. Returns
    ``None`` if the drift is zero or pointing *away* from pre-ictal.
    """
    z_curr = np.asarray(z_curr[: rm.dims])
    z_prev = np.asarray(z_prev[: rm.dims])
    drift = z_curr - z_prev
    drift_mag = float(np.linalg.norm(drift))
    if drift_mag < 1e-6:
        return None
    target = rm.centroid_preictal - z_curr
    dist = float(np.linalg.norm(target))
    if dist < 1e-6:
        return 0.0
    dot = float((drift @ target) / dist)   # component of drift toward target
    if dot <= 0:
        return None
    steps = dist / dot                     # number of step_s steps to arrive
    return float(steps * step_s / 60.0)    # → minutes


def trajectory_metrics(Z: np.ndarray, rm: RiskModel,
                       window: int = 20, step_s: float = 2.5) -> dict[str, np.ndarray]:
    """Pre-compute the full per-frame time series (risk, vel, var, TTS)."""
    Zb = Z[:, : rm.dims]
    vel = np.concatenate([[0.0], np.linalg.norm(np.diff(Zb, axis=0), axis=1)])
    var = _rolling_local_variance(Zb, window)

    risk = np.array([
        risk_score(Zb[i], vel[i], var[i], rm) for i in range(len(Zb))
    ], dtype=np.float32)

    tts = np.full(len(Zb), np.nan, dtype=np.float32)
    for i in range(1, len(Zb)):
        est = time_to_seizure(Zb[i], Zb[i - 1], rm, step_s=step_s)
        if est is not None:
            tts[i] = est

    criticality = np.array([
        np.clip(var[i] / max(rm.variance_scale, 1e-9), 0.0, 5.0)
        for i in range(len(Zb))
    ], dtype=np.float32)

    return dict(velocity=vel, variance=var, risk=risk, tts=tts,
                criticality=criticality)
