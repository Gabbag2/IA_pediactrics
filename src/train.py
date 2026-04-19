"""Thin training entry point; orchestrates model fitting + persistence."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import PooledDataset, balance_classes
from .model import TrainConfig, fit_cebra, save


def fit(pooled: PooledDataset, cfg: TrainConfig, out_path: str | Path) -> dict:
    """Balance classes, fit plain CEBRA, persist checkpoint.

    Returns a metadata dict with the trained estimator, embedding, and indices.
    """
    idx = balance_classes(pooled, interictal_ratio=3.0, seed=cfg.seed)
    X_tr = pooled.X[idx]
    y_tr = pooled.y_state[idx]

    est = fit_cebra(X_tr, y_tr, cfg)
    save(est, out_path)

    # Embed the full pooled set for downstream analysis.
    Z = est.transform(pooled.X.astype(np.float32))
    return dict(
        estimator=est,
        embedding=Z.astype(np.float32),
        train_idx=idx,
    )
