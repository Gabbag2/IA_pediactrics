"""Feature cache I/O and dataset assembly."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PatientCache:
    patient: str
    X: np.ndarray            # (n_windows, n_features) float32
    y_state: np.ndarray      # (n_windows,) int8 in {0,1,2}
    t_to_onset: np.ndarray   # (n_windows,) float32 seconds (inf if no upcoming seizure)
    file_id: np.ndarray      # (n_windows,) int32
    files: np.ndarray        # (n_files,) str — file name for each file_id
    feature_names: np.ndarray  # (n_features,) str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            patient=np.array(self.patient),
            X=self.X,
            y_state=self.y_state,
            t_to_onset=self.t_to_onset,
            file_id=self.file_id,
            files=self.files,
            feature_names=self.feature_names,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PatientCache":
        z = np.load(path, allow_pickle=False)
        return cls(
            patient=str(z["patient"].item()),
            X=z["X"],
            y_state=z["y_state"],
            t_to_onset=z["t_to_onset"],
            file_id=z["file_id"],
            files=z["files"],
            feature_names=z["feature_names"],
        )


@dataclass
class PooledDataset:
    X: np.ndarray             # (N, F) float32 (already z-scored per-patient)
    y_state: np.ndarray       # (N,) int8
    t_to_onset: np.ndarray    # (N,) float32
    patient_id: np.ndarray    # (N,) int32
    patients: list[str]       # ordered patient names; patient_id indexes this list
    feature_names: list[str]


def zscore_per_patient(cache: PatientCache) -> np.ndarray:
    """Z-score features using interictal statistics to avoid leaking ictal scale."""
    mask = cache.y_state == 0  # interictal
    if mask.sum() < 5:
        mask = np.ones_like(cache.y_state, dtype=bool)
    mu = cache.X[mask].mean(axis=0)
    sd = cache.X[mask].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (cache.X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pool(caches: list[PatientCache]) -> PooledDataset:
    Xs, ys, ts, pids = [], [], [], []
    names = None
    patients: list[str] = []
    for i, c in enumerate(caches):
        Z = zscore_per_patient(c)
        Xs.append(Z)
        ys.append(c.y_state)
        ts.append(c.t_to_onset)
        pids.append(np.full(len(c.y_state), i, dtype=np.int32))
        patients.append(c.patient)
        if names is None:
            names = list(map(str, c.feature_names))
    return PooledDataset(
        X=np.concatenate(Xs, axis=0).astype(np.float32),
        y_state=np.concatenate(ys, axis=0).astype(np.int8),
        t_to_onset=np.concatenate(ts, axis=0).astype(np.float32),
        patient_id=np.concatenate(pids, axis=0),
        patients=patients,
        feature_names=names or [],
    )


def balance_classes(
    pooled: PooledDataset,
    interictal_ratio: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Return indices that undersample interictal to ``interictal_ratio ×`` pre-ictal."""
    rng = np.random.default_rng(seed)
    y = pooled.y_state
    n_pre = int((y == 1).sum())
    target_inter = int(max(1, interictal_ratio * n_pre))
    inter_idx = np.where(y == 0)[0]
    if len(inter_idx) > target_inter:
        inter_idx = rng.choice(inter_idx, size=target_inter, replace=False)
    pre_idx = np.where(y == 1)[0]
    ict_idx = np.where(y == 2)[0]
    idx = np.concatenate([inter_idx, pre_idx, ict_idx])
    idx.sort()
    return idx
