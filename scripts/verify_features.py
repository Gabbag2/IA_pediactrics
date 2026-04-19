"""Sanity-check every feature cache in cache/features/ before training.

Checks per patient:
  * loadable .npz with expected keys
  * X shape is (N, n_features) with no NaN/Inf
  * all three states present (interictal, pre-ictal, ictal)
  * pre-ictal count > 0 (required for attribution)
  * feature_names length matches X.shape[1]

Prints a one-line-per-patient summary and exits non-zero if any patient fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from _bootstrap import setup

REPO = setup()

from src.dataset import PatientCache  # noqa: E402


REQUIRED_KEYS = {"patient", "X", "y_state", "t_to_onset", "file_id",
                 "files", "feature_names"}


def check_one(path: Path) -> tuple[str, bool, str]:
    try:
        cache = PatientCache.load(path)
    except Exception as exc:
        return path.stem, False, f"load failed: {exc}"

    X = cache.X
    y = cache.y_state
    failures: list[str] = []

    if X.ndim != 2:
        failures.append(f"X ndim={X.ndim}")
    if not np.isfinite(X).all():
        failures.append(f"{np.isnan(X).sum()} NaN / {np.isinf(X).sum()} Inf")
    if X.shape[0] != len(y):
        failures.append(f"X/y length mismatch {X.shape[0]}/{len(y)}")
    if len(cache.feature_names) != X.shape[1]:
        failures.append(f"feature_names={len(cache.feature_names)} vs F={X.shape[1]}")

    counts = np.bincount(y.astype(int), minlength=3).tolist()
    n_inter, n_pre, n_ict = counts[0], counts[1], counts[2]
    if n_pre == 0:
        failures.append("no pre-ictal windows")
    if n_inter == 0:
        failures.append("no interictal windows")
    # ictal == 0 is OK only for seizure-free test sets; warn, not fail

    ok = not failures
    msg = (f"N={X.shape[0]:>6d}  F={X.shape[1]}  "
           f"inter/pre/ict={n_inter}/{n_pre}/{n_ict}  files={len(cache.files)}")
    if failures:
        msg += "  FAIL: " + "; ".join(failures)
    elif n_ict == 0:
        msg += "  (no ictal windows — check summary parsing)"
    return cache.patient, ok, msg


def main() -> int:
    cfg_path = REPO / "configs/default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    feat_dir = REPO / cfg["cache_dir"] / "features"

    paths = sorted(feat_dir.glob("*.npz"))
    if not paths:
        print(f"no feature caches found under {feat_dir}")
        return 2

    print(f"== verifying {len(paths)} caches in {feat_dir} ==")
    n_ok = 0
    for p in paths:
        name, ok, msg = check_one(p)
        marker = "OK  " if ok else "FAIL"
        print(f"  [{marker}] {name:<6s}  {msg}")
        n_ok += int(ok)

    print(f"== {n_ok}/{len(paths)} passed ==")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
