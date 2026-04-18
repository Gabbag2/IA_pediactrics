"""Train a hybrid CEBRA model on one or more patients' cached features.

Example:
    python scripts/02_train_xcebra.py --patients chb01,chb03,chb05,chb08,chb10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from _bootstrap import setup

REPO = setup()

from src.dataset import PatientCache, pool  # noqa: E402
from src.model import TrainConfig  # noqa: E402
from src.train import fit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    ap.add_argument("--patients", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="Permutation control: shuffle state labels before training")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = REPO / cfg["cache_dir"] / "features"

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    caches = [PatientCache.load(cache_dir / f"{p}.npz") for p in patients]
    pooled = pool(caches)

    if args.shuffle_labels:
        rng = np.random.default_rng(cfg["training"]["seed"])
        pooled.y_state[:] = rng.permutation(pooled.y_state)

    tcfg = cfg["training"]
    train_cfg = TrainConfig(
        latent_dim=tcfg["latent_dim"],
        behavior_dims=tcfg["behavior_dims"],
        conditional=tcfg["conditional"],
        temperature=tcfg["temperature"],
        time_offsets=tcfg["time_offset"],
        epochs=args.epochs or tcfg["epochs"],
        batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"],
        seed=tcfg["seed"],
        hybrid=True,
    )

    out_dir = REPO / cfg["cache_dir"] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "shuffled" if args.shuffle_labels else "pooled"
    stem = args.out or f"{tag}_{'_'.join(patients)}"
    ckpt = out_dir / f"{stem}.pt"
    emb_path = out_dir / f"{stem}_embedding.npz"

    print(f"== training on {len(patients)} patients, N={pooled.X.shape[0]}, "
          f"F={pooled.X.shape[1]}, state counts={np.bincount(pooled.y_state.astype(int), minlength=3).tolist()} ==")
    meta = fit(pooled, train_cfg, ckpt)

    np.savez_compressed(
        emb_path,
        Z=meta["embedding"], y_state=pooled.y_state,
        t_to_onset=pooled.t_to_onset, patient_id=pooled.patient_id,
        patients=np.array(pooled.patients),
        feature_names=np.array(pooled.feature_names),
        train_idx=meta["train_idx"],
        shuffled=np.array(args.shuffle_labels),
    )
    print(f"  model → {ckpt}\n  embedding → {emb_path}")


if __name__ == "__main__":
    main()
