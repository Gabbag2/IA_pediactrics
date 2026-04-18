"""Compute per-patient Jacobian attribution maps.

Loads the trained model + embedding produced by ``02_train_xcebra.py`` plus
each patient's cached features. For each patient, attribution is computed on
pre-ictal windows (falls back to the full set if none are available).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import setup

REPO = setup()

from src.attribution import attribution_for_patient  # noqa: E402
from src.dataset import PatientCache, zscore_per_patient  # noqa: E402
from src.features import FeatureLayout  # noqa: E402


def _load_estimator(ckpt: Path, cfg: dict, X_shape: tuple[int, int]) -> object:
    """Re-instantiate a CEBRA estimator and load weights from the checkpoint."""
    import cebra
    import numpy as np

    from src.model import _pick_device
    device = _pick_device("cuda_if_available")

    tcfg = cfg["training"]
    est = cebra.CEBRA(
        model_architecture="offset1-model",
        output_dimension=tcfg["latent_dim"],
        time_offsets=tcfg["time_offset"],
        conditional=tcfg["conditional"],
        temperature=tcfg["temperature"],
        batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"],
        max_iterations=1,
        hybrid=True,
        num_hidden_units=128,
        device=device,
    )
    # A fit is needed to initialise `.model_`; run a minimal 1-step fit.
    dummy_X = np.random.default_rng(0).standard_normal(X_shape).astype(np.float32)
    # Continuous label required for hybrid mode.
    dummy_y = np.linspace(0, 2, X_shape[0], dtype=np.float32)
    est.fit(dummy_X, dummy_y)
    state = torch.load(ckpt, map_location="cpu")
    est.model_.load_state_dict(state["model_state_dict"])
    return est


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    ap.add_argument("--model", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--patients", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = REPO / cfg["cache_dir"] / "features"
    out_dir = REPO / cfg["outputs_dir"] / "attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    caches = [PatientCache.load(cache_dir / f"{p}.npz") for p in patients]

    layout = FeatureLayout(
        channels=tuple(cfg["channels_common"]),
        bands=tuple(cfg["bands"].keys()),
        plv_pairs=tuple((a, b) for a, b in cfg["homologous_pairs"]),
        plv_bands=tuple(cfg["plv_bands"]),
    )
    n_features = layout.total
    est = _load_estimator(Path(args.model), cfg,
                          X_shape=(max(64, caches[0].X.shape[0]), n_features))
    attr_device = str(next(est.model_.parameters()).device)

    results_summary = []
    for cache in caches:
        Z = zscore_per_patient(cache)
        preictal = Z[cache.y_state == 1]
        if len(preictal) < 32:
            X_attr = Z
            note = "fallback_all"
        else:
            X_attr = preictal
            note = "preictal"

        res = attribution_for_patient(
            est=est, X_patient=X_attr, feature_layout=layout,
            behavior_dims=cfg["training"]["behavior_dims"],
            device=attr_device,
        )

        out_path = out_dir / f"{cache.patient}.npz"
        np.savez_compressed(
            out_path,
            jacobian=res.jacobian_mean_abs,
            behav=res.behav_importance,
            ch_band=res.ch_band_map,
            channel_importance=res.channel_importance,
            plv=res.plv_importance,
            graph=res.graph_importance,
            feature_names=np.array(layout.feature_names()),
            channels=np.array(layout.channels),
            bands=np.array(layout.bands),
            note=np.array(note),
        )
        # Rank top channels / bands for quick human inspection; no claim
        # against clinical ground truth is made here.
        order = np.argsort(-res.channel_importance)
        top = [(str(layout.channels[i]), float(res.channel_importance[i])) for i in order[:5]]
        band_totals = {b: float(res.ch_band_map[:, j].sum())
                       for j, b in enumerate(layout.bands)}
        results_summary.append({
            "patient": cache.patient,
            "note": note,
            "top_channels": top,
            "band_totals": band_totals,
        })
        print(f"  [{cache.patient}] {note}  top={top[:3]}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2))
    print(f"  summary → {summary_path}")


if __name__ == "__main__":
    main()
