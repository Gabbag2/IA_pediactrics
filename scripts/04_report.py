"""Build figures + summary JSON from trained model + attribution outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from _bootstrap import setup

REPO = setup()

from src.eval_latent import (  # noqa: E402
    align_to_onset, centroid_distance, drift_velocity, local_dimensionality,
    local_variance,
)
from src.viz import (  # noqa: E402
    plot_attribution_heatmap, plot_latent_3d, plot_lateralization,
    plot_trajectory_metrics,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    ap.add_argument("--embedding", required=True,
                    help=".npz produced by 02_train_xcebra.py")
    ap.add_argument("--attribution-dir", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = REPO / cfg["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_dir = Path(args.attribution_dir or out_dir / "attribution")

    # --- F1: latent 3D scatter ------------------------------------------------
    emb = np.load(args.embedding, allow_pickle=False)
    Z = emb["Z"]
    y = emb["y_state"]
    pid = emb["patient_id"]
    patients = [str(p) for p in emb["patients"]]
    plot_latent_3d(Z, y, pid, patients, out_dir / "F1_latent3d.png")
    print(f"  F1 → {out_dir / 'F1_latent3d.png'}")

    # --- F2: attribution heatmaps (one per patient) --------------------------
    bands = list(cfg["bands"].keys())
    channels = list(cfg["channels_common"])
    for p in patients:
        attr_path = attr_dir / f"{p}.npz"
        if not attr_path.exists():
            print(f"  [skip F2] {p}: no attribution")
            continue
        a = np.load(attr_path, allow_pickle=False)
        plot_attribution_heatmap(
            a["ch_band"], channels, bands,
            title=f"{p} — behaviour Jacobian (channel × band)",
            path=out_dir / f"F2_attr_{p}.png",
        )
        print(f"  F2[{p}] → F2_attr_{p}.png")

    # --- F3: trajectory metrics aligned to onset, averaged over patients -----
    t_to = emb["t_to_onset"]
    behav = Z[:, : cfg["training"]["behavior_dims"]]
    inter_mask = y == 0
    centroid = behav[inter_mask].mean(axis=0) if inter_mask.any() else behav.mean(axis=0)
    velocity = drift_velocity(behav, dt=cfg["window"]["step_s"])
    variance = local_variance(behav, window=20)
    dist = centroid_distance(behav, centroid)
    # Local dim can be slow; use a subsample for sanity in report.
    subset = np.random.default_rng(0).choice(len(behav), size=min(3000, len(behav)),
                                             replace=False)
    dim_full = np.full(len(behav), np.nan, dtype=np.float32)
    dim_full[subset] = local_dimensionality(behav[subset], k=20)

    metrics_series = {}
    for name, series in [
        ("drift velocity", velocity),
        ("local variance", variance),
        ("distance to interictal centroid", dist),
        ("local dimensionality", dim_full),
    ]:
        res = align_to_onset(
            series.reshape(-1, 1), t_to,
            pre_horizon_s=cfg["labels"]["pre_ictal_s"],
            step_s=cfg["window"]["step_s"],
            post_horizon_s=60.0,
        )
        metrics_series[name] = np.nan_to_num(res["means"][:, 0])
    plot_trajectory_metrics(res["bin_edges"], metrics_series,
                            out_dir / "F3_trajectory.png")
    print(f"  F3 → {out_dir / 'F3_trajectory.png'}")

    # --- F4: lateralisation vs clinical truth --------------------------------
    summary_path = attr_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        plot_lateralization(
            [s["clinical"] for s in summary],
            [s["predicted"] for s in summary],
            out_dir / "F4_lateralization.png",
        )
        # Numerical summary
        hits = sum(
            1 for s in summary
            if s["clinical"] in ("left", "right") and s["clinical"] == s["predicted"]
        )
        valid = sum(1 for s in summary if s["clinical"] in ("left", "right"))
        acc = hits / valid if valid else float("nan")
        report = {
            "patients": [s["patient"] for s in summary],
            "lateralization_accuracy": acc,
            "per_patient": summary,
        }
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
        print(f"  F4 → {out_dir / 'F4_lateralization.png'} (acc={acc:.2f})")
    else:
        print("  [skip F4] no attribution summary found")


if __name__ == "__main__":
    main()
