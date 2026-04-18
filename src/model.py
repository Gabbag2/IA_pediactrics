"""xCEBRA-flavoured model wrapper.

Two training paths are exposed:

* :func:`fit_hybrid_cebra` — uses the high-level ``cebra.CEBRA`` sklearn
  estimator with ``hybrid=True`` so that time-contrastive and behaviour-
  contrastive objectives are trained jointly on a shared embedding. Fast and
  well-tested; the resulting torch model is compatible with
  ``cebra.attribution.init`` for Jacobian-based attribution.

* :func:`fit_xcebra_subspace` — builds a
  ``cebra.models.SubspaceMultiobjectiveModel`` that slices the embedding into a
  behaviour subspace (dims ``0..behav_dims``) and a time subspace
  (``behav_dims..latent_dim``) and trains with ``JacobianReg`` on a rampup
  schedule. This is the faithful xCEBRA recipe; it's slower and more finicky,
  so we keep it as the stretch-goal path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import cebra
import cebra.models


@dataclass
class TrainConfig:
    latent_dim: int = 11
    behavior_dims: int = 4
    architecture: str = "offset1-model"
    conditional: str = "time_delta"
    temperature: float = 1.0
    time_offsets: int = 10
    epochs: int = 3000
    batch_size: int = 512
    learning_rate: float = 3e-4
    num_hidden_units: int = 128
    hybrid: bool = True
    device: str = "cuda_if_available"
    seed: int = 0
    jacobian_reg_max: float = 0.0
    jacobian_warmup_frac: float = 0.25
    jacobian_rampup_frac: float = 0.50


def _pick_device(pref: str = "cuda_if_available") -> str:
    """Return a usable torch device name.

    Tries CUDA first (probes a tiny tensor op to catch broken drivers), then
    Apple-silicon MPS, else CPU. The probe means we gracefully fall back when
    ``torch.cuda.is_available()`` returns True but the runtime later fails
    (e.g. mismatched CUDA libs, driver reset).
    """
    if pref not in ("cuda_if_available", "auto"):
        return pref
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda") + 1  # quick smoke test
            name = torch.cuda.get_device_name(0)
            print(f"[device] using CUDA ({name})")
            return "cuda"
        except Exception as e:  # pragma: no cover — hardware-dependent
            print(f"[device] CUDA advertised but unusable ({e}); falling back")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps") + 1
            print("[device] using MPS (Apple Silicon)")
            return "mps"
        except Exception as e:  # pragma: no cover
            print(f"[device] MPS advertised but unusable ({e}); falling back")
    print("[device] using CPU")
    return "cpu"


def fit_hybrid_cebra(
    X: np.ndarray,
    y_state: np.ndarray,
    cfg: TrainConfig,
) -> cebra.CEBRA:
    """Train a hybrid CEBRA model on feature matrix ``X`` with behaviour label
    ``y_state`` (int array).

    Returns the trained sklearn estimator. Access the underlying torch module
    via ``model.model_``.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _pick_device(cfg.device)

    est = cebra.CEBRA(
        model_architecture=cfg.architecture,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        temperature=cfg.temperature,
        output_dimension=cfg.latent_dim,
        max_iterations=cfg.epochs,
        time_offsets=cfg.time_offsets,
        conditional=cfg.conditional,
        device=device,
        hybrid=cfg.hybrid,
        num_hidden_units=cfg.num_hidden_units,
        verbose=True,
    )
    # Hybrid mode (time + behavior) in the sklearn API requires a *continuous*
    # behaviour label. We pass the state code as a float — the ordinal gap
    # between interictal(0)/pre-ictal(1)/ictal(2) is enough signal for the
    # time_delta distribution to group same-state windows together.
    est.fit(X.astype(np.float32), y_state.astype(np.float32))
    return est


# ----------------------------------------------------------------------------
# Stretch-goal path: full xCEBRA subspace training with JacobianReg rampup
# ----------------------------------------------------------------------------


class JacobianSchedule:
    """Linear rampup schedule for the JacobianReg weight.

    off for steps < warmup*total, linear from 0 → max_weight until
    (warmup+rampup)*total, then held at max_weight.
    """

    def __init__(self, total_steps: int, warmup_frac: float,
                 rampup_frac: float, max_weight: float) -> None:
        self.total = max(1, total_steps)
        self.w0 = int(warmup_frac * self.total)
        self.w1 = int((warmup_frac + rampup_frac) * self.total)
        self.max_weight = max_weight

    def __call__(self, step: int) -> float:
        if step < self.w0 or self.max_weight == 0.0:
            return 0.0
        if step >= self.w1:
            return self.max_weight
        return self.max_weight * (step - self.w0) / max(1, self.w1 - self.w0)


def _info_nce(z_ref: torch.Tensor, z_pos: torch.Tensor,
              z_neg: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard InfoNCE with cosine similarity (z_* expected ℓ2-normalised)."""
    pos = (z_ref * z_pos).sum(-1) / temperature
    neg = (z_ref @ z_neg.T) / temperature
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return nn.functional.cross_entropy(logits, targets)


def _sample_behavior_positives(
    y: np.ndarray, ref_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """For each reference window, sample a positive from the same state class."""
    out = np.empty_like(ref_idx)
    by_state = {c: np.where(y == c)[0] for c in np.unique(y)}
    for i, k in enumerate(ref_idx):
        pool = by_state[y[k]]
        out[i] = rng.choice(pool)
    return out


def fit_xcebra_subspace(
    X: np.ndarray,
    y_state: np.ndarray,
    cfg: TrainConfig,
) -> dict:
    """Train an xCEBRA subspace model with JacobianReg rampup.

    Uses ``cebra.models.create_multiobjective_model`` on an ``offset1-model``
    backbone with two slices: [0, behavior_dims) for behaviour-contrastive
    (state label), [behavior_dims, latent_dim) for time-contrastive.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = _pick_device(cfg.device)

    backbone = cebra.models.init(
        cfg.architecture,
        num_neurons=X.shape[1],
        num_units=cfg.num_hidden_units,
        num_output=cfg.latent_dim,
    )
    feature_ranges = [
        slice(0, cfg.behavior_dims),
        slice(cfg.behavior_dims, cfg.latent_dim),
    ]
    model = cebra.models.create_multiobjective_model(
        backbone, feature_ranges=feature_ranges, renormalize=True
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    jac_reg = cebra.models.JacobianReg(n=1).to(device)
    schedule = JacobianSchedule(
        cfg.epochs, cfg.jacobian_warmup_frac, cfg.jacobian_rampup_frac,
        cfg.jacobian_reg_max,
    )

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = len(X_t)
    losses: list[float] = []

    for step in range(cfg.epochs):
        ref_idx = rng.integers(0, n, size=cfg.batch_size)
        neg_idx = rng.integers(0, n, size=cfg.batch_size)
        # behaviour positives: same state class
        behav_pos_idx = _sample_behavior_positives(y_state, ref_idx, rng)
        # time positives: nearby in index (proxy for time since windows are ordered)
        time_pos_idx = np.clip(
            ref_idx + rng.integers(-cfg.time_offsets, cfg.time_offsets + 1,
                                    size=cfg.batch_size),
            0, n - 1,
        )

        x_ref = X_t[ref_idx].requires_grad_(True)
        x_behav_pos = X_t[behav_pos_idx]
        x_time_pos = X_t[time_pos_idx]
        x_neg = X_t[neg_idx]

        # Forward — SubspaceMultiobjectiveModel returns a tuple of per-slice
        # (and renormalised) embeddings when split_outputs=True.
        z_ref = model(x_ref.unsqueeze(-1))
        z_bpos = model(x_behav_pos.unsqueeze(-1))
        z_tpos = model(x_time_pos.unsqueeze(-1))
        z_neg = model(x_neg.unsqueeze(-1))

        # z_ref is a tuple (behav_embed, time_embed) when split_outputs=True
        loss_behav = _info_nce(z_ref[0].squeeze(-1), z_bpos[0].squeeze(-1),
                               z_neg[0].squeeze(-1), cfg.temperature)
        loss_time = _info_nce(z_ref[1].squeeze(-1), z_tpos[1].squeeze(-1),
                              z_neg[1].squeeze(-1), cfg.temperature)
        loss = loss_behav + loss_time

        lam = schedule(step)
        if lam > 0.0:
            # Concatenate behaviour+time slices for the Jacobian regulariser.
            z_full = torch.cat([z_ref[0], z_ref[1]], dim=-1).squeeze(-1)
            loss = loss + lam * jac_reg(x_ref, z_full)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % max(1, cfg.epochs // 20) == 0:
            print(f"[xcebra] step {step}/{cfg.epochs} loss={loss.item():.4f} "
                  f"behav={loss_behav.item():.4f} time={loss_time.item():.4f} lam={lam:.4g}")

    return dict(model=model, losses=losses, feature_ranges=feature_ranges, cfg=cfg)


def embed(est: cebra.CEBRA, X: np.ndarray) -> np.ndarray:
    return est.transform(X.astype(np.float32))


def save(est: cebra.CEBRA, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # CEBRA supports its own save helper, but a torch checkpoint is enough
    torch.save({
        "model_state_dict": est.model_.state_dict(),
        "state_dict": est.state_dict_,
        "feature_names": getattr(est, "feature_names_in_", None),
    }, path)
