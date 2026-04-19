"""kNN probe on the trained CEBRA embedding.

Loads the ``*_embedding.npz`` produced by ``02_train_xcebra.py`` and scores a
simple k-nearest-neighbours classifier on the 3-dim latent. Reports a 3x3
(interictal / pre-ictal / ictal) confusion matrix and binary sensitivity /
specificity for two clinically meaningful framings:

    * seizure detection   : ictal      vs {interictal, pre-ictal}
    * seizure forecasting : pre-ictal  vs interictal (ictal windows excluded)

Two split modes:

    * ``stratified`` (default)  — 80/20 stratified by (patient, state).
      Optimistic baseline; shows "does the embedding separate states at all".

    * ``lopo``                  — leave-one-patient-out.
      Honest generalisation test; with 5 pooled patients this yields 5 folds
      and the reported metrics are means (± std) across folds.

Example::

    python scripts/06_evaluate_knn.py \\
        --embedding cache/models/pooled_4858141_embedding.npz \\
        --mode stratified --k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier

from _bootstrap import setup

REPO = setup()


STATE_NAMES = ["interictal", "pre-ictal", "ictal"]


def _binary_metrics(y_true_pos: np.ndarray, y_pred_pos: np.ndarray) -> dict:
    tp = int(((y_true_pos == 1) & (y_pred_pos == 1)).sum())
    fn = int(((y_true_pos == 1) & (y_pred_pos == 0)).sum())
    fp = int(((y_true_pos == 0) & (y_pred_pos == 1)).sum())
    tn = int(((y_true_pos == 0) & (y_pred_pos == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    bal_acc = 0.5 * (sens + spec)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, balanced_accuracy=bal_acc)


def _fit_score(
    Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, y_te: np.ndarray,
    k: int,
) -> dict:
    # Points live on the unit sphere (cosine distance) — euclidean kNN is
    # order-preserving equivalent to cosine there and lets us use KDTree.
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance",
                               algorithm="auto")
    clf.fit(Z_tr, y_tr)
    y_pred = clf.predict(Z_te)
    y_proba = clf.predict_proba(Z_te)  # (N, C) in clf.classes_ order

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_te, y_pred, labels=[0, 1, 2], zero_division=0,
    )

    # One-vs-rest AUROC, robust to missing classes in the test split.
    auroc = {}
    proba_by_cls = {int(c): i for i, c in enumerate(clf.classes_)}
    for c, name in enumerate(STATE_NAMES):
        if c not in proba_by_cls:
            auroc[name] = float("nan")
            continue
        y_bin = (y_te == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auroc[name] = float("nan")
            continue
        auroc[name] = float(roc_auc_score(y_bin, y_proba[:, proba_by_cls[c]]))

    # Binary framings
    detection = _binary_metrics((y_te == 2).astype(int),
                                (y_pred == 2).astype(int))
    fmask = y_te != 2
    forecasting = _binary_metrics((y_te[fmask] == 1).astype(int),
                                  (y_pred[fmask] == 1).astype(int))

    return dict(
        confusion_matrix=cm.tolist(),
        per_class=[
            dict(state=name, precision=float(prec[i]),
                 recall=float(rec[i]), f1=float(f1[i]),
                 support=int(sup[i]))
            for i, name in enumerate(STATE_NAMES)
        ],
        auroc=auroc,
        detection_ictal_vs_rest=detection,
        forecasting_preictal_vs_interictal=forecasting,
    )


def _stratified_split(
    y: np.ndarray, pid: np.ndarray, test_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(y), dtype=bool)
    for p in np.unique(pid):
        for s in np.unique(y):
            idx = np.where((pid == p) & (y == s))[0]
            if len(idx) < 2:
                continue
            rng.shuffle(idx)
            n_te = max(1, int(round(test_frac * len(idx))))
            test_mask[idx[:n_te]] = True
    return ~test_mask, test_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True,
                    help="Path to *_embedding.npz from 02_train_xcebra.py")
    ap.add_argument("--mode", choices=["stratified", "lopo"],
                    default="stratified")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="[stratified only] fraction of windows per (patient, state) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb = np.load(args.embedding, allow_pickle=False)
    Z = emb["Z"].astype(np.float32)
    y = emb["y_state"].astype(int)
    pid = emb["patient_id"].astype(int)
    patients = [str(p) for p in emb["patients"]]

    print(f"== kNN probe on {args.embedding} ==")
    print(f"  Z={Z.shape}  patients={patients}  "
          f"state counts={np.bincount(y, minlength=3).tolist()}  "
          f"k={args.k}  mode={args.mode}")

    report: dict = dict(mode=args.mode, k=args.k, patients=patients,
                        embedding=str(args.embedding))

    if args.mode == "stratified":
        tr, te = _stratified_split(y, pid, args.test_frac, args.seed)
        print(f"  train={tr.sum()}  test={te.sum()}  "
              f"(test state counts={np.bincount(y[te], minlength=3).tolist()})")
        res = _fit_score(Z[tr], y[tr], Z[te], y[te], args.k)
        report.update(res)
        _print_fold("overall", res)
    else:
        folds = []
        for i, name in enumerate(patients):
            te = pid == i
            tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            res = _fit_score(Z[tr], y[tr], Z[te], y[te], args.k)
            res["held_out"] = name
            folds.append(res)
            _print_fold(name, res)
        report["folds"] = folds
        report["summary"] = _aggregate(folds)
        print()
        _print_summary(report["summary"])

    out_path = Path(args.out) if args.out else \
        REPO / "outputs" / f"knn_metrics_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  metrics → {out_path}")


def _print_fold(tag: str, r: dict) -> None:
    d = r["detection_ictal_vs_rest"]
    f = r["forecasting_preictal_vs_interictal"]
    print(f"\n  --- {tag} ---")
    print(f"    confusion (rows=true, cols=pred, order=inter/pre/ict):")
    for row in r["confusion_matrix"]:
        print(f"      {row}")
    for pc in r["per_class"]:
        print(f"    {pc['state']:>10s}  P={pc['precision']:.3f} "
              f"R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
    print(f"    AUROC (OvR) : " + "  ".join(
        f"{k}={v:.3f}" if v == v else f"{k}=nan" for k, v in r["auroc"].items()
    ))
    print(f"    detection   (ictal vs rest)          "
          f"sens={d['sensitivity']:.3f}  spec={d['specificity']:.3f}  "
          f"PPV={d['ppv']:.3f}  bal_acc={d['balanced_accuracy']:.3f}")
    print(f"    forecasting (pre-ictal vs interictal) "
          f"sens={f['sensitivity']:.3f}  spec={f['specificity']:.3f}  "
          f"PPV={f['ppv']:.3f}  bal_acc={f['balanced_accuracy']:.3f}")


def _aggregate(folds: list[dict]) -> dict:
    def _mean_std(vals: list[float]) -> tuple[float, float]:
        arr = np.array([v for v in vals if v == v], dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std(ddof=0))

    out: dict = {}
    for key in ("detection_ictal_vs_rest", "forecasting_preictal_vs_interictal"):
        m = {}
        for metric in ("sensitivity", "specificity", "ppv", "npv",
                       "balanced_accuracy"):
            mean, std = _mean_std([f[key][metric] for f in folds])
            m[metric] = dict(mean=mean, std=std)
        out[key] = m
    auroc = {}
    for cls in STATE_NAMES:
        mean, std = _mean_std([f["auroc"][cls] for f in folds])
        auroc[cls] = dict(mean=mean, std=std)
    out["auroc"] = auroc
    return out


def _print_summary(s: dict) -> None:
    print("  === LOPO summary (mean ± std across folds) ===")
    for key, label in [("detection_ictal_vs_rest", "detection (ictal vs rest)"),
                       ("forecasting_preictal_vs_interictal",
                        "forecasting (pre-ictal vs interictal)")]:
        b = s[key]
        print(f"    {label:<42s} "
              f"sens={b['sensitivity']['mean']:.3f}±{b['sensitivity']['std']:.3f} "
              f"spec={b['specificity']['mean']:.3f}±{b['specificity']['std']:.3f} "
              f"PPV={b['ppv']['mean']:.3f}±{b['ppv']['std']:.3f} "
              f"bal_acc={b['balanced_accuracy']['mean']:.3f}±"
              f"{b['balanced_accuracy']['std']:.3f}")
    for cls, v in s["auroc"].items():
        print(f"    AUROC {cls:<10s} = {v['mean']:.3f}±{v['std']:.3f}")


if __name__ == "__main__":
    main()
