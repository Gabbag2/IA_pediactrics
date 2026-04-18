#!/bin/bash
#SBATCH --job-name=xcebra-chbmit
#SBATCH --output=logs/xcebra_chbmit_%j.out
#SBATCH --error=logs/xcebra_chbmit_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a100|h100|h200"

# ──────────────────────────────────────────────────────────────────────────
# xCEBRA on CHB-MIT pediatric scalp EEG — training + attribution + report
#
# Assumes ``cache/features/<patient>.npz`` has been built locally and rsync'd
# to the cluster (feature extraction is CPU-bound and does not need a GPU).
#
# Stages:
#   1. Train hybrid CEBRA on the 5 pooled core patients (5000 iters, GPU)
#   2. Per-patient Jacobian attribution on pre-ictal windows
#   3. Report: latent 3D, attribution heatmaps, trajectory figures, summary JSON
#
# Usage:
#   mkdir -p logs
#   sbatch submit_xcebra_chbmit.sh
#   # or override patients / iterations / run tag:
#   sbatch --export=ALL,PATIENTS="chb01,chb03",MAX_ITER=3000,TAG=debug \
#          submit_xcebra_chbmit.sh
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Activate environment (adjust if your cluster uses conda instead of venv)
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

PATIENTS="${PATIENTS:-chb01,chb03,chb05,chb08,chb10}"
MAX_ITER="${MAX_ITER:-5000}"
TAG="${TAG:-pooled_${SLURM_JOB_ID}}"
CONFIG="${CONFIG:-configs/default.yaml}"

MODEL_DIR="cache/models"
CKPT="${MODEL_DIR}/${TAG}.pt"
EMB="${MODEL_DIR}/${TAG}_embedding.npz"

echo "=== xCEBRA on CHB-MIT ==="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Config:   ${CONFIG}"
echo "Patients: ${PATIENTS}"
echo "Iters:    ${MAX_ITER}"
echo "Tag:      ${TAG}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

mkdir -p "${MODEL_DIR}" logs outputs

# ─── 1. Train ─────────────────────────────────────────────────────────────
echo "--- [1/3] training xCEBRA ---"
python scripts/02_train_xcebra.py \
    --config "${CONFIG}" \
    --patients "${PATIENTS}" \
    --epochs "${MAX_ITER}" \
    --out "${TAG}"

# ─── 2. Attribution ───────────────────────────────────────────────────────
echo ""
echo "--- [2/3] per-patient Jacobian attribution ---"
python scripts/03_attribution.py \
    --config "${CONFIG}" \
    --model "${CKPT}" \
    --patients "${PATIENTS}"

# ─── 3. Report ────────────────────────────────────────────────────────────
echo ""
echo "--- [3/3] report (figures + summary JSON) ---"
python scripts/04_report.py \
    --config "${CONFIG}" \
    --embedding "${EMB}"

echo ""
echo "=== Done: $(date) ==="
echo "Model:     ${CKPT}"
echo "Embedding: ${EMB}"
echo "Outputs:   outputs/"
