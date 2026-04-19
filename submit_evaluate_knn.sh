#!/bin/bash
#SBATCH --job-name=chbmit-knn
#SBATCH --output=logs/knn_%j.out
#SBATCH --error=logs/knn_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_homogen
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# ──────────────────────────────────────────────────────────────────────────
# kNN probe on a trained CEBRA embedding — computes sens / spec / AUROC for
# both framings (ictal-vs-rest detection, pre-ictal-vs-interictal forecasting)
# in both stratified and leave-one-patient-out splits.
#
# Usage:
#   mkdir -p logs
#   sbatch --export=ALL,EMBEDDING=cache/models/pooled_4858141_embedding.npz \
#          submit_evaluate_knn.sh
#
#   # defaults + overrides:
#   sbatch --export=ALL,EMBEDDING=<path>,K=11 submit_evaluate_knn.sh
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Match whichever activation the rest of the pipeline uses. Swap for
# conda activate <env> once you migrate off the .venv.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EMBEDDING="${EMBEDDING:?Set EMBEDDING=cache/models/pooled_<JOBID>_embedding.npz}"
K="${K:-5}"
TEST_FRAC="${TEST_FRAC:-0.2}"
SEED="${SEED:-0}"

if [[ ! -f "${EMBEDDING}" ]]; then
    echo "ERROR: embedding not found at ${EMBEDDING}"
    exit 1
fi

mkdir -p outputs logs

echo "=== kNN probe ==="
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Embedding: ${EMBEDDING}"
echo "k:         ${K}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-?}"
echo ""

echo "--- [1/2] stratified 80/20 ---"
python -u scripts/06_evaluate_knn.py \
    --embedding "${EMBEDDING}" \
    --mode stratified \
    --k "${K}" \
    --test-frac "${TEST_FRAC}" \
    --seed "${SEED}"

echo ""
echo "--- [2/2] leave-one-patient-out ---"
python -u scripts/06_evaluate_knn.py \
    --embedding "${EMBEDDING}" \
    --mode lopo \
    --k "${K}" \
    --seed "${SEED}"

echo ""
echo "=== Done: $(date) ==="
ls -lh outputs/knn_metrics_*.json
