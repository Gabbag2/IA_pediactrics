#!/bin/bash
#SBATCH --job-name=xcebra-lopo
#SBATCH --output=logs/xcebra_lopo_%A_%a.out
#SBATCH --error=logs/xcebra_lopo_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a100|h100|h200"
#SBATCH --array=0-22

# ──────────────────────────────────────────────────────────────────────────
# Leave-one-patient-out xCEBRA sweep over all 23 CHB-MIT patients.
#
# Each array task holds out one patient, trains the hybrid CEBRA model on the
# other 22, then runs attribution on the held-out patient. Produces 23 models
# + 23 held-out attribution maps.
#
# Prerequisites: cache/features/<patient>.npz for all 23 patients (run
# submit_build_features.sh with --array=0-22 first).
#
# Usage:
#   mkdir -p logs
#   sbatch submit_xcebra_lopo.sh
#
# Override:
#   sbatch --export=ALL,MAX_ITER=3000,PATIENTS_ALL="chb01,chb02,..." \
#          --array=0-N submit_xcebra_lopo.sh
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

PATIENTS_ALL="${PATIENTS_ALL:-chb01,chb02,chb03,chb04,chb05,chb06,chb07,chb08,chb09,chb10,chb11,chb12,chb13,chb14,chb15,chb16,chb17,chb18,chb19,chb20,chb21,chb22,chb23}"
MAX_ITER="${MAX_ITER:-5000}"
CONFIG="${CONFIG:-configs/default.yaml}"

IFS=',' read -ra PATIENT_ARR <<< "${PATIENTS_ALL}"
idx="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${idx}" -ge "${#PATIENT_ARR[@]}" ]]; then
    echo "Array index ${idx} >= ${#PATIENT_ARR[@]} patients — nothing to do."
    exit 0
fi
HOLDOUT="${PATIENT_ARR[$idx]}"

# Train patients = all except the held-out one
TRAIN_PATIENTS=""
for p in "${PATIENT_ARR[@]}"; do
    if [[ "$p" != "$HOLDOUT" ]]; then
        TRAIN_PATIENTS="${TRAIN_PATIENTS:+${TRAIN_PATIENTS},}${p}"
    fi
done

TAG="lopo_${HOLDOUT}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
CKPT="cache/models/${TAG}.pt"

echo "=== xCEBRA LOPO sweep ==="
echo "Job:      ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} [task ${idx}]"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Held out: ${HOLDOUT}"
echo "Train:    ${TRAIN_PATIENTS}"
echo "Iters:    ${MAX_ITER}"
echo "Tag:      ${TAG}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

mkdir -p cache/models logs outputs

# Train on the 22 non-held-out patients
python scripts/02_train_xcebra.py \
    --config "${CONFIG}" \
    --patients "${TRAIN_PATIENTS}" \
    --epochs "${MAX_ITER}" \
    --out "${TAG}"

# Attribution on the held-out patient only (that's the whole point of LOPO)
echo ""
echo "--- attribution on held-out ${HOLDOUT} ---"
python scripts/03_attribution.py \
    --config "${CONFIG}" \
    --model "${CKPT}" \
    --patients "${HOLDOUT}"

echo ""
echo "=== Done: $(date) ==="
