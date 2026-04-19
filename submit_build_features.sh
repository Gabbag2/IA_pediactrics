#!/bin/bash
#SBATCH --job-name=chbmit-features
#SBATCH --output=logs/chbmit_features_%A_%a.out
#SBATCH --error=logs/chbmit_features_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_homogen
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=0-4

# ──────────────────────────────────────────────────────────────────────────
# Feature extraction for CHB-MIT — one patient per array task.
#
# Runs in parallel on the ``cpu_homogen`` partition (Cascade Lake, 192 GB / 32
# cores per node). Sample entropy and PLV are the bottlenecks; numpy/scipy use
# OpenBLAS threads under the hood (set below), but we also pin antropy to a
# single worker and lean on the array for patient-level parallelism.
#
# Usage:
#   mkdir -p logs
#   # default: 5 core patients (array 0..4)
#   sbatch submit_build_features.sh
#
#   # override patient list (array must match length-1)
#   sbatch --array=0-22 --export=ALL,PATIENTS="$(ls /path/to/chbmit)" \
#          submit_build_features.sh
#
#   # different data root / output
#   sbatch --export=ALL,DATA_ROOT=/scratch/chbmit submit_build_features.sh
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Activate environment (adjust if your cluster uses conda)
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# BLAS thread pool — set to --cpus-per-task so numpy/scipy saturate the node
# share but don't oversubscribe across array tasks on a shared node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Allow a patients file (one comma-separated line, or one patient per line) to
# sidestep SLURM's --export comma-splitting bug. Pass with:
#   sbatch --export=ALL,PATIENTS_FILE=configs/patients_all.txt submit_build_features.sh
if [[ -n "${PATIENTS_FILE:-}" && -f "${PATIENTS_FILE}" ]]; then
    PATIENTS="$(tr '\n' ',' < "${PATIENTS_FILE}" | sed 's/,\+/,/g; s/,$//')"
fi
PATIENTS="${PATIENTS:-chb01,chb03,chb05,chb08,chb10}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/chbmit/chbmit}"
CONFIG="${CONFIG:-configs/default.yaml}"
echo "PATIENTS_COUNT: $(echo "${PATIENTS}" | tr ',' '\n' | wc -l)"
echo "PATIENTS: ${PATIENTS}"

# Split CSV into an array so $SLURM_ARRAY_TASK_ID picks one patient.
IFS=',' read -ra PATIENT_ARR <<< "${PATIENTS}"
idx="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${idx}" -ge "${#PATIENT_ARR[@]}" ]]; then
    echo "Array index ${idx} >= ${#PATIENT_ARR[@]} patients — nothing to do."
    exit 0
fi
PATIENT="${PATIENT_ARR[$idx]}"

OUT_NPZ="cache/features/${PATIENT}.npz"
if [[ "${FORCE:-0}" != "1" && -f "${OUT_NPZ}" ]]; then
    echo "[skip] ${OUT_NPZ} already exists (set FORCE=1 to rebuild)."
    ls -lh "${OUT_NPZ}"
    exit 0
fi

echo "=== CHB-MIT feature extraction ==="
echo "Job:       ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} [task ${idx}]"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Patient:   ${PATIENT}"
echo "Data root: ${DATA_ROOT}"
echo "Config:    ${CONFIG}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-?}"
echo ""

mkdir -p cache/features logs

python scripts/01_build_features.py \
    --config "${CONFIG}" \
    --patients "${PATIENT}" \
    --data-root "${DATA_ROOT}"

echo ""
echo "=== Done: $(date) ==="
ls -lh "cache/features/${PATIENT}.npz"
