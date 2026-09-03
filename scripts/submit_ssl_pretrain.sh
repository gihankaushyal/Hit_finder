#!/bin/bash
# Usage:
#   Smoke run (~100 epochs): sbatch scripts/submit_ssl_pretrain.sh <fold_id> 100
#   Full run  (400 epochs):  sbatch scripts/submit_ssl_pretrain.sh <fold_id>
#
# CXI files are staged from NFS to local NVMe (/tmp) before training starts.
# This eliminates the NFS I/O bottleneck (~1.3 h/epoch → ~5 min/epoch target).
# Staged files are cleaned up automatically when the job exits.
#SBATCH --job-name=sfx-ssl-pretrain
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ssl-pretrain-%j.out
#SBATCH --error=logs/ssl-pretrain-%j.err

set -euo pipefail
FOLD="${1:?fold id required (1-4)}"
EPOCHS="${2:-}"   # optional; omit for full 400-epoch run

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

# ---------------------------------------------------------------------------
# Stage CXI data to local NVMe (/tmp) — avoids NFS read bottleneck
# ---------------------------------------------------------------------------
NFS_SRC="/data/bioxfel/user/gihan/Resonet/production"
STAGE="/tmp/sfx_stage_${SLURM_JOB_ID}"
mkdir -p "${STAGE}"

# Clean up staged data when the job exits (normal, error, or scancel)
trap 'echo "[stage] cleaning up ${STAGE}"; rm -rf "${STAGE}"' EXIT

echo "[stage] copying CXI files to ${STAGE} (background, parallel) ..."
for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
    src="${NFS_SRC}/${det_dir}"
    if [ -d "${src}" ]; then
        mkdir -p "${STAGE}/${det_dir}"
        cp "${src}"/compressed*.cxi "${STAGE}/${det_dir}/" &
    fi
done
wait
echo "[stage] staging complete. $(du -sh ${STAGE} | cut -f1) copied to local NVMe."

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
EPOCHS_ARG=""
if [ -n "${EPOCHS}" ]; then
    EPOCHS_ARG="--epochs ${EPOCHS}"
fi

python -u -m src.training.train_ssl_pretrain \
    --config configs/ssl/mae_pretrain.yaml \
    --fold "${FOLD}" \
    --resume \
    --stage-dir "${STAGE}" \
    ${EPOCHS_ARG}
