#!/bin/bash
# submit_crop_shortcut_epochs_diag.sh — crop-shortcut diagnostic across training epochs.
#
# Runs scripts/diagnose_crop_shortcut.py on the per-epoch checkpoints a run saved via
# training.save_epochs (checkpoints/<run>/epoch<N>.pt), plus its best.pt. Shows whether
# position dependence and crop-level AP change over epochs while tile-vote val AP falls.
#
# Usage:
#   sbatch scripts/submit_crop_shortcut_epochs_diag.sh [max_frames] [run_version]
#   FOLDS="4 1" SUFFIX=careful-ep EPOCHS="1 3 10" sbatch scripts/submit_crop_shortcut_epochs_diag.sh
#SBATCH --job-name=sfx-crop-epochs
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/crop-epochs-%j.out
#SBATCH --error=logs/crop-epochs-%j.err

set -euo pipefail

MAX_FRAMES="${1:-1200}"
RUN_VERSION="${2:-v2}"
FOLDS="${FOLDS:-4 1}"
SUFFIX="${SUFFIX:-careful-ep}"
EPOCHS="${EPOCHS:-1 3 10}"
OUT_DIR="docs/figures/crop_shortcut"

module load mamba/latest
source activate sfx-hitfinder
mkdir -p logs "${OUT_DIR}"

for FOLD in ${FOLDS}; do
    RUN_DIR="checkpoints/vits16-mae-finetune-${RUN_VERSION}-fold${FOLD}-seed42-${SUFFIX}"
    for EP in ${EPOCHS}; do
        CKPT="${RUN_DIR}/epoch${EP}.pt"
        if [ ! -f "${CKPT}" ]; then
            echo "[skip] fold ${FOLD} epoch ${EP}: no checkpoint at ${CKPT}"
            continue
        fi
        echo "=============== fold ${FOLD} / ${SUFFIX} / epoch ${EP} ==============="
        python -u scripts/diagnose_crop_shortcut.py \
            --arch vit \
            --config configs/ssl/mae_finetune.yaml \
            --checkpoint "${CKPT}" \
            --fold "${FOLD}" \
            --split cross_detector_eval \
            --max-frames "${MAX_FRAMES}" \
            --out "${OUT_DIR}/fold${FOLD}_finetune-${SUFFIX}_epoch${EP}_${RUN_VERSION}_cross.json" \
            || echo "[FAILED] fold ${FOLD} epoch ${EP}"
    done
done

echo "[done] results in ${OUT_DIR}/"
