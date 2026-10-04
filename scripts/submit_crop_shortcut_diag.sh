#!/bin/bash
# submit_crop_shortcut_diag.sh — measure crop-construction shortcut reliance.
#
# Runs scripts/diagnose_crop_shortcut.py over all 4 LODO folds for both Stage B
# arms (full fine-tune and linear probe) on the held-out detector, where AP has
# headroom. Results land in docs/figures/crop_shortcut/ as JSON.
#
# Usage:
#   sbatch scripts/submit_crop_shortcut_diag.sh [max_frames]
#SBATCH --job-name=sfx-crop-diag
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/crop-diag-%j.out
#SBATCH --error=logs/crop-diag-%j.err

set -euo pipefail

# Held-out detectors are hit-dominated, so a large pool is needed before enough
# peak-free frames accumulate to form the condition-A/B negative set.
MAX_FRAMES="${1:-1200}"
OUT_DIR="docs/figures/crop_shortcut"

module load mamba/latest
source activate sfx-hitfinder
mkdir -p logs "${OUT_DIR}"

for FOLD in 1 2 3 4; do
    for ARM in finetune probe; do
        CKPT="checkpoints/vits16-mae-${ARM}-fold${FOLD}-seed42-v2/best.pt"
        if [ ! -f "${CKPT}" ]; then
            echo "[skip] fold ${FOLD} ${ARM}: no checkpoint at ${CKPT}"
            continue
        fi
        echo "=============== fold ${FOLD} / ${ARM} ==============="
        python -u scripts/diagnose_crop_shortcut.py \
            --config configs/ssl/mae_finetune.yaml \
            --checkpoint "${CKPT}" \
            --fold "${FOLD}" \
            --split cross_detector_eval \
            --max-frames "${MAX_FRAMES}" \
            --out "${OUT_DIR}/fold${FOLD}_${ARM}_cross.json"
    done
done

echo "[done] results in ${OUT_DIR}/"
