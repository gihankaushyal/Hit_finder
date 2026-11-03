#!/bin/bash
# submit_crop_shortcut_recipe_diag.sh — crop-shortcut diagnostic on the recipe-experiment runs.
#
# Same measurement as submit_crop_shortcut_diag.sh (scripts/diagnose_crop_shortcut.py,
# held-out detector), but for fine-tune runs tagged with --run-suffix, i.e.
#   checkpoints/vits16-mae-finetune-<run_version>-fold<N>-seed42-<suffix>/best.pt
#
# Usage:
#   sbatch scripts/submit_crop_shortcut_recipe_diag.sh [max_frames] [run_version]
#   FOLDS="4 1" SUFFIXES="careful lpft-careful" sbatch scripts/submit_crop_shortcut_recipe_diag.sh
#SBATCH --job-name=sfx-crop-recipe
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/crop-recipe-%j.out
#SBATCH --error=logs/crop-recipe-%j.err

set -euo pipefail

MAX_FRAMES="${1:-1200}"
RUN_VERSION="${2:-v2}"
FOLDS="${FOLDS:-4 1 2 3}"
SUFFIXES="${SUFFIXES:-careful lpft-careful}"
OUT_DIR="docs/figures/crop_shortcut"

module load mamba/latest
source activate sfx-hitfinder
mkdir -p logs "${OUT_DIR}"

for FOLD in ${FOLDS}; do
    for SUFFIX in ${SUFFIXES}; do
        CKPT="checkpoints/vits16-mae-finetune-${RUN_VERSION}-fold${FOLD}-seed42-${SUFFIX}/best.pt"
        if [ ! -f "${CKPT}" ]; then
            echo "[skip] fold ${FOLD} ${SUFFIX}: no checkpoint at ${CKPT}"
            continue
        fi
        echo "=============== fold ${FOLD} / finetune-${SUFFIX} ==============="
        # One failing arm must not abort the rest.
        python -u scripts/diagnose_crop_shortcut.py \
            --arch vit \
            --config configs/ssl/mae_finetune.yaml \
            --checkpoint "${CKPT}" \
            --fold "${FOLD}" \
            --split cross_detector_eval \
            --max-frames "${MAX_FRAMES}" \
            --out "${OUT_DIR}/fold${FOLD}_finetune-${SUFFIX}_${RUN_VERSION}_cross.json" \
            || echo "[FAILED] fold ${FOLD} ${SUFFIX}"
    done
done

echo "[done] results in ${OUT_DIR}/"
