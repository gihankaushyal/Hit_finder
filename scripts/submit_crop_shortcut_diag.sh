#!/bin/bash
# submit_crop_shortcut_diag.sh — measure crop-construction shortcut reliance.
#
# Runs scripts/diagnose_crop_shortcut.py over the folds in $FOLDS (default "4 1") for
# the Stage B arms (full fine-tune, linear probe) plus the Track 1 ResNet18 baseline
# on the held-out detector, where AP has
# headroom. Results land in docs/figures/crop_shortcut/ as JSON.
#
# Usage:
#   sbatch scripts/submit_crop_shortcut_diag.sh [max_frames] [run_version]
#
#   run_version  pipeline generation of the Stage B runs to inspect (default v2), i.e.
#                checkpoints/vits16-mae-{finetune,probe}-<run_version>-fold<N>-seed42/best.pt
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
RUN_VERSION="${2:-v2}"
OUT_DIR="docs/figures/crop_shortcut"

module load mamba/latest
# sbatch inherits the submitting shell's active env; activating it a second time
# silently falls back to the base python (no numpy). Submit from a fresh terminal
# with no conda env active.
source activate sfx-hitfinder
mkdir -p logs "${OUT_DIR}"

# FOLDS env var overrides the default (fold 4 = largest fine-tune gap, fold 1 = hardest detector).
FOLDS="${FOLDS:-4 1}"
N_OK=0; N_SKIP=0

for FOLD in ${FOLDS}; do
    for ARM in finetune probe resnet; do
        if [ "${ARM}" = "resnet" ]; then
            CKPT="checkpoints/resnet18-asymmetric-${RUN_VERSION}-fold${FOLD}-seed42/best.pt"
            CONFIG="configs/supervised/resnet18_asymmetric.yaml"
            ARCH="resnet"
        else
            CKPT="checkpoints/vits16-mae-${ARM}-${RUN_VERSION}-fold${FOLD}-seed42/best.pt"
            CONFIG="configs/ssl/mae_finetune.yaml"
            ARCH="vit"
        fi
        if [ ! -f "${CKPT}" ]; then
            echo "[skip] fold ${FOLD} ${ARM}: no checkpoint at ${CKPT}"
            N_SKIP=$((N_SKIP + 1))
            continue
        fi
        echo "=============== fold ${FOLD} / ${ARM} ==============="
        python -u scripts/diagnose_crop_shortcut.py \
            --arch "${ARCH}" \
            --config "${CONFIG}" \
            --checkpoint "${CKPT}" \
            --fold "${FOLD}" \
            --split cross_detector_eval \
            --max-frames "${MAX_FRAMES}" \
            --out "${OUT_DIR}/fold${FOLD}_${ARM}_${RUN_VERSION}_cross.json"
        N_OK=$((N_OK + 1))
    done
done

echo "[done] ran=${N_OK} skipped=${N_SKIP}; results in ${OUT_DIR}/"
# A job that skipped every arm must not show up as COMPLETED.
if [ "${N_OK}" -eq 0 ]; then
    exit 1
fi
