#!/bin/bash
# submit_crop_shortcut_trial.sh — 100-frame smoke check of diagnose_crop_shortcut.py on one fold.
#
# Confirms the script runs end to end (pools fill, JSON is written) before the full
# scripts/submit_crop_shortcut_diag.sh run. Numbers from 100 frames are NOT conclusions.
#
# Usage:
#   sbatch scripts/submit_crop_shortcut_trial.sh [fold] [max_frames] [run_version]
#   e.g.  sbatch scripts/submit_crop_shortcut_trial.sh 4 100 v2
#SBATCH --job-name=sfx-crop-trial
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/crop-trial-%j.out
#SBATCH --error=logs/crop-trial-%j.err

set -euo pipefail

FOLD="${1:-4}"
MAX_FRAMES="${2:-100}"
RUN_VERSION="${3:-v2}"
OUT_DIR="docs/figures/crop_shortcut"

module load mamba/latest
# sbatch inherits the submitting shell's active env; activating it a second time
# silently falls back to the base python (no numpy). Submit from a fresh terminal
# with no conda env active.
source activate sfx-hitfinder
mkdir -p logs "${OUT_DIR}"
N_OK=0; N_SKIP=0; N_FAIL=0

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
    echo "=============== trial: fold ${FOLD} / ${ARM} (${MAX_FRAMES} frames) ==============="
    # One failing arm must not abort the rest, but it is counted and fails the job at the end.
    if python -u scripts/diagnose_crop_shortcut.py \
        --arch "${ARCH}" \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --fold "${FOLD}" \
        --split cross_detector_eval \
        --max-frames "${MAX_FRAMES}" \
        --out "${OUT_DIR}/trial_fold${FOLD}_${ARM}_${RUN_VERSION}.json"; then
        N_OK=$((N_OK + 1))
    else
        echo "[FAILED] fold ${FOLD} ${ARM}"
        N_FAIL=$((N_FAIL + 1))
    fi
done

echo "[done] ran=${N_OK} skipped=${N_SKIP} failed=${N_FAIL}; trial outputs in ${OUT_DIR}/trial_fold${FOLD}_*"
# A job that did nothing, or lost an arm, must not show up as COMPLETED.
if [ "${N_OK}" -eq 0 ] || [ "${N_FAIL}" -gt 0 ]; then
    exit 1
fi
