#!/bin/bash
# submit_jitter_measurements.sh — the two measurements that precede the positive-jitter runs.
#
#   1. scripts/measure_crop_stats.py     valid fraction, pad-band exposure and peaks per crop of
#                                        training crops under centred / uniform / uniform+within_frame,
#                                        next to the evaluation tiles (JSON in docs/figures/crop_stats/)
#   2. scripts/measure_padding_effect.py frame AP re-scored with only tiles >= 0.5 / 0.75 valid, for the
#                                        existing best.pt of each model (JSON in docs/figures/padding_effect/)
#
# Neither trains or changes anything. Usage (fresh terminal, no conda env active):
#   sbatch scripts/submit_jitter_measurements.sh [max_frames] [run_version]
#   FOLDS="4 1 2 3" sbatch scripts/submit_jitter_measurements.sh
#   SUFFIX=-jitter sbatch scripts/submit_jitter_measurements.sh   # measure the jittered runs instead
#   SEED=43 sbatch scripts/submit_jitter_measurements.sh          # another training seed
# SUFFIX is the run suffix of the checkpoints to measure (empty = baseline, -jitter, ...); the
# careful arm reads <prefix>-careful<SUFFIX>, e.g. ...-seed42-careful-jitter.
#SBATCH --job-name=sfx-jitter-meas
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/jitter-meas-%j.out
#SBATCH --error=logs/jitter-meas-%j.err

set -euo pipefail

MAX_FRAMES="${1:-1200}"
RUN_VERSION="${2:-v2}"
FOLDS="${FOLDS:-4 1}"
SEED="${SEED:-42}"
SUFFIX="${SUFFIX:-}"
STATS_DIR="docs/figures/crop_stats"
PAD_DIR="docs/figures/padding_effect"

module load mamba/latest
# sbatch inherits the submitting shell's active env; activating it a second time
# silently falls back to the base python (no numpy). Submit from a fresh terminal
# with no conda env active.
source activate sfx-hitfinder
mkdir -p logs "${STATS_DIR}" "${PAD_DIR}"
N_OK=0; N_PAD_OK=0; N_SKIP=0; N_FAIL=0

run() {  # run <label> <command...>: count success/failure without aborting the job
    local label="$1"; shift
    echo "=============== ${label} ==============="
    if "$@"; then N_OK=$((N_OK + 1)); else echo "[FAILED] ${label}"; N_FAIL=$((N_FAIL + 1)); fi
}

for FOLD in ${FOLDS}; do
    run "crop stats fold ${FOLD}" python -u scripts/measure_crop_stats.py \
        --config configs/ssl/mae_finetune.yaml --fold "${FOLD}" \
        --out "${STATS_DIR}/fold${FOLD}.json"

    for ARM in finetune probe resnet careful; do
        if [ "${ARM}" = "careful" ] && [[ "${SUFFIX}" == -careful* ]]; then
            echo "[skip] fold ${FOLD} careful: SUFFIX ${SUFFIX} already selects the careful runs (the finetune arm)"
            continue
        fi
        case "${ARM}" in
            resnet)  CKPT="checkpoints/resnet18-asymmetric-${RUN_VERSION}-fold${FOLD}-seed${SEED}${SUFFIX}/best.pt"
                     CONFIG="configs/supervised/resnet18_asymmetric.yaml"; ARCH=resnet ;;
            careful) CKPT="checkpoints/vits16-mae-finetune-${RUN_VERSION}-fold${FOLD}-seed${SEED}-careful${SUFFIX}/best.pt"
                     CONFIG="configs/ssl/mae_finetune_careful.yaml"; ARCH=vit ;;
            *)       CKPT="checkpoints/vits16-mae-${ARM}-${RUN_VERSION}-fold${FOLD}-seed${SEED}${SUFFIX}/best.pt"
                     CONFIG="configs/ssl/mae_finetune.yaml"; ARCH=vit ;;
        esac
        if [ ! -f "${CKPT}" ]; then
            echo "[skip] fold ${FOLD} ${ARM}: no checkpoint at ${CKPT}"
            N_SKIP=$((N_SKIP + 1))
            continue
        fi
        OK_BEFORE=${N_OK}
        run "padding effect fold ${FOLD} ${ARM}" python -u scripts/measure_padding_effect.py \
            --arch "${ARCH}" --config "${CONFIG}" --checkpoint "${CKPT}" --fold "${FOLD}" \
            --max-frames "${MAX_FRAMES}" --out "${PAD_DIR}/fold${FOLD}_${ARM}_${RUN_VERSION}_seed${SEED}${SUFFIX}.json"
        N_PAD_OK=$((N_PAD_OK + N_OK - OK_BEFORE))
    done
done

echo "[done] ran=${N_OK} skipped=${N_SKIP} failed=${N_FAIL}; results in ${STATS_DIR}/ and ${PAD_DIR}/"
# A job that did nothing, or lost a step, must not show up as COMPLETED. The crop-stats
# runs succeed on their own, so count the padding-effect steps separately.
if [ "${N_PAD_OK}" -eq 0 ] || [ "${N_FAIL}" -gt 0 ]; then
    exit 1
fi
