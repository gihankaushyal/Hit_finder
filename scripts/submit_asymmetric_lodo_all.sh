#!/bin/bash
# submit_asymmetric_lodo_all.sh — Orchestrate ResNet18 LODO training across folds.
#
# Stages each fold's frame-cache working set to /tmp/sfx_frame_cache on NVMe,
# runs that fold's LODO training job from NVMe, then cleans up before moving to
# the next fold. Folds run sequentially because each fold's cache working set
# fills the NVMe tier — they cannot overlap. All jobs are pinned to scg020 so
# they share the same /tmp NVMe filesystem.
#
# Job chain (SLURM dependencies):
#   stage(1) → train(1) → cleanup(1)
#                             ↓ afterany
#                          stage(2) → train(2) → cleanup(2) → …
#
# Usage:
#   bash scripts/submit_asymmetric_lodo_all.sh [OPTIONS]
#
# Options:
#   --folds <1 2 3 4>   Space-separated fold IDs to run (default: 1 2 3 4).
#   -h, --help          Show this help message and exit.
#
# Examples:
#   bash scripts/submit_asymmetric_lodo_all.sh                 # all 4 folds
#   bash scripts/submit_asymmetric_lodo_all.sh --folds 3 4    # skip folds 1-2 (already done)
#
# See also:
#   scripts/submit_asymmetric_lodo_fold.sh   single-fold submission (reads CACHE_NVME env var)
#   scripts/stage_frame_cache.sh             per-fold staging job (called automatically)
#   scripts/cleanup_ssl_stage.sh             cleanup job (called automatically)

set -euo pipefail

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \{0,1\}//; p }' "$0"
}

FOLDS=(1 2 3 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --folds) shift; FOLDS=(); while [[ $# -gt 0 && "$1" =~ ^[1-4]$ ]]; do FOLDS+=("$1"); shift; done ;;
        *) echo "Error: unknown option '$1'" >&2; echo "Run '$0 --help' for usage." >&2; exit 1 ;;
    esac
done

if [ "${#FOLDS[@]}" -eq 0 ]; then
    echo "Error: no valid fold IDs provided." >&2; exit 1
fi

for fold in "${FOLDS[@]}"; do
    if [[ ! "${fold}" =~ ^[1-4]$ ]]; then
        echo "Error: invalid fold_id '${fold}' — must be 1–4." >&2; exit 1
    fi
done

mkdir -p logs
CACHE_NVME="/tmp/sfx_frame_cache"
CONFIG="configs/supervised/resnet18_asymmetric.yaml"

# Sequential fold slots: stage(N) → train(N) → cleanup(N) → stage(N+1) → …
# Each fold's cache working set fills NVMe, so folds cannot overlap.
PREV_DEP=""
for fold in "${FOLDS[@]}"; do
    if [ -n "${PREV_DEP}" ]; then
        STAGE_JID=$(sbatch --parsable --dependency=afterany:"${PREV_DEP}" \
            --export=ALL,CONFIG="${CONFIG}" \
            scripts/stage_frame_cache.sh "${fold}")
    else
        STAGE_JID=$(sbatch --parsable \
            --export=ALL,CONFIG="${CONFIG}" \
            scripts/stage_frame_cache.sh "${fold}")
    fi
    echo "Stage fold ${fold}: ${STAGE_JID}"

    JID=$(sbatch --parsable --dependency=afterok:"${STAGE_JID}" \
        --export=ALL,CACHE_NVME="${CACHE_NVME}" \
        scripts/submit_asymmetric_lodo_fold.sh "${fold}")
    echo "Train fold ${fold}: ${JID} (after ${STAGE_JID})"

    CLEAN_JID=$(sbatch --parsable --dependency=afterany:"${JID}" \
        scripts/cleanup_ssl_stage.sh)
    echo "Cleanup fold ${fold}: ${CLEAN_JID} (after ${JID})"
    PREV_DEP="${CLEAN_JID}"
done

echo ""
echo "Watch queue:  squeue -u \$USER"
