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
# Run naming convention (--run-name-prefix, REQUIRED): <backbone>-asymmetric-v<N>
# e.g. resnet18-asymmetric-v2. See src/training/train_asymmetric.py's module
# docstring for the full rationale. Before submitting each fold, this script
# checks checkpoints/<prefix>-fold<N>-seed<S>/best.pt; if it already exists it
# prompts interactively — type "resume" or "override" — since this script runs
# directly in your terminal (unlike the SLURM batch jobs it submits, which have
# no tty and cannot prompt).
#
# Usage:
#   bash scripts/submit_asymmetric_lodo_all.sh --run-name-prefix <prefix> [OPTIONS]
#
# Options:
#   --run-name-prefix <prefix>   Required. <backbone>-asymmetric-v<N>, e.g. resnet18-asymmetric-v2.
#   --folds <1 2 3 4>            Space-separated fold IDs to run (default: 1 2 3 4).
#   -h, --help                   Show this help message and exit.
#
# Examples:
#   bash scripts/submit_asymmetric_lodo_all.sh --run-name-prefix resnet18-asymmetric-v2
#   bash scripts/submit_asymmetric_lodo_all.sh --run-name-prefix resnet18-asymmetric-v2 --folds 3 4
#
# See also:
#   scripts/submit_asymmetric_lodo_fold.sh   single-fold submission (reads CACHE_NVME/RESUME_FLAG env vars)
#   scripts/stage_frame_cache.sh             per-fold staging job (called automatically)
#   scripts/cleanup_ssl_stage.sh             cleanup job (called automatically)

set -euo pipefail

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \{0,1\}//; p }' "$0"
}

FOLDS=(1 2 3 4)
RUN_NAME_PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --run-name-prefix) shift; RUN_NAME_PREFIX="${1:?--run-name-prefix requires a value}"; shift ;;
        --folds) shift; FOLDS=(); while [[ $# -gt 0 && "$1" =~ ^[1-4]$ ]]; do FOLDS+=("$1"); shift; done ;;
        *) echo "Error: unknown option '$1'" >&2; echo "Run '$0 --help' for usage." >&2; exit 1 ;;
    esac
done

if [ -z "${RUN_NAME_PREFIX}" ]; then
    echo "Error: --run-name-prefix is required, e.g. resnet18-asymmetric-v2" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

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
# seed lives in base.yaml and may be overridden in the model config (load_config()
# deep-merges base.yaml with model values winning) — check the model file first.
SEED="$(grep -E '^\s*seed:' "${CONFIG}" configs/base.yaml 2>/dev/null | head -1 | awk '{print $2}')"

# Sequential fold slots: stage(N) → train(N) → cleanup(N) → stage(N+1) → …
# Each fold's cache working set fills NVMe, so folds cannot overlap.
PREV_DEP=""
for fold in "${FOLDS[@]}"; do
    # Interactive resume/override prompt — only possible here (a terminal);
    # the SLURM batch job this submits has no tty and cannot prompt.
    CKPT_PATH="checkpoints/${RUN_NAME_PREFIX}-fold${fold}-seed${SEED}/best.pt"
    RESUME_FLAG="--resume-training"
    if [[ -f "${CKPT_PATH}" ]]; then
        echo "Checkpoint already exists for fold ${fold}: ${CKPT_PATH}"
        while true; do
            read -r -p "Type 'resume' to continue training, or 'override' to discard and restart: " choice
            case "${choice}" in
                resume) RESUME_FLAG="--resume-training"; break ;;
                override) RESUME_FLAG="--override-training"; break ;;
                *) echo "Please type exactly 'resume' or 'override'." ;;
            esac
        done
    fi

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
        --export=ALL,CACHE_NVME="${CACHE_NVME}",RESUME_FLAG="${RESUME_FLAG}" \
        scripts/submit_asymmetric_lodo_fold.sh "${fold}" "${RUN_NAME_PREFIX}")
    echo "Train fold ${fold}: ${JID} (after ${STAGE_JID})"

    CLEAN_JID=$(sbatch --parsable --dependency=afterany:"${JID}" \
        scripts/cleanup_ssl_stage.sh)
    echo "Cleanup fold ${fold}: ${CLEAN_JID} (after ${JID})"
    PREV_DEP="${CLEAN_JID}"
done

echo ""
echo "Watch queue:  squeue -u \$USER"
