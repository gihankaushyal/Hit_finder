#!/bin/bash
# submit_ssl_finetune_all.sh — Orchestrate Stage B fine-tuning + linear probe.
#
# Stages each fold's frame-cache working set to /tmp/sfx_frame_cache on NVMe,
# runs that fold's full fine-tune and linear probe jobs in parallel from NVMe,
# then cleans up before moving to the next fold. Folds run sequentially because
# each fold's cache working set fills the NVMe tier — they cannot overlap.
# All jobs are pinned to scg020 so they share the same /tmp NVMe filesystem.
#
# Job chain (SLURM dependencies):
#   stage(1) → [finetune 1, probe 1] → cleanup(1)
#                                         ↓ afterany
#                                      stage(2) → [finetune 2, probe 2] → cleanup(2) → …
#
# Usage:
#   bash scripts/submit_ssl_finetune_all.sh [OPTIONS]
#
# Options:
#   --folds <1 2 3 4>   Space-separated fold IDs to run (default: 1 2 3 4).
#   --no-probe          Skip linear probe jobs (submit fine-tune only).
#   --no-finetune       Skip full fine-tune jobs (submit probe only).
#   -h, --help          Show this help message and exit.
#
# Examples:
#   bash scripts/submit_ssl_finetune_all.sh                  # all 8 jobs
#   bash scripts/submit_ssl_finetune_all.sh --folds 2 3 4   # skip fold 1 (already done)
#   bash scripts/submit_ssl_finetune_all.sh --no-probe       # fine-tune only
#
# See also:
#   scripts/submit_ssl_finetune.sh   single-fold submission (reads CACHE_NVME env var)
#   scripts/stage_frame_cache.sh     per-fold staging job (called automatically)
#   scripts/cleanup_ssl_stage.sh     cleanup job (called automatically)

set -euo pipefail

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \{0,1\}//; p }' "$0"
}

FOLDS=(1 2 3 4)
RUN_FINETUNE=true
RUN_PROBE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --folds) shift; FOLDS=(); while [[ $# -gt 0 && "$1" =~ ^[1-4]$ ]]; do FOLDS+=("$1"); shift; done ;;
        --no-probe) RUN_PROBE=false; shift ;;
        --no-finetune) RUN_FINETUNE=false; shift ;;
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

# Sequential fold slots: stage(N) → [finetune N, probe N] → stage(N+1) → …
# Each fold's cache working set fills NVMe, so folds cannot overlap; the
# fine-tune and probe for one fold share a working set and do run in parallel.
PREV_DEP=""
for fold in "${FOLDS[@]}"; do
    if [ -n "${PREV_DEP}" ]; then
        STAGE_JID=$(sbatch --parsable --dependency=afterany:"${PREV_DEP}" \
            scripts/stage_frame_cache.sh "${fold}")
    else
        STAGE_JID=$(sbatch --parsable scripts/stage_frame_cache.sh "${fold}")
    fi
    echo "Stage fold ${fold}:     ${STAGE_JID}"

    FOLD_JIDS=()
    if $RUN_FINETUNE; then
        JID=$(sbatch --parsable --dependency=afterok:"${STAGE_JID}" \
            --export=ALL,CACHE_NVME="${CACHE_NVME}" \
            scripts/submit_ssl_finetune.sh "${fold}")
        FOLD_JIDS+=("${JID}")
        echo "Fine-tune fold ${fold}: ${JID} (after ${STAGE_JID})"
    fi
    if $RUN_PROBE; then
        JID=$(sbatch --parsable --dependency=afterok:"${STAGE_JID}" \
            --export=ALL,CACHE_NVME="${CACHE_NVME}" \
            scripts/submit_ssl_finetune.sh "${fold}" --linear-probe)
        FOLD_JIDS+=("${JID}")
        echo "Probe     fold ${fold}: ${JID} (after ${STAGE_JID})"
    fi

    if [ "${#FOLD_JIDS[@]}" -eq 0 ]; then
        echo "Error: --no-probe and --no-finetune leave nothing to run." >&2
        exit 1
    fi

    DEPS=$(IFS=:; echo "${FOLD_JIDS[*]}")
    CLEAN_JID=$(sbatch --parsable --dependency=afterany:"${DEPS}" \
        scripts/cleanup_ssl_stage.sh)
    echo "Cleanup fold ${fold}:   ${CLEAN_JID} (after ${DEPS})"
    PREV_DEP="${CLEAN_JID}"
done

echo ""
echo "Watch queue:  squeue -u \$USER"
