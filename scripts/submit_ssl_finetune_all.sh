#!/bin/bash
# submit_ssl_finetune_all.sh — Orchestrate Stage B fine-tuning + linear probe.
#
# Stages all four CXI detector directories to /tmp/sfx_stage_shared once, then
# launches 4 full fine-tune and 4 linear probe jobs in parallel from NVMe.
# All jobs are pinned to scg020 so they share the same /tmp NVMe filesystem.
# A cleanup job removes the shared stage directory after all jobs finish.
#
# Job chain (SLURM dependencies):
#   stage_ssl_data.sh  →  [8 finetune/probe jobs in parallel]  →  cleanup_ssl_stage.sh
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
#   scripts/submit_ssl_finetune.sh   single-fold submission (reads SHARED_STAGE env var)
#   scripts/stage_ssl_data.sh        staging job (called automatically)
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
STAGE="/tmp/sfx_stage_shared"

# 1. Submit staging job.
STAGE_JID=$(sbatch --parsable scripts/stage_ssl_data.sh)
echo "Staging job:  ${STAGE_JID}"

# 2. Submit fine-tune and probe jobs, each dependent on staging.
ALL_JIDS=()

for fold in "${FOLDS[@]}"; do
    if $RUN_FINETUNE; then
        JID=$(SHARED_STAGE="${STAGE}" sbatch --parsable \
            --dependency=afterok:"${STAGE_JID}" \
            --export=ALL,SHARED_STAGE="${STAGE}" \
            scripts/submit_ssl_finetune.sh "${fold}")
        ALL_JIDS+=("${JID}")
        echo "Fine-tune fold ${fold}: ${JID} (depends on ${STAGE_JID})"
    fi

    if $RUN_PROBE; then
        JID=$(SHARED_STAGE="${STAGE}" sbatch --parsable \
            --dependency=afterok:"${STAGE_JID}" \
            --export=ALL,SHARED_STAGE="${STAGE}" \
            scripts/submit_ssl_finetune.sh "${fold}" --linear-probe)
        ALL_JIDS+=("${JID}")
        echo "Probe    fold ${fold}: ${JID} (depends on ${STAGE_JID})"
    fi
done

# 3. Cleanup when all jobs finish (pass or fail).
DEPS=$(IFS=:; echo "${ALL_JIDS[*]}")
CLEANUP_JID=$(sbatch --parsable \
    --dependency=afterany:"${DEPS}" \
    scripts/cleanup_ssl_stage.sh)
echo "Cleanup job:  ${CLEANUP_JID} (depends on ${DEPS})"

echo ""
echo "Watch queue:  squeue -u \$USER"
