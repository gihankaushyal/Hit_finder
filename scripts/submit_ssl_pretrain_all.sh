#!/bin/bash
# submit_ssl_pretrain_all.sh — Submit SSL MAE pretraining with shared NVMe staging.
#
# Stages all four CXI detector directories to /tmp/sfx_stage_shared once, then
# launches the requested fold training jobs in parallel. All jobs are pinned to
# scg020 so they share the same /tmp NVMe filesystem (~336 GB). A cleanup job
# removes the shared stage directory after all fold jobs finish (pass or fail).
#
# Job chain (SLURM dependencies):
#   stage_ssl_data.sh  →  [fold jobs in parallel]  →  cleanup_ssl_stage.sh
#
# Usage:
#   bash scripts/submit_ssl_pretrain_all.sh [OPTIONS] <fold_id> [fold_id ...]
#
# Arguments:
#   fold_id   One or more fold IDs (1–4) to train. At least one required.
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Examples:
#   bash scripts/submit_ssl_pretrain_all.sh 3 4       # rerun folds 3 and 4
#   bash scripts/submit_ssl_pretrain_all.sh 1 2 3 4   # full 4-fold run
#
# See also:
#   scripts/submit_ssl_pretrain.sh   single-fold submission (private staging)
#   scripts/stage_ssl_data.sh        staging job (called automatically)
#   scripts/cleanup_ssl_stage.sh     cleanup job (called automatically)

set -euo pipefail

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FOLDS=("$@")
if [ "${#FOLDS[@]}" -eq 0 ]; then
    echo "Error: at least one fold_id required." >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

# Validate all fold IDs before submitting anything.
for fold in "${FOLDS[@]}"; do
    if [[ ! "${fold}" =~ ^[1-4]$ ]]; then
        echo "Error: invalid fold_id '${fold}' — must be 1, 2, 3, or 4." >&2
        exit 1
    fi
done

STAGE="/tmp/sfx_stage_shared"

# 1. Submit staging job.
STAGE_JID=$(sbatch --parsable scripts/stage_ssl_data.sh)
echo "Staging job:  ${STAGE_JID}"

# 2. Submit fold training jobs, each dependent on staging succeeding.
FOLD_JIDS=()
for fold in "${FOLDS[@]}"; do
    JID=$(SHARED_STAGE="${STAGE}" sbatch --parsable \
        --dependency=afterok:"${STAGE_JID}" \
        --export=ALL,SHARED_STAGE="${STAGE}" \
        scripts/submit_ssl_pretrain.sh "${fold}")
    FOLD_JIDS+=("${JID}")
    echo "Fold ${fold} job: ${JID} (depends on ${STAGE_JID})"
done

# 3. Submit cleanup job, runs after all folds finish (pass or fail).
DEPS=$(IFS=:; echo "${FOLD_JIDS[*]}")
CLEANUP_JID=$(sbatch --parsable \
    --dependency=afterany:"${DEPS}" \
    scripts/cleanup_ssl_stage.sh)
echo "Cleanup job:  ${CLEANUP_JID} (depends on ${DEPS})"

echo ""
echo "Watch queue:  squeue -u \$USER"
