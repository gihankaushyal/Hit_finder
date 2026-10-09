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
# Run naming convention (--run-name-prefix, REQUIRED): <backbone>-mae-v<N>
# e.g. vits16-mae-v2, expanded to <backbone>-mae-finetune-v<N> and
# <backbone>-mae-probe-v<N>. --pretrain-run-prefix (REQUIRED) names the pretrain
# run whose checkpoints are loaded, e.g. mae-vits16-v2; every requested fold
# must already have checkpoints/<pretrain-prefix>-fold<N>-seed<S>/last.pt or
# nothing is submitted. Before submitting each fold, this script checks the
# fine-tune and the probe best.pt; if one already exists it prompts
# interactively — type "resume" or "override" — since this script runs directly
# in your terminal (unlike the SLURM batch jobs it submits, which have no tty
# and cannot prompt).
#
# Usage:
#   bash scripts/submit_ssl_finetune_all.sh --run-name-prefix <prefix> --pretrain-run-prefix <prefix> [OPTIONS]
#
# Options:
#   --run-name-prefix <prefix>       Required. <backbone>-mae-v<N>, e.g. vits16-mae-v2.
#   --pretrain-run-prefix <prefix>   Required. mae-<backbone>-v<N>, e.g. mae-vits16-v2.
#   --folds <1 2 3 4>                Space-separated fold IDs to run (default: 1 2 3 4).
#   --no-probe                       Skip linear probe jobs (submit fine-tune only).
#   --no-finetune                    Skip full fine-tune jobs (submit probe only).
#   -h, --help                       Show this help message and exit.
#
# Examples:
#   bash scripts/submit_ssl_finetune_all.sh --run-name-prefix vits16-mae-v2 --pretrain-run-prefix mae-vits16-v2
#   bash scripts/submit_ssl_finetune_all.sh --run-name-prefix vits16-mae-v2 --pretrain-run-prefix mae-vits16-v2 --folds 2 3 4
#   bash scripts/submit_ssl_finetune_all.sh --run-name-prefix vits16-mae-v2 --pretrain-run-prefix mae-vits16-v2 --no-probe
#
# See also:
#   scripts/submit_ssl_finetune.sh   single-fold submission (reads CACHE_NVME/RESUME_FLAG env vars)
#   scripts/stage_frame_cache.sh     per-fold staging job (called automatically)
#   scripts/cleanup_ssl_stage.sh     cleanup job (called automatically)

set -euo pipefail

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \{0,1\}//; p }' "$0"
}

FOLDS=(1 2 3 4)
RUN_FINETUNE=true
RUN_PROBE=true
RUN_NAME_PREFIX=""
PRETRAIN_RUN_PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --run-name-prefix) shift; RUN_NAME_PREFIX="${1:?--run-name-prefix requires a value}"; shift ;;
        --pretrain-run-prefix) shift; PRETRAIN_RUN_PREFIX="${1:?--pretrain-run-prefix requires a value}"; shift ;;
        --folds) shift; FOLDS=(); while [[ $# -gt 0 && "$1" =~ ^[1-4]$ ]]; do FOLDS+=("$1"); shift; done ;;
        --no-probe) RUN_PROBE=false; shift ;;
        --no-finetune) RUN_FINETUNE=false; shift ;;
        *) echo "Error: unknown option '$1'" >&2; echo "Run '$0 --help' for usage." >&2; exit 1 ;;
    esac
done

if [ -z "${RUN_NAME_PREFIX}" ]; then
    echo "Error: --run-name-prefix is required, e.g. vits16-mae-v2" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi
if [ -z "${PRETRAIN_RUN_PREFIX}" ]; then
    echo "Error: --pretrain-run-prefix is required, e.g. mae-vits16-v2" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

# Same patterns as SSL_FINETUNE_PREFIX_RE / SSL_PRETRAIN_PREFIX_RE in
# src/training/run_naming.py — checked here too so a typo fails now, not hours
# later when the batch job starts.
if [[ ! "${RUN_NAME_PREFIX}" =~ ^[a-z0-9]+-mae-v[0-9]+$ ]]; then
    echo "Error: invalid --run-name-prefix '${RUN_NAME_PREFIX}' — expected <backbone>-mae-v<N>, e.g. vits16-mae-v2" >&2
    exit 1
fi
if [[ ! "${PRETRAIN_RUN_PREFIX}" =~ ^mae-[a-z0-9]+-v[0-9]+$ ]]; then
    echo "Error: invalid --pretrain-run-prefix '${PRETRAIN_RUN_PREFIX}' — expected mae-<backbone>-v<N>, e.g. mae-vits16-v2" >&2
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

if ! $RUN_FINETUNE && ! $RUN_PROBE; then
    echo "Error: --no-probe and --no-finetune leave nothing to run." >&2
    exit 1
fi

mkdir -p logs
CACHE_NVME="/tmp/sfx_frame_cache"
CONFIG="configs/ssl/mae_finetune.yaml"
# seed lives in base.yaml and may be overridden in the model config (load_config()
# deep-merges base.yaml with model values winning) — check the model file first.
SEED="$(grep -E '^\s*seed:' "${CONFIG}" configs/base.yaml 2>/dev/null | head -1 | awk '{print $2}')"

# vits16-mae-v2 → vits16-mae-finetune-v2 / vits16-mae-probe-v2, matching
# expand_finetune_prefix() in src/training/run_naming.py.
PREFIX_BASE="${RUN_NAME_PREFIX%-v*}"
PREFIX_VERSION="${RUN_NAME_PREFIX##*-v}"
FINETUNE_PREFIX="${PREFIX_BASE}-finetune-v${PREFIX_VERSION}"
PROBE_PREFIX="${PREFIX_BASE}-probe-v${PREFIX_VERSION}"

# Preflight: every requested fold needs its pretrain checkpoint. Checked for all
# folds before anything is submitted, so a gap never leaves a half-built chain.
MISSING=0
for fold in "${FOLDS[@]}"; do
    PRETRAIN_CKPT="checkpoints/${PRETRAIN_RUN_PREFIX}-fold${fold}-seed${SEED}/last.pt"
    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
        echo "Error: pretrain checkpoint not found for fold ${fold}: ${PRETRAIN_CKPT}" >&2
        MISSING=$(( MISSING + 1 ))
    fi
done
if [ "${MISSING}" -gt 0 ]; then
    echo "Nothing submitted: ${MISSING} fold(s) have no pretrain checkpoint under '${PRETRAIN_RUN_PREFIX}'." >&2
    exit 1
fi

# Interactive resume/override prompt — only possible here (a terminal); the
# SLURM batch jobs this submits have no tty and cannot prompt. Sets RESUME_FLAG.
resolve_resume_flag() {
    local label="$1" ckpt="$2" choice
    RESUME_FLAG="--resume-training"
    if [[ -f "${ckpt}" ]]; then
        echo "Checkpoint already exists for ${label}: ${ckpt}"
        while true; do
            read -r -p "Type 'resume' to continue training, or 'override' to discard and restart: " choice
            case "${choice}" in
                resume) RESUME_FLAG="--resume-training"; break ;;
                override) RESUME_FLAG="--override-training"; break ;;
                *) echo "Please type exactly 'resume' or 'override'." ;;
            esac
        done
    fi
}

# Sequential fold slots: stage(N) → [finetune N, probe N] → stage(N+1) → …
# Each fold's cache working set fills NVMe, so folds cannot overlap; the
# fine-tune and probe for one fold share a working set and do run in parallel.
PREV_DEP=""
for fold in "${FOLDS[@]}"; do
    FINETUNE_FLAG=""
    PROBE_FLAG=""
    if $RUN_FINETUNE; then
        resolve_resume_flag "fine-tune fold ${fold}" \
            "checkpoints/${FINETUNE_PREFIX}-fold${fold}-seed${SEED}/best.pt"
        FINETUNE_FLAG="${RESUME_FLAG}"
    fi
    if $RUN_PROBE; then
        resolve_resume_flag "probe fold ${fold}" \
            "checkpoints/${PROBE_PREFIX}-fold${fold}-seed${SEED}/best.pt"
        PROBE_FLAG="${RESUME_FLAG}"
    fi

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
            --export=ALL,CACHE_NVME="${CACHE_NVME}",RESUME_FLAG="${FINETUNE_FLAG}" \
            scripts/submit_ssl_finetune.sh "${fold}" "${RUN_NAME_PREFIX}" "${PRETRAIN_RUN_PREFIX}")
        FOLD_JIDS+=("${JID}")
        echo "Fine-tune fold ${fold}: ${JID} (after ${STAGE_JID})"
    fi
    if $RUN_PROBE; then
        JID=$(sbatch --parsable --dependency=afterok:"${STAGE_JID}" \
            --export=ALL,CACHE_NVME="${CACHE_NVME}",RESUME_FLAG="${PROBE_FLAG}" \
            scripts/submit_ssl_finetune.sh "${fold}" "${RUN_NAME_PREFIX}" "${PRETRAIN_RUN_PREFIX}" --linear-probe)
        FOLD_JIDS+=("${JID}")
        echo "Probe     fold ${fold}: ${JID} (after ${STAGE_JID})"
    fi

    DEPS=$(IFS=:; echo "${FOLD_JIDS[*]}")
    CLEAN_JID=$(sbatch --parsable --dependency=afterany:"${DEPS}" \
        scripts/cleanup_ssl_stage.sh)
    echo "Cleanup fold ${fold}:   ${CLEAN_JID} (after ${DEPS})"
    PREV_DEP="${CLEAN_JID}"
done

echo ""
echo "Watch queue:  squeue -u \$USER"
