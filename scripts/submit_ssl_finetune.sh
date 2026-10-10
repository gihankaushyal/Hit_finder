#!/bin/bash
# submit_ssl_finetune.sh — SLURM job: MAE fine-tune or linear probe for one LODO fold.
#
# Loads the MAE pretrain checkpoint of the same fold and trains through the
# unchanged asymmetric pipeline, reading frames from the frame cache.
#
# Run naming convention (run_name_prefix, REQUIRED): <backbone>-mae-v<N>
# e.g. vits16-mae-v2. The training script expands it to
# <backbone>-mae-finetune-v<N> or, with --linear-probe, <backbone>-mae-probe-v<N>.
# See src/training/train_ssl_finetune.py's module docstring.
#
# The pretrain checkpoint is derived from pretrain_run_prefix:
#   checkpoints/<pretrain_run_prefix>-fold<N>-seed<S>/last.pt
#
# Usage:
#   sbatch scripts/submit_ssl_finetune.sh <fold_id> <run_name_prefix> <pretrain_run_prefix> [--linear-probe]
#   bash   scripts/submit_ssl_finetune.sh --help
#
# Arguments:
#   fold_id               LODO fold (1–4).
#   run_name_prefix       Required. <backbone>-mae-v<N>, e.g. vits16-mae-v2.
#   pretrain_run_prefix   Required. Prefix of the pretrain run to load, e.g. mae-vits16-v2.
#   --linear-probe        Optional. Freeze the encoder (linear probe).
#
# Examples:
#   sbatch scripts/submit_ssl_finetune.sh 1 vits16-mae-v2 mae-vits16-v2
#   sbatch scripts/submit_ssl_finetune.sh 1 vits16-mae-v2 mae-vits16-v2 --linear-probe
#
# Environment variables (set automatically by submit_ssl_finetune_all.sh):
#   CACHE_NVME    Path to this fold's NVMe-staged frame-cache tier
#                 (default /tmp/sfx_frame_cache). See scripts/stage_frame_cache.sh.
#   RESUME_FLAG   --resume-training (default) or --override-training. Decides what
#                 happens when best.pt already exists for the resolved run name.
#                 A requeued job (SLURM_RESTART_COUNT > 0) always resumes, even if
#                 --override-training was requested, so it never deletes its own progress.
#
# See also:
#   scripts/submit_ssl_finetune_all.sh   multi-fold submission with per-fold NVMe staging
#   src/training/train_ssl_finetune.py   training entry point
#SBATCH --job-name=sfx-ssl-finetune
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/ssl-finetune-%j.out
#SBATCH --error=logs/ssl-finetune-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FOLD="${1:?fold id required (1-4)}"
RUN_NAME_PREFIX="${2:?run_name_prefix required, e.g. vits16-mae-v2}"
PRETRAIN_RUN_PREFIX="${3:?pretrain_run_prefix required, e.g. mae-vits16-v2}"
shift 3
EXTRA=("$@")
RESUME_FLAG="${RESUME_FLAG:---resume-training}"

# A requeued job (SLURM_RESTART_COUNT > 0) re-runs with the same environment, so a
# job started with --override-training would delete the checkpoint it wrote before
# the node failure. Resume instead; the override already happened on the first start.
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && "${RESUME_FLAG}" == "--override-training" ]]; then
    echo "Requeued job (restart ${SLURM_RESTART_COUNT}): using --resume-training instead of --override-training." >&2
    RESUME_FLAG="--resume-training"
fi

CONFIG="configs/ssl/mae_finetune.yaml"
PRETRAIN_CONFIG="configs/ssl/mae_pretrain.yaml"
# The pretrain checkpoint directory is named with the seed the pretrain run used, so
# that seed comes from the pretrain config; the fine-tune/probe seed from this one.
# seed lives in base.yaml and may be overridden in the model config (load_config()
# deep-merges base.yaml with model values winning) — check the model file first.
# Only a top-level `seed:` counts (python reads cfg["seed"]); nested keys are ignored.
read_seed() {
    local cfg="$1" seed
    seed="$({ grep -hE '^seed:' "${cfg}" configs/base.yaml 2>/dev/null || true; } | head -1 | awk '{print $2}')" || seed=""
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Error: could not read a top-level integer 'seed:' from ${cfg} or configs/base.yaml." >&2
        echo "Run this script from the repository root." >&2
        exit 1
    fi
    echo "${seed}"
}
SEED="$(read_seed "${CONFIG}")"
PRETRAIN_SEED="$(read_seed "${PRETRAIN_CONFIG}")"
PRETRAIN_CKPT="checkpoints/${PRETRAIN_RUN_PREFIX}-fold${FOLD}-seed${PRETRAIN_SEED}/last.pt"

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "Error: pretrain checkpoint not found: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

python -m src.training.train_ssl_finetune \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --run-name-prefix "${RUN_NAME_PREFIX}" \
    --pretrain-checkpoint "${PRETRAIN_CKPT}" \
    --cache-nvme "${CACHE_NVME:-/tmp/sfx_frame_cache}" \
    "${RESUME_FLAG}" \
    "${EXTRA[@]}"
