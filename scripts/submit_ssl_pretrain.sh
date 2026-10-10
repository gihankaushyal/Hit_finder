#!/bin/bash
# submit_ssl_pretrain.sh — SLURM job: MAE SSL pretraining for a single LODO fold.
#
# Reads preprocessed frames from the on-disk frame cache (read→assemble→PF8→GCN,
# precomputed once — see configs/ssl/mae_pretrain.yaml's cache: block). Pass
# CACHE_NVME to read from a fold-staged NVMe tier (see scripts/stage_frame_cache.sh);
# omit it to read from the default NVMe path (empty unless staged beforehand).
#
# Run naming convention (run_name_prefix, REQUIRED): mae-<backbone>-v<N>
# e.g. mae-vits16-v2. See src/training/train_ssl_pretrain.py's module docstring.
# Checkpoints land in checkpoints/<prefix>-fold<N>-seed<S>/.
#
# Usage:
#   sbatch scripts/submit_ssl_pretrain.sh <fold_id> <run_name_prefix> [epochs]
#   bash   scripts/submit_ssl_pretrain.sh --help
#
# Arguments:
#   fold_id           LODO fold to train (1–4). Determines which detector is held out.
#   run_name_prefix   Required. mae-<backbone>-v<N>, e.g. mae-vits16-v2.
#   epochs            Optional epoch count override. Omit for the full 400-epoch run.
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Examples:
#   sbatch scripts/submit_ssl_pretrain.sh 3 mae-vits16-v2          # full 400-epoch run, fold 3
#   sbatch scripts/submit_ssl_pretrain.sh 1 mae-vits16-v2 100      # smoke run, 100 epochs
#
# Environment variables (set automatically by submit_ssl_pretrain_all.sh):
#   CACHE_NVME    Path to this fold's NVMe-staged frame-cache tier
#                 (default /tmp/sfx_frame_cache). See scripts/stage_frame_cache.sh.
#   RESUME_FLAG   --resume-training (default) or --override-training. Decides what
#                 happens when last.pt already exists for the resolved run name.
#                 The default lets a job resubmitted after a timeout continue.
#                 A requeued job (SLURM_RESTART_COUNT > 0) always resumes, even if
#                 --override-training was requested, so it never deletes its own progress.
#
# See also:
#   scripts/submit_ssl_pretrain_all.sh   multi-fold submission with per-fold NVMe staging
#   scripts/stage_frame_cache.sh         per-fold staging job (called automatically)
#   configs/ssl/mae_pretrain.yaml        pretraining hyperparameters
#   src/training/train_ssl_pretrain.py   training entry point
#SBATCH --job-name=sfx-ssl-pretrain
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/ssl-pretrain-%j.out
#SBATCH --error=logs/ssl-pretrain-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FOLD="${1:?fold id required (1-4)}"
RUN_NAME_PREFIX="${2:?run_name_prefix required, e.g. mae-vits16-v2}"
EPOCHS="${3:-}"   # optional; omit for full 400-epoch run
RESUME_FLAG="${RESUME_FLAG:---resume-training}"

# A requeued job (SLURM_RESTART_COUNT > 0) re-runs with the same environment, so a
# job started with --override-training would delete the checkpoint it wrote before
# the node failure. Resume instead; the override already happened on the first start.
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && "${RESUME_FLAG}" == "--override-training" ]]; then
    echo "Requeued job (restart ${SLURM_RESTART_COUNT}): using --resume-training instead of --override-training." >&2
    RESUME_FLAG="--resume-training"
fi

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

EPOCHS_ARG=""
if [ -n "${EPOCHS}" ]; then
    EPOCHS_ARG="--epochs ${EPOCHS}"
fi

/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -u -m src.training.train_ssl_pretrain \
    --config configs/ssl/mae_pretrain.yaml \
    --fold "${FOLD}" \
    --run-name-prefix "${RUN_NAME_PREFIX}" \
    "${RESUME_FLAG}" \
    --cache-nvme "${CACHE_NVME:-/tmp/sfx_frame_cache}" \
    ${EPOCHS_ARG}
