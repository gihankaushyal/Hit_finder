#!/bin/bash
# submit_ssl_pretrain.sh — SLURM job: MAE SSL pretraining for a single LODO fold.
#
# Reads preprocessed frames from the on-disk frame cache (read→assemble→PF8→GCN,
# precomputed once — see configs/ssl/mae_pretrain.yaml's cache: block). Pass
# CACHE_NVME to read from a fold-staged NVMe tier (see scripts/stage_frame_cache.sh);
# omit it to read from the default NVMe path (empty unless staged beforehand).
#
# Usage:
#   sbatch scripts/submit_ssl_pretrain.sh <fold_id> [epochs]
#   bash   scripts/submit_ssl_pretrain.sh --help
#
# Arguments:
#   fold_id   LODO fold to train (1–4). Determines which detector is held out.
#   epochs    Optional epoch count override. Omit for the full 400-epoch run.
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Examples:
#   sbatch scripts/submit_ssl_pretrain.sh 3          # full 400-epoch run, fold 3
#   sbatch scripts/submit_ssl_pretrain.sh 1 100      # smoke run, 100 epochs
#
# Environment variables (set automatically by submit_ssl_pretrain_all.sh):
#   CACHE_NVME   Path to this fold's NVMe-staged frame-cache tier
#                (default /tmp/sfx_frame_cache). See scripts/stage_frame_cache.sh.
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
EPOCHS="${2:-}"   # optional; omit for full 400-epoch run

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
    --resume \
    --cache-nvme "${CACHE_NVME:-/tmp/sfx_frame_cache}" \
    ${EPOCHS_ARG}
