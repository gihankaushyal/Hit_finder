#!/bin/bash
# submit_asymmetric_lodo_fold.sh — SLURM job: ResNet18 LODO training for a single fold.
#
# Reads preprocessed frames from the on-disk frame cache (read→assemble→PF8→GCN,
# precomputed once — see configs/supervised/resnet18_asymmetric.yaml's cache:
# block). Pass CACHE_NVME to read from a fold-staged NVMe tier (see
# scripts/stage_frame_cache.sh); omit it to read from the default NVMe path
# (empty unless staged beforehand).
#
# Usage:
#   sbatch scripts/submit_asymmetric_lodo_fold.sh <fold_id>
#   bash   scripts/submit_asymmetric_lodo_fold.sh --help
#
# Arguments:
#   fold_id   LODO fold to train (1–4). Determines which detector is held out.
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Examples:
#   sbatch scripts/submit_asymmetric_lodo_fold.sh 3   # train fold 3
#
# Environment variables (set automatically by submit_asymmetric_lodo_all.sh):
#   CACHE_NVME   Path to this fold's NVMe-staged frame-cache tier
#                (default /tmp/sfx_frame_cache). See scripts/stage_frame_cache.sh.
#
# See also:
#   scripts/submit_asymmetric_lodo_all.sh   multi-fold submission with per-fold NVMe staging
#   scripts/stage_frame_cache.sh            per-fold staging job (called automatically)
#   configs/supervised/resnet18_asymmetric.yaml   training hyperparameters
#   src/training/train_asymmetric.py        training entry point
#SBATCH --job-name=sfx-lodo-fold
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lodo-cached-fold-%j.out
#SBATCH --error=logs/lodo-cached-fold-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FOLD="${1:?fold id required (1-4)}"

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

python -u -m src.training.train_asymmetric \
    --config configs/supervised/resnet18_asymmetric.yaml \
    --folds "${FOLD}" \
    --cache-nvme "${CACHE_NVME:-/tmp/sfx_frame_cache}" \
    --resume-training \
    --tags supervised,resnet18,asymmetric-pipeline,lodo-cached
