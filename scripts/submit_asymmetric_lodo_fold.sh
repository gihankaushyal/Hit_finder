#!/bin/bash
# submit_asymmetric_lodo_fold.sh — SLURM job: ResNet18 LODO training for a single fold.
#
# Reads preprocessed frames from the on-disk frame cache (read→assemble→PF8→GCN,
# precomputed once — see configs/supervised/resnet18_asymmetric.yaml's cache:
# block). Pass CACHE_NVME to read from a fold-staged NVMe tier (see
# scripts/stage_frame_cache.sh); omit it to read from the default NVMe path
# (empty unless staged beforehand).
#
# Run naming convention (--run-name-prefix, REQUIRED): <backbone>-asymmetric-v<N>
# e.g. resnet18-asymmetric-v2. See src/training/train_asymmetric.py's module
# docstring for the full rationale. If a checkpoint already exists for the
# resolved run name (checkpoints/<prefix>-fold<N>-seed<S>/best.pt),
# train_asymmetric.py exits with a collision error unless RESUME_FLAG below
# resolves to --resume-training or --override-training. This script runs
# non-interactively on the compute node (no tty) — the interactive "resume or
# override?" prompt lives in submit_asymmetric_lodo_all.sh, which runs directly
# in the user's terminal *before* calling sbatch, and passes its answer down
# via the RESUME_FLAG environment variable (--export=ALL,RESUME_FLAG=...).
#
# Usage:
#   sbatch scripts/submit_asymmetric_lodo_fold.sh <fold_id> <run_name_prefix>
#   bash   scripts/submit_asymmetric_lodo_fold.sh --help
#
# Arguments:
#   fold_id           LODO fold to train (1–4). Determines which detector is held out.
#   run_name_prefix   Required. <backbone>-asymmetric-v<N>, e.g. resnet18-asymmetric-v2.
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Examples:
#   sbatch scripts/submit_asymmetric_lodo_fold.sh 3 resnet18-asymmetric-v2
#
# Environment variables (set automatically by submit_asymmetric_lodo_all.sh):
#   CACHE_NVME    Path to this fold's NVMe-staged frame-cache tier
#                 (default /tmp/sfx_frame_cache). See scripts/stage_frame_cache.sh.
#   RESUME_FLAG   --resume-training (default), --override-training or
#                 --inference-only (evaluate the existing best.pt without
#                 training), resolved by the orchestrator's interactive prompt.
#                 A requeued job (SLURM_RESTART_COUNT > 0) always resumes, even if
#                 --override-training was requested, so it never deletes its own progress.
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
RUN_NAME_PREFIX="${2:?run_name_prefix required, e.g. resnet18-asymmetric-v2}"
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

python -u -m src.training.train_asymmetric \
    --config configs/supervised/resnet18_asymmetric.yaml \
    --run-name-prefix "${RUN_NAME_PREFIX}" \
    --folds "${FOLD}" \
    --cache-nvme "${CACHE_NVME:-/tmp/sfx_frame_cache}" \
    "${RESUME_FLAG}" \
    --tags supervised,resnet18,asymmetric-pipeline,lodo-cached
