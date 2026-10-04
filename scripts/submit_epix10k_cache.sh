#!/bin/bash
# submit_epix10k_cache.sh — SLURM job: build the frame cache for ePix10k only.
#
# No ePix10k cache has ever been built. Sized by analogy to the Eiger4M build
# (job 64612695, 16 workers/128G, MaxRSS~113G) rather than AGIPD's settings:
# ePix10k's assembled canvas (~1667x1667, 2.78M px per
# notebooks/detector_assembly_crosscheck.ipynb) is almost identical in size to
# Eiger4M's (~1687x1687, 2.85M px) and its raw frame shape (4000, 5632, 384)
# matches Eiger4M's exactly, so the same worker/memory settings should carry
# the same safety margin.
#
# Usage:
#   sbatch scripts/submit_epix10k_cache.sh
#
# See also:
#   scripts/build_frame_cache.py       the builder this script invokes
#   scripts/submit_eiger_cache.sh      same pattern, sized for Eiger4M's canvas
#   scripts/submit_jungfrau_cache.sh   same pattern, sized for Jungfrau's canvas
#SBATCH --job-name=sfx-epix10k-cache
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/epix10k-cache-%j.out
#SBATCH --error=logs/epix10k-cache-%j.err

module load mamba/latest

# `source activate sfx-hitfinder` has been observed to silently fail to switch
# the interpreter in non-interactive SLURM shells (falls back to base python,
# which lacks scipy/reborn/etc.) — invoke the env's python directly instead.
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -u -m scripts.build_frame_cache \
    --config configs/ssl/mae_pretrain.yaml \
    --detectors ePix10k \
    --workers 16
