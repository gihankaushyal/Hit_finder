#!/bin/bash
# submit_jungfrau_cache.sh — SLURM job: build the frame cache for JUNGFRAU_4M only.
#
# Re-run of the Jungfrau cache build after job 64071392 TIMEOUT'd with 7 OOM
# kills at --mem=64G/--workers=32 (only 3/10 files completed in 24h). sacct
# showed MaxRSS sitting at ~64GB for both the Jungfrau job and a same-frame-
# count comparison job that succeeded — Jungfrau's larger assembled canvas
# (4.72M px, float64 in PADAssembler.assemble_data) pushed the aggregate
# 32-worker memory demand over the edge. This run halves worker count (lowers
# aggregate concurrent memory) and quadruples the memory request.
#
# Usage:
#   sbatch scripts/submit_jungfrau_cache.sh
#
# See also:
#   scripts/build_frame_cache.py   the builder this script invokes
#SBATCH --job-name=sfx-jungfrau-cache
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/jungfrau-cache-%j.out
#SBATCH --error=logs/jungfrau-cache-%j.err

module load mamba/latest

# `source activate sfx-hitfinder` has been observed to silently fail to switch
# the interpreter in non-interactive SLURM shells (falls back to base python,
# which lacks scipy/reborn/etc.) — invoke the env's python directly instead.
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -u -m scripts.build_frame_cache \
    --config configs/ssl/mae_pretrain.yaml \
    --detectors JUNGFRAU_4M \
    --workers 16
