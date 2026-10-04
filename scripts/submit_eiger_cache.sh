#!/bin/bash
# submit_eiger_cache.sh — SLURM job: build the frame cache for Eiger4M only.
#
# No Eiger4M cache has ever been built. Sized by analogy to the two prior
# detector builds rather than AGIPD's exact settings: AGIPD's own build (job
# 64056541, 32 workers/64G) peaked at MaxRSS=67GB — right at the ReqMem
# ceiling, barely surviving. Eiger4M's assembled canvas (~1687x1687, 2.85M px
# per notebooks/detector_assembly_crosscheck.ipynb) is ~1.8x larger than
# AGIPD's (~1273x1273, 1.6M px), so AGIPD's settings would likely repeat the
# near-OOM margin. This halves worker count (as the Jungfrau fix did) and
# raises memory to 128G — enough headroom for the larger canvas without the
# 256G Jungfrau needed for its still-larger 4.72M px canvas.
#
# Usage:
#   sbatch scripts/submit_eiger_cache.sh
#
# See also:
#   scripts/build_frame_cache.py   the builder this script invokes
#   scripts/submit_jungfrau_cache.sh   same pattern, sized for Jungfrau's canvas
#SBATCH --job-name=sfx-eiger-cache
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/eiger-cache-%j.out
#SBATCH --error=logs/eiger-cache-%j.err

module load mamba/latest

# `source activate sfx-hitfinder` has been observed to silently fail to switch
# the interpreter in non-interactive SLURM shells (falls back to base python,
# which lacks scipy/reborn/etc.) — invoke the env's python directly instead.
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -u -m scripts.build_frame_cache \
    --config configs/ssl/mae_pretrain.yaml \
    --detectors Eiger4M \
    --workers 16
