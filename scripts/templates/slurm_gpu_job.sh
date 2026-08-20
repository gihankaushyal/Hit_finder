#!/bin/bash
# =============================================================================
# ASU Sol HPC — GPU Job Template (grp_cxfel, scg020, H100)
# DO NOT DELETE OR MODIFY — protected reference template
#
# Verified working: 2026-08-18 (job 61743479, resonet smoketest)
#
# Key lessons learned:
#   - Use `source activate` NOT `conda activate` (conda init not available in jobs)
#   - GRES must name the GPU type explicitly: gpu:h100:1 (not gpu:1)
#   - QOS is grp_cxfel (with 'e') — not grp_cxfl
#   - Use -p / -q short flags to match Sol conventions
#   - Log names: use a descriptive prefix + -%j so job IDs are traceable
# =============================================================================

#SBATCH --job-name=sfx-<jobname>
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/<jobname>-%j.out
#SBATCH --error=logs/<jobname>-%j.err

# --- Environment ---
module load mamba/latest
source activate sfx-hitfinder      # NOTE: source activate, not conda activate

# --- Credentials (never inline keys; never commit .secrets/) ---
source .secrets/wandb.env

# --- Job ---
mkdir -p logs
python scripts/<your_script>.py --config configs/<your_config>.yaml
