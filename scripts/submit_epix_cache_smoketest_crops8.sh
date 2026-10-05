#!/bin/bash
#SBATCH --job-name=sfx-epix-cache-smoketest-crops8
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=logs/epix-cache-smoketest-crops8-%j.out
#SBATCH --error=logs/epix-cache-smoketest-crops8-%j.err

module load mamba/latest

source .secrets/wandb.env

# `source activate sfx-hitfinder` has been observed to silently fail to switch
# the interpreter in non-interactive SLURM shells (falls back to base python,
# which lacks torch/scipy/reborn/etc.) — invoke the env's python directly instead.
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -u -m src.training.train_asymmetric \
    --config configs/supervised/resnet18_epix_cache_smoketest_crops8.yaml \
    --intra \
    --tags supervised,resnet18,epix-cache-smoketest,crops8
