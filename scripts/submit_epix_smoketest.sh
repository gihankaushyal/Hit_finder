#!/bin/bash
#SBATCH --job-name=sfx-epix-smoketest
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=logs/epix-smoketest-%j.out
#SBATCH --error=logs/epix-smoketest-%j.err

module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python -u scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_epix_smoketest.yaml \
    --intra \
    --tags supervised,resnet18,epix-smoketest
