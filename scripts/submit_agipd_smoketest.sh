#!/bin/bash
#SBATCH --job-name=sfx-agipd-smoketest
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=logs/agipd-smoketest-%j.out
#SBATCH --error=logs/agipd-smoketest-%j.err

module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python -u scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_resonet.yaml \
    --folds 1 \
    --tags supervised,resnet18,agipd-smoketest
