#!/bin/bash
#SBATCH --job-name=sfx-resonet-smoketest
#SBATCH -p general
#SBATCH -q grp_cxfl
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_resonet.yaml \
    --folds 4
