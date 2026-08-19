#!/bin/bash
#SBATCH --job-name=sfx-resonet-smoketest
#SBATCH --partition=general
#SBATCH --qos=grp_cxfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_resonet.yaml \
    --folds 4
