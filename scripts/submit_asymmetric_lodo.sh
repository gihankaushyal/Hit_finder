#!/bin/bash
#SBATCH --job-name=sfx-lodo-full
#SBATCH -p general
#SBATCH -q grp_cxfl
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_asymmetric.yaml
