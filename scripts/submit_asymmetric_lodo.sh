#!/bin/bash
#SBATCH --job-name=sfx-lodo-full
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lodo-full-%j.out
#SBATCH --error=logs/lodo-full-%j.err

module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_asymmetric.yaml
