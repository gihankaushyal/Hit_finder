#!/bin/bash
#SBATCH --job-name=sfx-agipd-lodo
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/agipd-lodo-%j.out
#SBATCH --error=logs/agipd-lodo-%j.err

module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python -u scripts/train_asymmetric.py \
    --config configs/supervised/resnet18_asymmetric.yaml \
    --folds 1 \
    --tags supervised,resnet18,asymmetric-pipeline,agipd-lodo-rerun
