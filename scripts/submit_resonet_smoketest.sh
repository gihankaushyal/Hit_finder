#!/bin/bash
#SBATCH --job-name=sfx-resonet-smoketest
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=logs/resonet-smoketest-%j.out
#SBATCH --error=logs/resonet-smoketest-%j.err

module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python -u -m src.training.train_asymmetric \
    --config configs/supervised/resnet18_resonet.yaml \
    --folds 4 \
    --tags supervised,resnet18,resonet-smoketest
