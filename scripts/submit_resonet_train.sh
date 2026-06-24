#!/bin/bash
#SBATCH --job-name=sfx-resonet
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source /packages/apps/mamba/2.6.2/etc/profile.d/conda.sh
conda activate sfx-hitfinder

/home/gketawal/.conda/envs/sfx-hitfinder/bin/python \
    scripts/train_resonet_cxi.py \
    --config configs/supervised/resnet18_resonet.yaml
