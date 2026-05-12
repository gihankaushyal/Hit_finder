#!/bin/bash
#SBATCH --job-name=sfx-supervised
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

python src/training/train_supervised.py --config configs/supervised/resnet18.yaml
