#!/bin/bash
#SBATCH --job-name=sfx-ssl-pretrain
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load mamba/latest
conda activate sfx-hitfinder

python src/training/train_ssl_pretrain.py --config configs/ssl/mae_pretrain.yaml
