#!/bin/bash
#SBATCH --job-name=lodo_f${FOLD}
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH --nodelist=scg020
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/lodo_fold${FOLD}_%j.log
#SBATCH --error=logs/lodo_fold${FOLD}_%j.err

mkdir -p logs

/home/gketawal/.conda/envs/sfx-hitfinder/bin/python \
    scripts/train_lodo.py \
    --config configs/supervised/resnet18_lodo.yaml \
    --folds ${FOLD}
