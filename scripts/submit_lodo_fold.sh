#!/bin/bash
# Usage: FOLD=1 sbatch scripts/submit_lodo_fold.sh
#        or via submit_lodo.sh which passes --output/--error on the sbatch
#        command line (shell expands $FOLD before SLURM sees the directive).
#
# Do NOT put --output/--error in #SBATCH headers here — SLURM parses those
# before the shell runs, so $FOLD would be the literal string "${FOLD}".
#SBATCH --job-name=lodo_f${FOLD}
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH --nodelist=scg020
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=14:00:00

mkdir -p logs

/home/gketawal/.conda/envs/sfx-hitfinder/bin/python \
    scripts/train_lodo.py \
    --config configs/supervised/resnet18_lodo.yaml \
    --folds ${FOLD}
