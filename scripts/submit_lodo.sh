#!/bin/bash
# Submit one SLURM job per LODO fold.
# --output/--error are passed on the sbatch command line so the shell expands
# $FOLD before SLURM sees the value (unlike #SBATCH headers, which are parsed
# before the shell runs and cannot expand environment variables).
#
# Usage: bash scripts/submit_lodo.sh [fold1 fold2 ...]
#        Default: all four folds (1 2 3 4)

set -euo pipefail

FOLDS="${@:-1 2 3 4}"
mkdir -p logs

for FOLD in $FOLDS; do
    sbatch \
        --export=ALL,FOLD=${FOLD} \
        --output="logs/lodo_fold${FOLD}_%j.log" \
        --error="logs/lodo_fold${FOLD}_%j.err" \
        scripts/submit_lodo_fold.sh
    echo "Submitted fold ${FOLD}"
done
