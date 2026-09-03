#!/bin/bash
# Submit all 4 LODO folds as independent parallel SLURM jobs.
# Each fold gets its own H100 and runs ~10h. All 4 finish in parallel.
#
# Usage:
#   bash scripts/submit_lodo_parallel.sh
#   bash scripts/submit_lodo_parallel.sh --folds 1 3   # subset of folds

set -euo pipefail

CONFIG="configs/supervised/resnet18_asymmetric.yaml"
FOLDS="${*:-2 }"

for FOLD in $FOLDS; do
    JOBID=$(sbatch --parsable \
        --job-name="sfx-lodo-fold${FOLD}" \
        -p general \
        -q grp_cxfel \
        --gres=gpu:h100:1 \
        --nodelist=scg020 \
        -N 1 \
        -c 8 \
        --mem=128G \
        --time=24:00:00 \
        --output="logs/lodo-fold${FOLD}-%j.out" \
        --error="logs/lodo-fold${FOLD}-%j.err" \
        --wrap="module load mamba/latest && source activate sfx-hitfinder && source .secrets/wandb.env && python -u -m src.training.train_asymmetric --config ${CONFIG} --folds ${FOLD} --tags supervised,resnet18,asymmetric-pipeline,lodo-parallel --resume-training"
    )
    echo "Fold ${FOLD} → job ${JOBID}  (logs/lodo-fold${FOLD}-${JOBID}.{out,err})"
done
