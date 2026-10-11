#!/bin/bash
#SBATCH --job-name=sfx-epix-smoketest
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=logs/epix-smoketest-%j.out
#SBATCH --error=logs/epix-smoketest-%j.err

# DEPRECATED (pipeline v1): this script writes the legacy run prefix
# 'resnet18smoke-asymmetric-v1', which predates the frame cache and the run-naming standard.
# It refuses to run so a v1 run cannot be started by accident.
echo "DEPRECATED: $(basename "$0") is a pipeline-v1 script (run prefix 'resnet18smoke-asymmetric-v1') and no longer runs." >&2
echo "Use instead:" >&2
echo '  sbatch scripts/submit_epix_cache_smoketest.sh   (frame-cache smoke test, v2 prefix)' >&2
echo "See src/training/run_naming.py for the run-name convention." >&2
exit 1


module load mamba/latest
source activate sfx-hitfinder

source .secrets/wandb.env

python -u -m src.training.train_asymmetric \
    --config configs/supervised/resnet18_epix_smoketest.yaml \
    --run-name-prefix resnet18smoke-asymmetric-v1 \
    --intra \
    --tags supervised,resnet18,epix-smoketest
