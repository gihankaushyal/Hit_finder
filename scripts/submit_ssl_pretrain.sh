#!/bin/bash
# Usage:
#   Smoke run (~100 epochs): sbatch scripts/submit_ssl_pretrain.sh <fold_id> 100
#   Full run  (400 epochs):  sbatch scripts/submit_ssl_pretrain.sh <fold_id>
#SBATCH --job-name=sfx-ssl-pretrain
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ssl-pretrain-%j.out
#SBATCH --error=logs/ssl-pretrain-%j.err

set -euo pipefail
FOLD="${1:?fold id required (1-4)}"
EPOCHS="${2:-}"   # optional; omit for full 400-epoch run

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

EPOCHS_ARG=""
if [ -n "${EPOCHS}" ]; then
    EPOCHS_ARG="--epochs ${EPOCHS}"
fi

python -u -m src.training.train_ssl_pretrain \
    --config configs/ssl/mae_pretrain.yaml \
    --fold "${FOLD}" \
    --resume \
    ${EPOCHS_ARG}
