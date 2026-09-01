#!/bin/bash
# Usage: sbatch scripts/submit_ssl_finetune.sh <fold_id> [--linear-probe]
#SBATCH --job-name=sfx-ssl-finetune
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl-finetune-%j.out
#SBATCH --error=logs/ssl-finetune-%j.err

set -euo pipefail
FOLD="${1:?fold id required (1-4)}"
EXTRA="${2:-}"

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

python -m src.training.train_ssl_finetune \
    --config configs/ssl/mae_finetune.yaml \
    --fold "${FOLD}" \
    --pretrain-checkpoint "checkpoints/mae-vits16-fold${FOLD}-seed42/last.pt" \
    ${EXTRA}
