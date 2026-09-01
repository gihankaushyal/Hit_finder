#!/bin/bash
# Usage: sbatch scripts/submit_ssl_pretrain.sh <fold_id>
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

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

python -m src.training.train_ssl_pretrain \
    --config configs/ssl/mae_pretrain.yaml \
    --fold "${FOLD}" \
    --resume
