#!/bin/bash
# cleanup_ssl_stage.sh — SLURM job: remove the SSL NVMe staging directories.
#
# Deletes both the legacy raw-CXI staging directory (/tmp/sfx_stage_shared) and
# the frame-cache NVMe tier (/tmp/sfx_frame_cache) from scg020's local NVMe
# after all SSL pretrain/fine-tune fold jobs have finished (submitted with
# afterany dependency by submit_ssl_pretrain_all.sh / submit_ssl_finetune_all.sh
# so it runs regardless of fold job exit status).
#
# This script is called automatically by submit_ssl_pretrain_all.sh.
# Running it directly is supported for manual cleanup if needed.
#
# Usage:
#   sbatch scripts/cleanup_ssl_stage.sh
#   bash   scripts/cleanup_ssl_stage.sh --help
#
# Options:
#   -h, --help   Show this help message and exit.
#
# See also:
#   scripts/submit_ssl_pretrain_all.sh   orchestrates stage → train → cleanup
#   scripts/stage_ssl_data.sh            creates /tmp/sfx_stage_shared
#SBATCH --job-name=sfx-ssl-cleanup
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ssl-cleanup-%j.out
#SBATCH --error=logs/ssl-cleanup-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p logs

for STAGE in /tmp/sfx_stage_shared /tmp/sfx_frame_cache; do
    if [ -d "${STAGE}" ]; then
        echo "[cleanup] removing ${STAGE} ($(du -sh ${STAGE} | cut -f1))"
        rm -rf "${STAGE}"
    else
        echo "[cleanup] ${STAGE} already gone — nothing to do"
    fi
done
echo "[cleanup] done"
