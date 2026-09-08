#!/bin/bash
# submit_ssl_pretrain.sh — SLURM job: MAE SSL pretraining for a single LODO fold.
#
# Stages the three training detector directories to local NVMe (/tmp), runs MAE
# pretraining, then cleans up staged data on exit. The held-out detector for the
# given fold is never staged (saves ~20–60 GB and eliminates /tmp overflow when
# folds are submitted individually).
#
# When called by submit_ssl_pretrain_all.sh, the SHARED_STAGE env var is set and
# the staging block is skipped entirely — data is read from the shared directory
# staged by stage_ssl_data.sh.
#
# Usage:
#   sbatch scripts/submit_ssl_pretrain.sh <fold_id> [epochs]
#   bash   scripts/submit_ssl_pretrain.sh --help
#
# Arguments:
#   fold_id   LODO fold to train (1–4). Determines which detector is held out.
#   epochs    Optional epoch count override. Omit for the full 400-epoch run.
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Examples:
#   sbatch scripts/submit_ssl_pretrain.sh 3          # full 400-epoch run, fold 3
#   sbatch scripts/submit_ssl_pretrain.sh 1 100      # smoke run, 100 epochs
#
# Environment variables (set automatically by submit_ssl_pretrain_all.sh):
#   SHARED_STAGE   Path to a pre-staged CXI directory. When set, this script
#                  skips staging and cleanup and reads directly from that path.
#
# See also:
#   scripts/submit_ssl_pretrain_all.sh   multi-fold submission with shared staging
#   configs/ssl/mae_pretrain.yaml        pretraining hyperparameters
#   src/training/train_ssl_pretrain.py   training entry point
#SBATCH --job-name=sfx-ssl-pretrain
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/ssl-pretrain-%j.out
#SBATCH --error=logs/ssl-pretrain-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FOLD="${1:?fold id required (1-4)}"
EPOCHS="${2:-}"   # optional; omit for full 400-epoch run

module load mamba/latest
source activate sfx-hitfinder
source .secrets/wandb.env
mkdir -p logs

# ---------------------------------------------------------------------------
# Stage CXI data to local NVMe (/tmp) — avoids NFS read bottleneck
# ---------------------------------------------------------------------------
NFS_SRC="/data/bioxfel/user/gihan/Resonet/production"

if [ -n "${SHARED_STAGE:-}" ]; then
    # Shared pre-staged directory provided by submit_ssl_pretrain_all.sh.
    # Skip staging and cleanup — the cleanup job handles removal.
    STAGE="${SHARED_STAGE}"
    echo "[stage] using shared stage dir ${STAGE}"
else
    # Single-fold run: stage privately and clean up on exit.
    STAGE="/tmp/sfx_stage_${SLURM_JOB_ID}"
    mkdir -p "${STAGE}"
    trap 'echo "[stage] cleaning up ${STAGE}"; rm -rf "${STAGE}"' EXIT

    # Each fold excludes one detector from training — skip staging it.
    declare -A FOLD_EXCLUDE=([1]="agipd_20k" [2]="jungfrau_20k" [3]="epix10k_20k" [4]="eiger4m_20k")
    EXCLUDED="${FOLD_EXCLUDE[${FOLD}]}"
    echo "[stage] fold ${FOLD}: skipping ${EXCLUDED} (held-out detector)"

    # Pre-stage space check: abort if /tmp has less than 10% headroom.
    NEEDED_KB=0
    for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
        [ "${det_dir}" = "${EXCLUDED}" ] && continue
        src="${NFS_SRC}/${det_dir}"
        [ -d "${src}" ] && NEEDED_KB=$(( NEEDED_KB + $(du -sk "${src}" | cut -f1) ))
    done
    AVAIL_KB=$(df -k /tmp | awk 'NR==2 {print $4}')
    NEEDED_WITH_HEADROOM=$(( NEEDED_KB * 11 / 10 ))
    if [ "${AVAIL_KB}" -lt "${NEEDED_WITH_HEADROOM}" ]; then
        echo "[stage] ABORT: /tmp has $(( AVAIL_KB / 1024 / 1024 ))G free but need $(( NEEDED_WITH_HEADROOM / 1024 / 1024 ))G (source + 10%)" >&2
        exit 1
    fi
    echo "[stage] space OK: $(( AVAIL_KB / 1024 / 1024 ))G free, need ~$(( NEEDED_KB / 1024 / 1024 ))G"

    echo "[stage] copying CXI files to ${STAGE} (background, parallel) ..."
    for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
        [ "${det_dir}" = "${EXCLUDED}" ] && continue
        src="${NFS_SRC}/${det_dir}"
        if [ -d "${src}" ]; then
            mkdir -p "${STAGE}/${det_dir}"
            cp "${src}"/compressed*.cxi "${STAGE}/${det_dir}/" &
        fi
    done
    wait

    # Verify every staged file matches its source size — catch silent truncation.
    CORRUPT=0
    for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
        [ "${det_dir}" = "${EXCLUDED}" ] && continue
        src="${NFS_SRC}/${det_dir}"
        for src_file in "${src}"/compressed*.cxi; do
            fname=$(basename "${src_file}")
            dst_file="${STAGE}/${det_dir}/${fname}"
            src_sz=$(stat -c%s "${src_file}")
            dst_sz=$(stat -c%s "${dst_file}" 2>/dev/null || echo 0)
            if [ "${src_sz}" != "${dst_sz}" ]; then
                echo "[stage] CORRUPT: ${det_dir}/${fname} src=${src_sz} dst=${dst_sz}" >&2
                CORRUPT=$(( CORRUPT + 1 ))
            fi
        done
    done
    if [ "${CORRUPT}" -gt 0 ]; then
        echo "[stage] ABORT: ${CORRUPT} file(s) truncated during staging" >&2
        exit 1
    fi
    echo "[stage] staging complete. $(du -sh ${STAGE} | cut -f1) copied to local NVMe."
fi  # end of single-fold staging block

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
EPOCHS_ARG=""
if [ -n "${EPOCHS}" ]; then
    EPOCHS_ARG="--epochs ${EPOCHS}"
fi

/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -u -m src.training.train_ssl_pretrain \
    --config configs/ssl/mae_pretrain.yaml \
    --fold "${FOLD}" \
    --resume \
    --stage-dir "${STAGE}" \
    ${EPOCHS_ARG}
