#!/bin/bash
# stage_ssl_data.sh — SLURM job: stage all CXI detector dirs to local NVMe.
#
# Copies all four detector directories from NFS to /tmp/sfx_stage_shared on
# scg020's local NVMe (~120 GB total). Runs a pre-copy space check (source size
# + 10% headroom) and a post-copy file-size verification to catch silent
# truncation. Exits non-zero on either failure so dependent SLURM jobs
# (fold training jobs submitted via submit_ssl_pretrain_all.sh) never start.
#
# This script is called automatically by submit_ssl_pretrain_all.sh.
# Running it directly is supported but not normally required.
#
# Usage:
#   sbatch scripts/stage_ssl_data.sh
#   bash   scripts/stage_ssl_data.sh --help
#
# Options:
#   -h, --help   Show this help message and exit.
#
# Output:
#   /tmp/sfx_stage_shared/{agipd_20k,jungfrau_20k,epix10k_20k,eiger4m_20k}/
#
# See also:
#   scripts/submit_ssl_pretrain_all.sh   orchestrates stage → train → cleanup
#   scripts/cleanup_ssl_stage.sh         removes /tmp/sfx_stage_shared
#SBATCH --job-name=sfx-ssl-stage
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ssl-stage-%j.out
#SBATCH --error=logs/ssl-stage-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

NFS_SRC="/data/bioxfel/user/gihan/Resonet/production"
STAGE="/tmp/sfx_stage_shared"
mkdir -p logs

# Pre-stage space check: all 4 dirs + 10% headroom.
NEEDED_KB=0
for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
    src="${NFS_SRC}/${det_dir}"
    [ -d "${src}" ] && NEEDED_KB=$(( NEEDED_KB + $(du -sk "${src}" | cut -f1) ))
done
AVAIL_KB=$(df -k /tmp | awk 'NR==2 {print $4}')
NEEDED_WITH_HEADROOM=$(( NEEDED_KB * 11 / 10 ))
if [ "${AVAIL_KB}" -lt "${NEEDED_WITH_HEADROOM}" ]; then
    echo "[stage] ABORT: /tmp has $(( AVAIL_KB / 1024 / 1024 ))G free but need $(( NEEDED_WITH_HEADROOM / 1024 / 1024 ))G" >&2
    exit 1
fi
echo "[stage] space OK: $(( AVAIL_KB / 1024 / 1024 ))G free, need ~$(( NEEDED_KB / 1024 / 1024 ))G"

echo "[stage] copying all detector dirs to ${STAGE} (parallel) ..."
mkdir -p "${STAGE}"
for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
    src="${NFS_SRC}/${det_dir}"
    if [ -d "${src}" ]; then
        mkdir -p "${STAGE}/${det_dir}"
        cp "${src}"/compressed*.cxi "${STAGE}/${det_dir}/" &
    fi
done
wait

# Verify every staged file matches its source size.
CORRUPT=0
for det_dir in agipd_20k jungfrau_20k epix10k_20k eiger4m_20k; do
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
    rm -rf "${STAGE}"
    exit 1
fi
echo "[stage] staging complete. $(du -sh ${STAGE} | cut -f1) in ${STAGE}"
