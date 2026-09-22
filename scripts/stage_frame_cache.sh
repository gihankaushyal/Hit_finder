#!/bin/bash
# stage_frame_cache.sh — SLURM job: stage one fold's frame-cache entries to NVMe.
#
# Copies cache entry directories from the permanent NFS cache to
# /tmp/sfx_frame_cache on scg020's local NVMe, in the priority order emitted by
# scripts/plan_cache_staging.py: train sessions first, then val. Copying stops
# when free space drops below the reserve; whatever did not fit streams from NFS
# at ~1.8 GB/s, which is fine because the expensive part (assembly, PF8, GCN) is
# already precomputed either way.
#
# Fold 1 (AGIPD held out) needs ~364 GB for train+val against ~333 GB of /tmp,
# so partial staging there is expected, not a failure.
#
# Usage:
#   sbatch scripts/stage_frame_cache.sh <fold_id>
#   bash   scripts/stage_frame_cache.sh --help
#
# Environment:
#   CACHE_NFS   permanent cache root (default /data/bioxfel/user/gihan/Hit_finder_cache)
#   CACHE_NVME  local tier          (default /tmp/sfx_frame_cache)
#   CONFIG      config to derive the split from (default configs/ssl/mae_finetune.yaml)
#
# See also:
#   scripts/build_frame_cache.py      builds the NFS cache (one time)
#   scripts/cleanup_ssl_stage.sh      removes the NVMe tier
#SBATCH --job-name=sfx-cache-stage
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --nodelist=scg020
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cache-stage-%j.out
#SBATCH --error=logs/cache-stage-%j.err

set -euo pipefail

usage() {
    sed -n '2,/^#SBATCH/{ /^#SBATCH/d; s/^# \{0,1\}//; p }' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

FOLD="${1:?fold id required (1-4)}"
CACHE_NFS="${CACHE_NFS:-/data/bioxfel/user/gihan/Hit_finder_cache}"
CACHE_NVME="${CACHE_NVME:-/tmp/sfx_frame_cache}"
CONFIG="${CONFIG:-configs/ssl/mae_finetune.yaml}"
# Leave 20 GB free so the node's /tmp does not fill completely.
RESERVE_KB=$(( 20 * 1024 * 1024 ))

module load mamba/latest
source activate sfx-hitfinder
mkdir -p logs "${CACHE_NVME}"

if [ ! -f "${CACHE_NFS}/cache_manifest.json" ]; then
    echo "[stage] ABORT: no cache_manifest.json in ${CACHE_NFS}" >&2
    echo "[stage] build it first: python scripts/build_frame_cache.py --config ${CONFIG}" >&2
    exit 1
fi

# The manifest must land on NVMe too, or _verify_cache_or_raise sees an
# unverifiable tier and FrameCache resolves entries against an unlabelled root.
cp "${CACHE_NFS}/cache_manifest.json" "${CACHE_NVME}/"

STAGED=0
SKIPPED=0
while read -r entry; do
    src="${CACHE_NFS}/${entry}"
    dst="${CACHE_NVME}/${entry}"
    [ -d "${src}" ] || { echo "[stage] missing in NFS cache: ${entry}" >&2; continue; }
    [ -d "${dst}" ] && continue

    need_kb=$(du -sk "${src}" | cut -f1)
    avail_kb=$(df -k "${CACHE_NVME}" | awk 'NR==2 {print $4}')
    if [ $(( avail_kb - need_kb )) -lt "${RESERVE_KB}" ]; then
        SKIPPED=$(( SKIPPED + 1 ))
        continue
    fi

    mkdir -p "$(dirname "${dst}")"
    cp -r "${src}" "${dst}.tmp"
    mv "${dst}.tmp" "${dst}"
    STAGED=$(( STAGED + 1 ))
done < <(python scripts/plan_cache_staging.py --config "${CONFIG}" --fold "${FOLD}")

# Per-detector valid masks are tiny and shared by every entry — always copy them.
for det_dir in "${CACHE_NFS}"/*/; do
    det=$(basename "${det_dir}")
    if [ -f "${det_dir}/valid_mask.npy" ]; then
        mkdir -p "${CACHE_NVME}/${det}"
        cp "${det_dir}/valid_mask.npy" "${CACHE_NVME}/${det}/"
    fi
done

echo "[stage] fold ${FOLD}: ${STAGED} entries staged, ${SKIPPED} left on NFS"
echo "[stage] ${CACHE_NVME} now holds $(du -sh "${CACHE_NVME}" | cut -f1)"
