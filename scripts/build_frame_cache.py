"""Build the preprocessing frame cache — run once, then reuse for every run.

Computes read → assemble → hitfinder → GCN for every frame of every CXI file
and writes the result as memory-mappable float16 arrays. Uses the same
_compute_gcn_frame the training pipeline calls on a cache miss, so the cache
cannot drift from the live pipeline.

Usage:
    python scripts/build_frame_cache.py --config configs/ssl/mae_finetune.yaml
    python scripts/build_frame_cache.py --config configs/ssl/mae_finetune.yaml \\
        --detectors AGIPD ePix10k --workers 32

Existing complete entries are skipped, so an interrupted build can be resumed
by re-running the same command.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.frame_cache import (  # noqa: E402
    CACHE_DTYPE,
    CENTROIDS_NAME,
    FRAMES_NAME,
    MANIFEST_NAME,
    VALID_MASK_NAME,
    centroid_key,
    entry_dir,
    manifest_diff,
    write_manifest,
)
from src.preprocessing.io import count_frames, read_detector_description  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def build_one_cxi(
    cxi_path: Path,
    cache_root: Path,
    cfg: dict,
    backend: str | None = None,
    overwrite: bool = False,
) -> int:
    """Cache every frame of one CXI file. Returns the number of frames written.

    Writes into '<entry>.tmp' and os.replace()s it into position, so a killed
    build leaves either nothing or a complete entry — never a partial one that
    would be silently read as valid.
    """
    from src.data.dataset import _compute_gcn_frame
    from src.hitfinders import get_hitfinder

    cxi_path = Path(cxi_path)
    final = entry_dir(cache_root, cxi_path)
    if final.is_dir() and (final / FRAMES_NAME).is_file() and not overwrite:
        return 0

    hf_cfg = dict(cfg)
    if backend is not None:
        hf_cfg["hitfinder"] = dict(cfg.get("hitfinder", {}), backend=backend)
    hitfinder = get_hitfinder(hf_cfg)

    desc = read_detector_description(cxi_path)
    n_frames = count_frames(cxi_path)

    tmp = final.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    geom_cache: dict[Path, dict[str, float]] = {}
    holder: list = [None]
    frames_out: np.memmap | None = None
    centroids_out: dict[str, np.ndarray] = {}
    mask_out: np.ndarray | None = None

    for idx in range(n_frames):
        frame, mask, centroids = _compute_gcn_frame(
            cxi_path, idx, desc, geom_cache, hitfinder, holder
        )
        if frames_out is None:
            # Open a disk-backed memmap once the assembled shape is known (it
            # is geometry-derived, identical for every frame in the file), so
            # a 5000-frame x ~2000x2000 file (~40GB) never needs a matching
            # in-RAM buffer — each write below goes straight to the tmp file,
            # bounding this worker's RSS regardless of n_frames.
            frames_out = np.lib.format.open_memmap(
                tmp / FRAMES_NAME,
                mode="w+",
                dtype=CACHE_DTYPE,
                shape=(n_frames, *frame.shape),
            )
            mask_out = mask
        frames_out[idx] = frame.astype(CACHE_DTYPE)
        centroids_out[centroid_key(idx)] = centroids.astype(np.float32)

    if frames_out is None:
        shutil.rmtree(tmp)
        raise ValueError(f"{cxi_path} contains no frames")

    frames_out.flush()
    del frames_out
    np.savez(tmp / CENTROIDS_NAME, **centroids_out)

    # valid_mask is a pure function of (desc, shape) — one copy per detector dir.
    mask_path = final.parent / VALID_MASK_NAME
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not mask_path.is_file():
        mask_tmp = mask_path.with_name(
            f"{mask_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npy"
        )
        np.save(mask_tmp, np.asarray(mask_out, dtype=bool))
        os.replace(mask_tmp, mask_path)

    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return n_frames


def _build_worker(args: tuple[Path, Path, dict, bool]) -> tuple[str, int, str]:
    """Pool entry point. Returns (name, n_frames, error_message)."""
    cxi_path, cache_root, cfg, overwrite = args
    try:
        n = build_one_cxi(cxi_path, cache_root, cfg, overwrite=overwrite)
        return (cxi_path.name, n, "")
    except Exception as exc:  # noqa: BLE001 — one bad file must not kill the build
        return (cxi_path.name, 0, f"{type(exc).__name__}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--cache-root",
        default=None,
        help="Override cache.root from the config.",
    )
    p.add_argument(
        "--detectors",
        nargs="*",
        default=None,
        help="Subset of lodo.detector_dirs keys to build (default: all).",
    )
    p.add_argument("--workers", type=int, default=32)
    p.add_argument(
        "--overwrite", action="store_true", help="Rebuild entries that already exist."
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    cache_root = Path(args.cache_root or cfg["cache"]["root"])
    cache_root.mkdir(parents=True, exist_ok=True)

    # An existing manifest whose params differ from the current config — a
    # pipeline_version bump (e.g. Jungfrau's _to_2d -> PADAssembler switch) OR
    # a changed hitfinder param (pf8 threshold, backend, gpu_script_path,
    # ...) — means some entries under this root were built under different
    # logic. The per-entry skip below only checks file existence, not params,
    # so a plain re-run would leave those entries stale while write_manifest()
    # at the end silently claims the current params. Require an explicit
    # --overwrite instead of rebuilding a possibly-469GB cache (or lying
    # about its freshness) by default.
    def _abort(reason: str) -> None:
        print(f"[build] ABORT: {reason}")
        sys.exit(1)

    manifest_path = cache_root / MANIFEST_NAME
    if manifest_path.is_file():
        diff = manifest_diff(cache_root, cfg)
        stale = diff is not None
        if stale and not args.overwrite:
            _abort(
                f"{cache_root} manifest is stale: {diff}. Existing entries "
                "were built under different parameters and will NOT be "
                "refreshed by a plain re-run. Re-run with --overwrite to "
                "rebuild every entry under the current parameters."
            )
        if stale and args.detectors:
            # write_manifest() below stamps params for the WHOLE cache_root,
            # not just the rebuilt subset. A partial `--overwrite
            # --detectors X` run would leave every non-rebuilt detector's
            # entries on the OLD params while the manifest falsely claims
            # the entire root is current. Require a full rebuild (no
            # --detectors) to bump the on-disk params.
            _abort(
                f"{cache_root} manifest is stale: {diff}, and --detectors "
                "restricts this run to a subset. A partial rebuild cannot "
                "safely bump the manifest's params for detectors it does "
                "not touch. Re-run with --overwrite and no --detectors "
                "filter to rebuild the whole cache."
            )

    pattern = cfg["lodo"].get("cxi_pattern", "compressed*.cxi")
    detector_dirs = cfg["lodo"]["detector_dirs"]
    if args.detectors:
        detector_dirs = {k: v for k, v in detector_dirs.items() if k in args.detectors}

    jobs: list[tuple[Path, Path, dict, bool]] = []
    for _detector, dir_str in detector_dirs.items():
        for cxi in sorted(Path(dir_str).glob(pattern)):
            jobs.append((cxi, cache_root, cfg, args.overwrite))

    print(f"[build] {len(jobs)} CXI files → {cache_root} using {args.workers} workers")

    total, failures = 0, []
    with Pool(processes=args.workers) as pool:
        for name, n, err in pool.imap_unordered(_build_worker, jobs):
            if err:
                failures.append((name, err))
                print(f"[build] FAIL {name}: {err}", flush=True)
            else:
                total += n
                print(f"[build] ok   {name}: {n} frames", flush=True)

    if failures:
        print(f"[build] ABORT: {len(failures)} file(s) failed; manifest NOT written")
        sys.exit(1)

    # Written last: its presence is the signal that the cache is complete.
    write_manifest(cache_root, cfg)
    print(f"[build] done. {total} frames cached. Manifest written to {cache_root}")


if __name__ == "__main__":
    main()
