"""Two-tier on-disk cache for the deterministic preprocessing prefix.

Stores the result of read → assemble → hitfinder → GCN so it runs once per
frame instead of once per frame per epoch. See
docs/superpowers/specs/2026-09-14-frame-cache-design.md.

Layout::

    <cache_root>/
    ├── cache_manifest.json
    └── <detector_dir_name>/
        ├── valid_mask.npy          # (H, W) bool, once per detector
        └── <cxi_stem>/
            ├── frames.npy          # (N, H, W) float16, read via mmap
            └── centroids.npz       # c_00000 ... c_0NNNN, each (Ni, 2) float32

Entries are keyed by ``(detector_dir_name, cxi_stem)`` rather than by absolute
path: the ``--stage-dir`` flag rewrites CXI paths to an NVMe copy, and
absolute-path keys would silently stop resolving.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.preprocessing.normalize import GCN_EPSILON, LCN_EPSILON, LCN_WINDOW_DEFAULT
from src.preprocessing.pipeline import EDGE_EROSION_PX

CACHE_DTYPE = np.float16
MANIFEST_NAME = "cache_manifest.json"
FRAMES_NAME = "frames.npy"
CENTROIDS_NAME = "centroids.npz"
VALID_MASK_NAME = "valid_mask.npy"

# Geometry files whose contents change the assembled output. Hashed into the
# manifest so editing a .geom invalidates the cache.
GEOMETRY_DIR = Path(__file__).resolve().parents[1] / "preprocessing" / "data"

# PF8 settings that change centroid output. backend is included because
# different backends can produce different peak sets for the same parameters.
_HITFINDER_KEYS = (
    "backend",
    "pf8_threshold",
    "pf8_min_snr",
    "pf8_min_pix_count",
    "pf8_max_pix_count",
    "pf8_local_bg_radius",
    "pf8_min_res",
    "pf8_max_res",
    "pf8_use_saturated",
)


class CacheStaleError(RuntimeError):
    """The on-disk cache was built with different parameters than the caller uses."""


class CacheMissError(KeyError):
    """The requested frame is not present in any cache root."""


def centroid_key(frame_idx: int) -> str:
    """Name of the per-frame array inside centroids.npz."""
    return f"c_{frame_idx:05d}"


def cache_key(cxi_path: Path) -> tuple[str, str]:
    """(detector_dir_name, cxi_stem) — stable across NFS and NVMe-staged paths."""
    p = Path(cxi_path)
    return p.parent.name, p.stem


def entry_dir(root: Path, cxi_path: Path) -> Path:
    """Directory holding one CXI file's cached frames and centroids."""
    det, stem = cache_key(cxi_path)
    return Path(root) / det / stem


def _geometry_digest() -> dict[str, str]:
    """sha256 of every detector geometry file, keyed by filename.

    Content hashing rather than mtime: staging and rsync perturb mtimes without
    changing the geometry, and a false-positive stale error on a 469 GB cache is
    expensive.
    """
    digest: dict[str, str] = {}
    if not GEOMETRY_DIR.is_dir():
        return digest
    for f in sorted(GEOMETRY_DIR.iterdir()):
        if f.suffix in (".geom", ".json") and f.is_file():
            digest[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return digest


def _git_sha() -> str:
    """Current commit, or 'unknown' outside a git checkout. Recorded, never compared."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build_manifest(cfg: dict) -> dict:
    """Describe every input that changes cached bytes.

    Only manifest["params"] participates in staleness comparison. git_sha and
    created are provenance only — a new commit must not invalidate the cache.
    """
    hf = cfg.get("hitfinder", {})
    params = {
        "hitfinder": {k: hf.get(k) for k in _HITFINDER_KEYS},
        "gcn_eps": GCN_EPSILON,
        "lcn_eps": LCN_EPSILON,
        "lcn_window": LCN_WINDOW_DEFAULT,
        "edge_erosion_px": EDGE_EROSION_PX,
        "cache_dtype": np.dtype(CACHE_DTYPE).name,
        "geometry": _geometry_digest(),
    }
    return {
        "params": params,
        "git_sha": _git_sha(),
        "created": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(cache_root: Path, cfg: dict) -> None:
    """Write the manifest. Call LAST so an interrupted build looks unbuilt."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / MANIFEST_NAME).write_text(json.dumps(build_manifest(cfg), indent=2))


def _first_difference(want: dict, have: dict, prefix: str = "") -> str | None:
    """Name of the first differing key, depth-first. None when equal."""
    for key in sorted(set(want) | set(have)):
        path = f"{prefix}{key}"
        a, b = want.get(key), have.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            nested = _first_difference(a, b, prefix=f"{path}.")
            if nested is not None:
                return nested
        elif a != b:
            return f"{path} (cache={b!r}, config={a!r})"
    return None


def verify_manifest(cache_root: Path, cfg: dict) -> None:
    """Raise CacheStaleError naming the first differing key.

    Fails loudly and early rather than letting a stale cache silently corrupt
    an experiment.
    """
    manifest_path = Path(cache_root) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CacheStaleError(
            f"no manifest at {manifest_path}; build the cache with "
            "`python -m scripts.build_frame_cache` or pass --no-cache."
        )
    on_disk = json.loads(manifest_path.read_text()).get("params", {})
    diff = _first_difference(build_manifest(cfg)["params"], on_disk)
    if diff is not None:
        raise CacheStaleError(
            f"frame cache at {cache_root} is stale: {diff}. "
            "Rebuild the cache or pass --no-cache."
        )


class FrameCache:
    """Resolve cached frames against an ordered list of roots (NVMe, then NFS).

    Handles are opened lazily inside get() and never in __init__, because
    DataLoader workers fork after construction (CLAUDE.md rule #4). A pid guard
    drops any handle inherited across a fork.
    """

    def __init__(self, roots: Sequence[Path]) -> None:
        self._roots = [Path(r) for r in roots]
        self._pid = os.getpid()
        self._frames: dict[tuple[str, str, str], np.ndarray] = {}
        self._masks: dict[tuple[str, str], np.ndarray] = {}
        self._centroids: dict[tuple[str, str, str], np.lib.npyio.NpzFile] = {}

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def _reset_if_forked(self) -> None:
        if os.getpid() != self._pid:
            self._frames.clear()
            self._masks.clear()
            self._centroids.clear()
            self._pid = os.getpid()

    def _mask(self, root: Path, det: str) -> np.ndarray:
        key = (str(root), det)
        mask = self._masks.get(key)
        if mask is None:
            mask = np.load(root / det / VALID_MASK_NAME)
            mask.setflags(write=False)  # shared across every frame of this detector
            self._masks[key] = mask
        return mask

    def get(
        self, cxi_path: Path, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (gcn_frame float32, valid_mask bool read-only, centroids (N,2) float32).

        Raises:
            CacheMissError: The entry is absent from every root, or the entry
                exists but does not contain frame_idx (truncated build).
        """
        self._reset_if_forked()
        det, stem = cache_key(cxi_path)
        found_out_of_range = False
        for root in self._roots:
            entry = root / det / stem
            frames_path = entry / FRAMES_NAME
            if not frames_path.is_file():
                continue

            key = (str(root), det, stem)
            memmap = self._frames.get(key)
            if memmap is None:
                memmap = np.load(frames_path, mmap_mode="r")
                self._frames[key] = memmap
            if not 0 <= frame_idx < memmap.shape[0]:
                # This root's copy is truncated/partial; a later root may have
                # a complete copy of the same entry, so keep searching instead
                # of failing on the first (possibly stale) match.
                found_out_of_range = True
                continue

            npz = self._centroids.get(key)
            if npz is None:
                npz = np.load(entry / CENTROIDS_NAME)
                self._centroids[key] = npz

            # Slicing a memmap and casting produces a fresh writable array, so
            # downstream in-place ops (fill_gaps_after_gcn) cannot touch the file.
            frame = np.asarray(memmap[frame_idx], dtype=np.float32)
            centroids = np.asarray(npz[centroid_key(frame_idx)], dtype=np.float32)
            return frame, self._mask(root, det), centroids

        if found_out_of_range:
            raise CacheMissError(
                f"{det}/{stem} frame {frame_idx} out of range in every root that "
                "has this entry (all copies truncated/partial): "
                + ", ".join(str(r) for r in self._roots)
            )
        raise CacheMissError(
            f"{det}/{stem} frame {frame_idx} not found in any of: "
            + ", ".join(str(r) for r in self._roots)
        )


def verify_cache_or_raise(frame_cache: FrameCache | None, cfg: dict) -> None:
    """Fail fast if the configured frame cache does not match this run's config.

    A tier with no manifest is skipped, not an error: the NVMe tier is staged
    per fold and may legitimately be empty. But if *no* tier carries a manifest
    the caller has pointed at a cache that was never built, which is an error.
    """
    if frame_cache is None:
        return
    verified = 0
    for root in frame_cache.roots:
        if (root / MANIFEST_NAME).exists():
            verify_manifest(root, cfg)
            verified += 1
    if verified == 0:
        roots_desc = ", ".join(str(r) for r in frame_cache.roots)
        raise CacheStaleError(
            f"no cache manifest found in any configured root: {roots_desc} — "
            "each configured root exists as a directory but contains no "
            f"{MANIFEST_NAME} (e.g. an empty staging directory, or a stray "
            "leftover dir). Has scripts/build_frame_cache.py (or the NVMe "
            "staging step) been run for this root? Build the cache with "
            "scripts/build_frame_cache.py, or pass --no-cache to run without it."
        )


def frame_cache_from_cfg(cfg: dict) -> FrameCache | None:
    """Build a two-tier FrameCache from the `cache:` block of a run config.

    Returns None when caching is disabled or no configured root exists on this
    node — callers treat None as "compute everything live", which is exactly the
    behaviour the pipeline had before the cache existed.
    """
    cache_cfg = cfg.get("cache") or {}
    if not cache_cfg.get("enabled", False):
        return None
    roots: list[Path] = []
    for key in ("nvme_root", "root"):
        value = cache_cfg.get(key)
        if value:
            path = Path(value)
            if path.is_dir():
                roots.append(path)
    if not roots:
        return None
    return FrameCache(roots)
