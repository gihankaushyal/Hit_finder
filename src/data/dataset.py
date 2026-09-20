"""PyTorch Datasets for SFX diffraction images. HDF5 opened lazily in __getitem__."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import warnings

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.geometry import get_assembler, get_geometry
from src.preprocessing.io import (
    count_frames,
    read_detector_description,
    read_embedded_labels,
    read_frame,
    read_image,
)
from src.hitfinders.base import Hitfinder
from src.preprocessing.augment import (
    PAD_BORDER_DEFAULT,
    pad_border,
    random_cutout,
    random_flip,
    random_rot90,
)
from src.preprocessing.normalize import gcn, lcn
from src.preprocessing.pipeline import (
    _to_2d,
    assemble_only,
    fill_gaps_after_gcn,
    get_valid_mask_for_frame,
)


class UnlabeledDataset(Dataset):
    """Dataset of assembled diffraction images with no labels.

    Intended for SSL (MAE) pretraining. Accepts .img, .h5, or .cxi files.
    Each item is a float32 tensor of shape (1, H, W).
    """

    def __init__(self, file_list: list[str | Path]) -> None:
        self._paths = [Path(p) for p in file_list]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = read_image(self._paths[idx])
        return torch.from_numpy(image).unsqueeze(0)


class MultiFrameCXIDataset(Dataset):
    """DEPRECATED: Use AsymmetricCXIDataset for new training pipelines.

    Dataset over multi-frame CXI files with embedded hit labels.

    Each CXI file contributes N frames. The dataset expands all files into a
    flat (file_path, frame_idx) index so a DataLoader can iterate frames
    individually.

    Label convention: embedded float32 value 1.0 → hit (class 1),
                      0.0 → non-hit (class 0).

    HDF5 files are opened lazily in __getitem__ (multiprocessing safe).

    Args:
        cxi_paths: List of paths to multi-frame CXI/HDF5 files.
        label_key: HDF5 key for the per-frame label array in each file.
        preprocess_fn: Optional callable applied to each raw (H, W) float32
            frame before converting to a tensor. Pass None to return raw frames.
    """

    def __init__(
        self,
        cxi_paths: list[str | Path],
        label_key: str = "entry_1/labels/hit",
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._preprocess_fn = preprocess_fn

        # Build flat index and cache labels eagerly — label arrays are small
        # and reading them per __getitem__ caused O(N) file opens per epoch.
        self._index: list[tuple[Path, int]] = []
        self._labels: list[int] = []
        for p in cxi_paths:
            p = Path(p)
            arr = read_embedded_labels(p, label_key)
            for i, raw in enumerate(arr):
                self._index.append((p, i))
                self._labels.append(int(round(float(raw))))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, frame_idx = self._index[idx]
        frame = read_frame(path, frame_idx)
        if self._preprocess_fn is not None:
            frame = self._preprocess_fn(frame)
        tensor = torch.from_numpy(frame).unsqueeze(0)
        return tensor, self._labels[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crop_contains_centroid(
    top: int, left: int, size: int, centroids: np.ndarray
) -> bool:
    """Return True if any centroid falls inside the crop region.

    Centroid (x, y) is inside if: left <= x < left+size AND top <= y < top+size.

    Args:
        top: Row offset of the crop's top-left corner.
        left: Column offset of the crop's top-left corner.
        size: Side length of the square crop in pixels.
        centroids: (N, 2) float32 array of [x, y] pairs. Empty array → False.

    Returns:
        True if at least one centroid lies strictly inside the crop region.
    """
    if centroids.shape[0] == 0:
        return False
    xs = centroids[:, 0]
    ys = centroids[:, 1]
    inside = (xs >= left) & (xs < left + size) & (ys >= top) & (ys < top + size)
    return bool(inside.any())


def _crop_within_margin(
    top: int, left: int, size: int, centroids: np.ndarray, margin: int = 50
) -> bool:
    """Return True if any centroid is within margin px of the crop region.

    Used for hard-negative mining: rejects crops where a Bragg peak sits just
    outside the boundary but its diffuse halo still overlaps the tile.

    A centroid (x, y) is "too close" if it lies in the expanded region
    [left-margin, left+size+margin) × [top-margin, top+size+margin).

    Args:
        top: Row offset of the crop's top-left corner.
        left: Column offset of the crop's top-left corner.
        size: Side length of the square crop in pixels.
        centroids: (N, 2) float32 array of [x, y] pairs. Empty array → False.
        margin: Clearance buffer in pixels around the crop boundary (default 50).

    Returns:
        True if at least one centroid is inside or within margin px of the crop.
    """
    if centroids.shape[0] == 0:
        return False
    xs = centroids[:, 0]
    ys = centroids[:, 1]
    near = (
        (xs >= left - margin)
        & (xs < left + size + margin)
        & (ys >= top - margin)
        & (ys < top + size + margin)
    )
    return bool(near.any())


def _path_a_crop(
    padded: np.ndarray,
    centroids: np.ndarray,
    rng: np.random.Generator,
    ph: int,
    pw: int,
    size: int,
) -> tuple[np.ndarray, int]:
    """Crop centred on a randomly chosen Bragg peak.

    Args:
        padded: (H, W) or (H, W, C) padded frame stack to crop from.
        centroids: (N, 2) float32 array of [x, y] pairs in padded coordinates.
            Must be non-empty.
        rng: Numpy Generator used to pick which centroid to centre on.
        ph: Padded frame height.
        pw: Padded frame width.
        size: Side length of the square crop in pixels.

    Returns:
        (crop, label) where label is always 1.
    """
    peak = centroids[int(rng.integers(0, len(centroids)))]
    cx = int(round(float(peak[0])))  # column
    cy = int(round(float(peak[1])))  # row
    left = int(np.clip(cx - size // 2, 0, pw - size))
    top = int(np.clip(cy - size // 2, 0, ph - size))
    crop = padded[top : top + size, left : left + size].copy()
    return crop, 1


def _sample_clear_crop(
    padded: np.ndarray,
    centroids: np.ndarray,
    rng: np.random.Generator,
    ph: int,
    pw: int,
    size: int,
    margin: int,
    max_tries: int,
) -> np.ndarray | None:
    """Randomly sample a crop position with `margin` px clearance from every centroid.

    Args:
        padded: (H, W, C) padded frame stack to crop from.
        centroids: (N, 2) float32 array of [x, y] pairs in padded coordinates.
        rng: Numpy Generator used for position sampling.
        ph: Padded frame height.
        pw: Padded frame width.
        size: Side length of the square crop in pixels.
        margin: Clearance buffer in pixels around the crop boundary.
        max_tries: Number of random positions to attempt before giving up.

    Returns:
        The cropped (size, size, C) array, or None if no clear position was
        found within max_tries attempts.
    """
    for _ in range(max_tries):
        top = int(rng.integers(0, ph - size + 1))
        left = int(rng.integers(0, pw - size + 1))
        if not _crop_within_margin(top, left, size, centroids, margin=margin):
            return padded[top : top + size, left : left + size].copy()
    return None


def _load_gcn_frame(
    path: Path,
    frame_idx: int,
    desc: str | None,
    geom_cache: dict[Path, dict[str, float]],
    hitfinder: Hitfinder | None,
    last_geom_path_holder: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """read → assemble (with _to_2d fallback) → hitfinder on raw → GCN → fill gaps.

    Shared by AsymmetricCXIDataset and SSLPretrainCXIDataset so both pipelines
    stay bit-identical up to the crop step. When ``hitfinder`` is None the
    hitfinder stage is skipped and an empty (0, 2) centroid array is returned.

    ``last_geom_path_holder`` is a 1-element mutable list so the caller's
    set_geometry dedup state survives across calls.

    Returns:
        (gcn_frame, valid_mask, centroids) — centroids are [x, y] float32 in
        the un-padded assembled coordinate frame.
    """
    frame = read_frame(path, frame_idx)

    # --- Assemble to native resolution ---
    if desc is not None and "JUNGFRAU" not in desc.upper():
        try:
            pads = get_geometry(desc)
            assembler = get_assembler(desc)
            assembled = assemble_only(frame, pads, desc, assembler=assembler)
        except (ValueError, KeyError, OSError):
            # ValueError: unrecognised descriptor that slipped past the
            # JUNGFRAU guard (e.g. novel variant); fall back to _to_2d.
            assembled = _to_2d(frame)
    else:
        assembled = _to_2d(frame)

    centroids = np.zeros((0, 2), dtype=np.float32)
    if hitfinder is not None:
        # Lazily read geometry keys on first access (lazy per CLAUDE.md rule #4).
        if path not in geom_cache:
            try:
                with h5py.File(path, "r") as _f:
                    geom_cache[path] = {
                        "dist": float(
                            _f["entry_1/instrument_1/detector_1/distance"][()]
                        ),
                        "wavelength": float(
                            _f["entry_1/instrument_1/source_1/wavelength"][()]
                        ),
                        # x_pixel_size == y_pixel_size confirmed for all four supported detectors
                        "pixel_size": float(
                            _f["entry_1/instrument_1/detector_1/x_pixel_size"][()]
                        ),
                    }
            except (KeyError, OSError) as exc:
                warnings.warn(
                    f"Geometry keys missing for {path} ({exc}); "
                    "set_geometry will be skipped for this file.",
                    stacklevel=2,
                )

        # Update geometry when the CXI file changes (avoids redundant calls per frame).
        if (
            path != last_geom_path_holder[0]
            and path in geom_cache
            and hasattr(hitfinder, "set_geometry")
        ):
            hitfinder.set_geometry(**geom_cache[path])
            last_geom_path_holder[0] = path

        # --- Run hitfinder on raw assembled frame (before GCN) ---
        centroids = hitfinder.find_peaks(assembled)  # (N, 2) float32 [x, y]

    # GCN applied to the full assembled frame before padding/crop; then gap/
    # padding/edge pixels are set to 0 (= global mean in GCN units) and
    # tracked in a valid-pixel mask so LCN can exclude them from local stats.
    valid_mask = get_valid_mask_for_frame(desc, assembled.shape)
    if valid_mask is None:
        warnings.warn(
            f"No valid-pixel mask for desc={desc!r} shape={assembled.shape}; "
            "gap fill and masked LCN disabled for this frame.",
            stacklevel=2,
        )
        valid_mask = np.ones(assembled.shape, dtype=bool)
    assembled = gcn(assembled)
    assembled = fill_gaps_after_gcn(assembled, desc, mask=valid_mask)
    return assembled, valid_mask, centroids


# ---------------------------------------------------------------------------
# AsymmetricCXIDataset
# ---------------------------------------------------------------------------


class AsymmetricCXIDataset(Dataset):
    """Primary training dataset: hitfinder-guided crop with augmentation and normalisation.

    For each frame:
      1. Read the embedded ground-truth label; the hitfinder only runs when
         the label is HIT (metadata gating — see step 3)
      2. Read + assemble to native resolution
      3. If metadata HIT: call set_geometry on hitfinder (if supported) with
         CXI dist/wavelength/pixel_size, then run hitfinder on the raw
         assembled frame → centroids (N_peaks, 2). If metadata NON-HIT, the
         hitfinder is skipped entirely and centroids is empty.
      4. Apply GCN to the full assembled frame
      5. Pad frame by PAD_BORDER_DEFAULT px on each edge; shift centroids
      6. Guided crop (224×224) → derived label:
           Metadata HIT, centroids found: 50/50 coin toss —
             Path A: crop centred on a random Bragg peak → label=1
             Path B: crop with 50 px clearance from all peaks (hard-negative
               background from a real hit frame) → label=0; falls back to
               Path A if no clear position is found within 50 attempts
           Metadata HIT, no centroids found: random crop → label=0
           Metadata NON-HIT: random crop (no clearance check needed,
             centroids is always empty) → label=0
      7. Augment (geometric): random_rot90 → random_flip
      8. Normalise: masked LCN (GCN already applied to full frame in step 4)
      9. Augment: peak-aware random_cutout (after LCN — holes are exact 0
         in LCN space and never enter the local statistics)
      10. Return (tensor(1, 224, 224) float32, derived_label)

    Args:
        session_ids: Session IDs to include.
        session_map: Maps session_id → Path to CXI file.
        hitfinder: Hitfinder Protocol instance (find_peaks method).
        label_key: HDF5 key for per-frame labels (used only to build the flat index).
        seed: Base RNG seed; per-sample seed is seed+idx for reproducibility.
        hit_frac: Probability of choosing Path A (peak-centred, label=1) over
            Path B (hard-negative) on a coin toss for metadata-HIT frames with
            centroids found. Default 0.5.
        hard_neg_max_attempts: Max random-position attempts when searching for
            a hard-negative crop with margin clearance before falling back to
            Path A (or returning None if no centroids exist). Default 50.
    """

    def __init__(
        self,
        session_ids: list[str],
        session_map: dict[str, str | Path],
        hitfinder: Hitfinder,
        label_key: str = "entry_1/labels/hit",
        seed: int = 42,
        hit_frac: float = 0.5,
        hard_neg_max_attempts: int = 50,
    ) -> None:
        self._hitfinder = hitfinder
        self._label_key = label_key
        self._seed = seed
        self._hit_frac = hit_frac
        self._hard_neg_max_attempts = hard_neg_max_attempts
        self._hard_neg_margin = 50
        self._last_geom_path_holder: list = [None]

        # Resolve session_map to Path objects for requested session_ids only.
        cxi_paths: list[Path] = []
        for sid in session_ids:
            cxi_paths.append(Path(session_map[sid]))

        # Read detector descriptions eagerly — geometry objects are NOT stored
        # (not picklable); get_geometry/get_assembler use a process-local cache.
        unique_paths = set(cxi_paths)
        self._path_to_desc: dict[Path, str] = {}
        # _path_to_geom is populated lazily in __getitem__ (CLAUDE.md rule #4:
        # never open HDF5 in __init__ — multiprocessing DataLoader workers will deadlock).
        self._path_to_geom: dict[Path, dict[str, float]] = {}
        for p in unique_paths:
            try:
                desc = read_detector_description(p)
                self._path_to_desc[p] = desc
            except (ValueError, KeyError, OSError):
                pass

        # Build flat (path, frame_idx) index and cache labels.
        self._index: list[tuple[Path, int]] = []
        self._labels: list[int] = []
        for p in cxi_paths:
            try:
                arr = read_embedded_labels(p, label_key)
            except (KeyError, OSError) as e:
                warnings.warn(
                    f"AsymmetricCXIDataset: cannot read labels from {p}: {e}; "
                    "file skipped.",
                    stacklevel=2,
                )
                continue
            for i, raw in enumerate(arr):
                self._index.append((p, i))
                self._labels.append(int(round(float(raw))))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | None:
        path, frame_idx = self._index[idx]
        desc = self._path_to_desc.get(path)
        # Metadata gating: only run the hitfinder for frames the embedded
        # ground-truth label marks as a hit. Non-hit frames skip PF8 entirely
        # — _load_gcn_frame's `if hitfinder is not None` guard already
        # short-circuits every hitfinder-related step and returns empty
        # centroids, so the rest of this method is unchanged for that case.
        label = self._labels[idx]
        hitfinder = self._hitfinder if label == 1 else None
        assembled, valid_mask, centroids = _load_gcn_frame(
            path,
            frame_idx,
            desc,
            self._path_to_geom,
            hitfinder,
            self._last_geom_path_holder,
        )

        # --- Pad and shift centroids into padded coordinate frame ---
        # Image, valid-pixel mask, and a peak-protection map are stacked
        # (H, W, 3) so every geometric op (crop, rot90, flip) transforms all
        # three with the same random draws. The protection map marks hitfinder
        # centroids so cutout never occludes the Bragg evidence for label=1
        # (no coordinate transforms needed — the channel travels with the image).
        centroids = centroids + PAD_BORDER_DEFAULT
        padded_img = pad_border(assembled)
        peak_protect = np.zeros(padded_img.shape, dtype=np.float64)
        for x, y in centroids:
            r, c = int(round(float(y))), int(round(float(x)))
            if 0 <= r < peak_protect.shape[0] and 0 <= c < peak_protect.shape[1]:
                peak_protect[r, c] = 1.0
        padded = np.dstack(
            [padded_img, pad_border(valid_mask.astype(np.float64)), peak_protect]
        )

        ph, pw = padded.shape[:2]
        rng = np.random.default_rng(self._seed + idx)

        _CROP = 224

        # --- Guided crop ---
        if centroids.shape[0] > 0:
            # Metadata HIT frame with centroids found: 50/50 coin toss between
            # Path A (peak-centred, label=1) and Path B (hard-negative crop
            # sampled from this same hit frame's background, label=0). Path B
            # falls back to Path A if no 50px-clear position exists — the
            # frame has known peaks, so a valid Path A crop always exists.
            if rng.random() < self._hit_frac:
                crop, derived_label = _path_a_crop(
                    padded, centroids, rng, ph, pw, _CROP
                )
            else:
                crop = _sample_clear_crop(
                    padded,
                    centroids,
                    rng,
                    ph,
                    pw,
                    _CROP,
                    margin=self._hard_neg_margin,
                    max_tries=self._hard_neg_max_attempts,
                )
                if crop is None:
                    crop, derived_label = _path_a_crop(
                        padded, centroids, rng, ph, pw, _CROP
                    )
                else:
                    derived_label = 0
        else:
            # Path B: miss crop — random position with 50 px clearance from all peaks → label=0
            # (centroids is empty here either because the frame is metadata
            # NON-HIT and the hitfinder never ran, or the frame is metadata HIT
            # but the hitfinder found nothing — both fall through identically.)
            crop = _sample_clear_crop(
                padded,
                centroids,
                rng,
                ph,
                pw,
                _CROP,
                margin=self._hard_neg_margin,
                max_tries=self._hard_neg_max_attempts,
            )
            if crop is None:
                return None
            derived_label = 0

        # --- Augmentation + normalisation: rot90 → flip → LCN → cutout ---
        crop = random_rot90(crop, rng)
        crop = random_flip(crop, rng)

        crop_img = crop[:, :, 0]
        crop_mask = crop[:, :, 1] > 0.5
        crop_protect = crop[:, :, 2] > 0.5

        # Masked LCN before cutout (GCN already applied to full frame above),
        # so holes never enter the local statistics.
        crop_img = lcn(crop_img, mask=crop_mask)

        # Peak-aware cutout after LCN: holes are exact 0 in LCN space; positions
        # avoid the (co-transformed) peak-protection channel so Bragg evidence
        # is never erased.
        crop_img = random_cutout(crop_img, rng, avoid=crop_protect)

        tensor = torch.from_numpy(np.ascontiguousarray(crop_img)).unsqueeze(0).float()
        return tensor, derived_label


# ---------------------------------------------------------------------------
# SSLPretrainCXIDataset
# ---------------------------------------------------------------------------

SSL_CROP_MAX_TRIES = 50
SSL_MIN_VALID_FRAC_DEFAULT = 0.5
_SSL_CROP = 224
_SSL_PATCH = 16


def _centroid_map(rel_centroids: np.ndarray, size: int = _SSL_CROP) -> np.ndarray:
    """Rasterise crop-relative [x, y] centroids into a (size, size) 0/1 map."""
    m = np.zeros((size, size), dtype=np.float64)
    for x, y in rel_centroids:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < size and 0 <= yi < size:
            m[yi, xi] = 1.0
    return m


def _patch_ids_from_map(
    centroid_map: np.ndarray, patch: int = _SSL_PATCH
) -> np.ndarray:
    """Collapse a (crop, crop) centroid map into a flat bool array of ViT patch ids."""
    grid = centroid_map.shape[0] // patch
    out = np.zeros(grid * grid, dtype=bool)
    ys, xs = np.nonzero(centroid_map)
    for y, x in zip(ys, xs):
        out[(y // patch) * grid + (x // patch)] = True
    return out


class SSLPretrainCXIDataset(Dataset):
    """Unlabeled random valid-region crops for MAE pretraining (Track 2).

    Pipeline (fixed, bit-identical to Track 1 up to the crop step):
    read → assemble → [hitfinder on raw frame, only when a hitfinder is
    supplied for peak-aware masking] → GCN(full) → fill gaps → random 224×224
    crop with >= min_valid_frac valid pixels → rot90/flip → masked LCN.
    No cutout — MAE masking replaces it. No labels.

    Each item is a tuple of stacked tensors:
        crops     (crops_per_frame, 1, 224, 224) float32
        peaks     (crops_per_frame, 196)          bool
        vmasks    (crops_per_frame, 1, 224, 224)  float32
    _load_gcn_frame runs ONCE per frame; all crops share the assembled result.
    peak_patches marks ViT-S/16 patch ids containing hitfinder centroids
    (all-False when no hitfinder is supplied or the crop has no peaks).

    Args:
        session_ids: Session IDs to include (strict-LODO exclusion is done by
            the caller — pass only the training detectors' sessions).
        session_map: Maps session_id → Path to CXI file.
        seed: Base RNG seed; per-sample seed derives from seed and idx.
        crops_per_frame: Number of random crops drawn from each frame per epoch.
        hitfinder: Optional Hitfinder for peak-aware masking centroids.
        min_valid_frac: Minimum fraction of valid pixels a crop must contain;
            after SSL_CROP_MAX_TRIES rejected draws the best candidate is used.
    """

    def __init__(
        self,
        session_ids: list[str],
        session_map: dict[str, str | Path],
        seed: int = 42,
        crops_per_frame: int = 1,
        hitfinder: Hitfinder | None = None,
        min_valid_frac: float = SSL_MIN_VALID_FRAC_DEFAULT,
    ) -> None:
        self._hitfinder = hitfinder
        self._seed = seed
        self._epoch: int = 0
        self._min_valid_frac = min_valid_frac
        self._crops_per_frame = crops_per_frame
        self._last_geom_path_holder: list = [None]

        cxi_paths = [Path(session_map[sid]) for sid in session_ids]

        # Descriptor cache populated lazily in __getitem__ (per DataLoader worker)
        # so that HDF5 opens do not happen in __init__ before worker fork.
        self._path_to_desc: dict[Path, str | None] = {}
        self._path_to_geom: dict[Path, dict[str, float]] = {}

        # Per-frame index — one entry per frame regardless of crops_per_frame.
        # _load_gcn_frame (HDF5 + Reborn assembly + GCN) is called once per frame;
        # all crops are drawn inside __getitem__ from the already-assembled result.
        self._index: list[tuple[Path, int]] = []
        for p in cxi_paths:
            try:
                n = count_frames(p)
            except (KeyError, OSError) as e:
                warnings.warn(
                    f"SSLPretrainCXIDataset: cannot count frames in {p}: {e}; "
                    "file skipped.",
                    stacklevel=2,
                )
                continue
            for i in range(n):
                self._index.append((p, i))

    def __len__(self) -> int:
        return len(self._index)

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter so each epoch draws different crops.

        Call this at the start of every training epoch (before iterating the
        DataLoader) so that per-frame RNG seeds vary across epochs and the model
        sees different 224×224 windows each pass.
        """
        self._epoch = epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, frame_idx = self._index[idx]
        # Fold epoch into seed so crop positions vary across epochs.
        rng = np.random.default_rng(
            self._seed * 1_000_003 + self._epoch * len(self._index) + idx
        )

        if path not in self._path_to_desc:
            try:
                self._path_to_desc[path] = read_detector_description(path)
            except (ValueError, KeyError, OSError):
                self._path_to_desc[path] = None

        # Assembly + GCN run ONCE per frame regardless of crops_per_frame.
        try:
            gcn_frame, valid_mask, centroids = _load_gcn_frame(
                path,
                frame_idx,
                self._path_to_desc.get(path),
                self._path_to_geom,
                self._hitfinder,
                self._last_geom_path_holder,
            )
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            warnings.warn(
                f"SSLPretrainCXIDataset: skipping frame {frame_idx} of {path}: {exc}",
                stacklevel=2,
            )
            n_patches = (_SSL_CROP // _SSL_PATCH) ** 2
            zeros_img = torch.zeros(self._crops_per_frame, 1, _SSL_CROP, _SSL_CROP)
            zeros_peaks = torch.zeros(
                self._crops_per_frame, n_patches, dtype=torch.bool
            )
            zeros_vmask = torch.zeros(self._crops_per_frame, 1, _SSL_CROP, _SSL_CROP)
            return zeros_img, zeros_peaks, zeros_vmask
        fh, fw = gcn_frame.shape

        crop_tensors: list[torch.Tensor] = []
        peak_tensors: list[torch.Tensor] = []
        vmask_tensors: list[torch.Tensor] = []

        for _ in range(self._crops_per_frame):
            best: tuple[float, int, int] | None = None
            for _attempt in range(SSL_CROP_MAX_TRIES):
                top = int(rng.integers(0, max(fh - _SSL_CROP, 0) + 1))
                left = int(rng.integers(0, max(fw - _SSL_CROP, 0) + 1))
                frac = float(
                    valid_mask[top : top + _SSL_CROP, left : left + _SSL_CROP].mean()
                )
                if best is None or frac > best[0]:
                    best = (frac, top, left)
                if frac >= self._min_valid_frac:
                    break
            _, top, left = best

            crop_img = gcn_frame[top : top + _SSL_CROP, left : left + _SSL_CROP]
            crop_mask = valid_mask[top : top + _SSL_CROP, left : left + _SSL_CROP]

            # Centroid channel travels through rot90/flip so patch ids stay correct
            # under augmentation (same co-transform idiom as AsymmetricCXIDataset).
            rel = centroids - np.array([left, top], dtype=np.float64)
            stacked = np.dstack(
                [crop_img, crop_mask.astype(np.float64), _centroid_map(rel)]
            )
            stacked = random_rot90(stacked, rng)
            stacked = random_flip(stacked, rng)

            img = lcn(stacked[:, :, 0], mask=stacked[:, :, 1] > 0.5)
            peak_patches = _patch_ids_from_map(stacked[:, :, 2] > 0.5)
            crop_valid = (stacked[:, :, 1] > 0.5).astype(np.float32)

            crop_tensors.append(
                torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
            )
            peak_tensors.append(torch.from_numpy(peak_patches))
            vmask_tensors.append(
                torch.from_numpy(np.ascontiguousarray(crop_valid)).unsqueeze(0).float()
            )

        return (
            torch.stack(crop_tensors),  # (N, 1, 224, 224)
            torch.stack(peak_tensors),  # (N, 196)
            torch.stack(vmask_tensors),  # (N, 1, 224, 224)
        )
