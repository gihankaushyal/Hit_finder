"""PyTorch Datasets for SFX diffraction images. HDF5 opened lazily in __getitem__."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
from src.preprocessing.normalize import gcn_apply, lcn
from src.preprocessing.pipeline import _to_2d, assemble_only



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


# ---------------------------------------------------------------------------
# AsymmetricCXIDataset
# ---------------------------------------------------------------------------


class AsymmetricCXIDataset(Dataset):
    """Primary training dataset: hitfinder-guided crop with augmentation and normalisation.

    For each frame:
      1. Read + assemble to native resolution
      2. Capture full-frame GCN statistics (μ, σ) before padding
      3. Run hitfinder → centroids (N_peaks, 2)
      4. Pad frame by PAD_BORDER_DEFAULT px on each edge; shift centroids
      5. Guided crop (224×224) → derived label:
           Path A (peaks found): crop centred on a random Bragg peak → label=1
           Path B (no peaks):    random crop with 50 px clearance from all peaks → label=0
      6. Augment: random_rot90 → random_flip → random_cutout
      7. Normalise: GCN with full-frame μ/σ → LCN
      8. Return (tensor(1, 224, 224) float32, derived_label)

    Args:
        session_ids: Session IDs to include.
        session_map: Maps session_id → Path to CXI file.
        hitfinder: Hitfinder Protocol instance (find_peaks method).
        label_key: HDF5 key for per-frame labels (used only to build the flat index).
        seed: Base RNG seed; per-sample seed is seed+idx for reproducibility.
    """

    def __init__(
        self,
        session_ids: list[str],
        session_map: dict[str, str | Path],
        hitfinder: Hitfinder,
        label_key: str = "entry_1/labels/hit",
        seed: int = 42,
    ) -> None:
        self._hitfinder = hitfinder
        self._label_key = label_key
        self._seed = seed

        # Resolve session_map to Path objects for requested session_ids only.
        cxi_paths: list[Path] = []
        for sid in session_ids:
            cxi_paths.append(Path(session_map[sid]))

        # Read detector descriptions eagerly — geometry objects are NOT stored
        # (not picklable); get_geometry/get_assembler use a process-local cache.
        unique_paths = set(cxi_paths)
        self._path_to_desc: dict[Path, str] = {}
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
            arr = read_embedded_labels(p, label_key)
            for i, raw in enumerate(arr):
                self._index.append((p, i))
                self._labels.append(int(round(float(raw))))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | None:
        path, frame_idx = self._index[idx]
        frame = read_frame(path, frame_idx)

        # --- Assemble to native resolution ---
        if path in self._path_to_desc:
            desc = self._path_to_desc[path]
            try:
                pads = get_geometry(desc)
                assembler = get_assembler(desc)
                assembled = assemble_only(frame, pads, desc, assembler=assembler)
            except (ValueError, KeyError, OSError):
                assembled = _to_2d(frame)
        else:
            assembled = _to_2d(frame)

        # Full-frame GCN stats captured before padding for consistent scale across crops.
        gcn_mu = float(assembled.mean())
        gcn_sigma = float(assembled.std())

        # --- Run hitfinder ---
        centroids = self._hitfinder.find_peaks(assembled)  # (N, 2) float32 [x, y]

        # --- Pad and shift centroids into padded coordinate frame ---
        padded = pad_border(assembled)
        centroids = centroids + PAD_BORDER_DEFAULT

        ph, pw = padded.shape
        rng = np.random.default_rng(self._seed + idx)

        _CROP = 224

        # --- Guided crop ---
        if centroids.shape[0] > 0:
            # Path A: hit crop centred on a randomly chosen Bragg peak → label=1
            peak = centroids[int(rng.integers(0, len(centroids)))]
            cx = int(round(float(peak[0])))  # column
            cy = int(round(float(peak[1])))  # row
            left = int(np.clip(cx - _CROP // 2, 0, pw - _CROP))
            top = int(np.clip(cy - _CROP // 2, 0, ph - _CROP))
            crop = padded[top : top + _CROP, left : left + _CROP].copy()
            derived_label = 1
        else:
            # Path B: miss crop — random position with 50 px clearance from all peaks → label=0
            crop = None
            for _ in range(50):
                top = int(rng.integers(0, ph - _CROP + 1))
                left = int(rng.integers(0, pw - _CROP + 1))
                if not _crop_within_margin(top, left, _CROP, centroids, margin=50):
                    crop = padded[top : top + _CROP, left : left + _CROP].copy()
                    break
            if crop is None:
                return None
            derived_label = 0

        # --- Augmentation: rot90 → flip → cutout ---
        crop = random_rot90(crop, rng)
        crop = random_flip(crop, rng)
        crop = random_cutout(crop, rng)

        # --- Normalisation: full-frame GCN stats → LCN ---
        crop = gcn_apply(crop, gcn_mu, gcn_sigma)
        crop = lcn(crop)

        tensor = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0).float()
        return tensor, derived_label
