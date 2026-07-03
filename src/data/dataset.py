"""PyTorch Datasets for SFX diffraction images. HDF5 opened lazily in __getitem__."""

from __future__ import annotations

import json
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
    random_crop,
    random_cutout,
    random_flip,
    random_rot90,
)
from src.preprocessing.normalize import gcn, lcn
from src.preprocessing.pipeline import (
    _to_2d,
    assemble_only,
    preprocess_assembled,
    preprocess_eval,
    preprocess_train,
    preprocess_with_geometry,
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


class SFXDataset(Dataset):
    """Dataset of labeled diffraction images for supervised training.

    Reads a plaintext split file (one absolute image path per line) and a
    JSON labels file mapping absolute path strings to integer labels
    (1 = hit, 0 = non-hit).

    Labels file format (labels.json):
        {
            "/absolute/path/to/frame_001.h5": 1,
            "/absolute/path/to/frame_002.h5": 0
        }

    Detector type is always read from file metadata — never inferred.
    HDF5 files are opened lazily in __getitem__ to support multiprocessing.
    """

    def __init__(self, split_file: str | Path, labels_file: str | Path) -> None:
        split_file = Path(split_file)
        self._paths = [
            Path(line.strip())
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]
        labels_file = Path(labels_file)
        self._labels: dict[str, int] = json.loads(labels_file.read_text())

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = read_image(self._paths[idx])
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = self._load_label(idx)
        return tensor, label

    def _load_label(self, idx: int) -> int:
        key = str(self._paths[idx])
        if key not in self._labels:
            raise KeyError(
                f"No label for '{key}'. Verify the path appears in labels_file."
            )
        return int(self._labels[key])


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
            frame before converting to a tensor. Defaults to
            preprocess_assembled (GCN → LCN → resize 224×224). Pass None to
            return raw frames.
    """

    def __init__(
        self,
        cxi_paths: list[str | Path],
        label_key: str = "entry_1/labels/hit",
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = preprocess_assembled,
        augment: bool = False,
        n_cutout_holes: int = 3,
        cutout_hole_size: int = 32,
    ) -> None:
        self._preprocess_fn = preprocess_fn
        self._augment = augment
        self._n_cutout_holes = n_cutout_holes
        self._cutout_hole_size = cutout_hole_size
        # Checked once at init so __getitem__ does not use `is` identity, which
        # breaks for any wrapped/partial version of preprocess_assembled.
        self._use_geometry = preprocess_fn is preprocess_assembled

        # Read detector descriptions eagerly so __getitem__ can route to
        # geometry-aware preprocessing. Geometry objects (PADGeometryList,
        # PADAssembler) are NOT stored as instance attributes — they are not
        # picklable under spawn/forkserver DataLoader workers. Instead we call
        # get_geometry/get_assembler lazily in __getitem__; both functions use
        # a module-level cache that is process-local and safe under any start method.
        unique_paths = {Path(p) for p in cxi_paths}
        self._path_to_desc: dict[Path, str] = {}
        for p in unique_paths:
            try:
                desc = read_detector_description(p)
                self._path_to_desc[p] = desc
            except (ValueError, KeyError, OSError):
                # File lacks description key or is unreadable — falls back to
                # preprocess_assembled.
                pass

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

        if self._augment:
            # Augmentation path: assemble to native resolution, then crop/rotate/flip/norm/cutout.
            # Jungfrau 4M has no desc entry (pre-assembled) — _to_2d gives the 2164×2068 canvas.
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
            rng = np.random.default_rng()
            frame = preprocess_train(
                assembled,
                rng,
                n_cutout_holes=self._n_cutout_holes,
                cutout_hole_size=self._cutout_hole_size,
            )
        elif self._preprocess_fn is not None:
            # Existing eval path (unchanged for non-augment use).
            if self._use_geometry and path in self._path_to_desc:
                desc = self._path_to_desc[path]
                try:
                    pads = get_geometry(desc)
                    assembler = get_assembler(desc)
                    frame = preprocess_with_geometry(
                        frame, pads, desc, assembler=assembler
                    )
                except (ValueError, KeyError, OSError):
                    frame = preprocess_assembled(frame)
            else:
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
    inside = (
        (xs >= left) & (xs < left + size) & (ys >= top) & (ys < top + size)
    )
    return bool(inside.any())


# ---------------------------------------------------------------------------
# AsymmetricCXIDataset
# ---------------------------------------------------------------------------


class AsymmetricCXIDataset(Dataset):
    """Primary training dataset: crop-content-labeled patches with hitfinder integration.

    Replaces MultiFrameCXIDataset as the main training path. For each frame:
      1. Read + assemble to native resolution
      2. Run hitfinder → centroids (N_peaks, 2)
      3. Sample a 224×224 crop with class-balanced labeling:
         - frame_label=1, rand < hit_frac: sample crops until ≥1 centroid inside → label=1
           (fallback to label=0 if hard_neg_max_attempts exhausted or 0 peaks)
         - frame_label=1, rand >= hit_frac: sample crop with 0 centroids inside → label=0 (hard neg)
         - frame_label=0: random crop → label=0
      4. Augment: random_rot90 → random_flip → gcn(patch) → lcn(patch) → random_cutout
      5. Return (tensor(1,224,224) float32, int_label)

    Class balance: hit_frac=0.5 → 50% label-1, 50% label-0 (split: 25% true misses from
    non-hit frames + 25% hard negatives from hit frames).

    Args:
        session_ids: Session IDs to include.
        session_map: Maps session_id → Path to CXI file.
        hitfinder: Hitfinder Protocol instance (find_peaks method).
        label_key: HDF5 key for per-frame labels.
        patch_size: Crop side length in pixels.
        lcn_window: LCN window size (must be odd; default 9).
        hit_frac: Fraction of items targeting label=1 patches (default 0.5).
        hard_neg_max_attempts: Max random crop attempts for hit/hard-neg sampling.
        n_cutout_holes: Cutout augmentation holes.
        cutout_hole_size: Cutout hole side length.
    """

    def __init__(
        self,
        session_ids: list[str],
        session_map: dict[str, str | Path],
        hitfinder: Hitfinder,
        label_key: str = "entry_1/labels/hit",
        patch_size: int = 224,
        lcn_window: int = 9,
        hit_frac: float = 0.5,
        hard_neg_max_attempts: int = 50,
        n_cutout_holes: int = 3,
        cutout_hole_size: int = 32,
    ) -> None:
        self._hitfinder = hitfinder
        self._label_key = label_key
        self._patch_size = patch_size
        self._lcn_window = lcn_window
        self._hit_frac = hit_frac
        self._hard_neg_max_attempts = hard_neg_max_attempts
        self._n_cutout_holes = n_cutout_holes
        self._cutout_hole_size = cutout_hole_size

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
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

        # --- Run hitfinder ---
        centroids = self._hitfinder.find_peaks(assembled)  # (N, 2) float32

        frame_label = self._labels[idx]
        size = self._patch_size
        h, w = assembled.shape

        # Guard: if the assembled image is smaller than patch_size, pad at
        # bottom/right only. Padding at the end leaves existing (x, y) centroid
        # coordinates valid in the padded image — coordinate origins are unchanged.
        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            assembled = np.pad(
                assembled,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0.0,
            )
            h, w = assembled.shape

        # --- Sample crop with class-balanced labeling ---
        # Fresh RNG per call: shared state would produce correlated crops across
        # DataLoader fork()ed workers on Linux.
        rng = np.random.default_rng()
        patch: np.ndarray
        int_label: int

        if frame_label == 1 and rng.random() < self._hit_frac:
            # Try to find a crop containing ≥1 centroid → label=1
            found = False
            if centroids.shape[0] > 0:
                for _ in range(self._hard_neg_max_attempts):
                    top = int(rng.integers(0, h - size + 1))
                    left = int(rng.integers(0, w - size + 1))
                    if _crop_contains_centroid(top, left, size, centroids):
                        patch = assembled[top : top + size, left : left + size]
                        int_label = 1
                        found = True
                        break
            if not found:
                # Fallback: random crop, label=0
                top = int(rng.integers(0, h - size + 1))
                left = int(rng.integers(0, w - size + 1))
                patch = assembled[top : top + size, left : left + size]
                int_label = 0

        elif frame_label == 1:
            # Hard negative: try to find a crop with NO centroid inside → label=0
            found_neg = False
            for _ in range(self._hard_neg_max_attempts):
                top = int(rng.integers(0, h - size + 1))
                left = int(rng.integers(0, w - size + 1))
                if not _crop_contains_centroid(top, left, size, centroids):
                    patch = assembled[top : top + size, left : left + size]
                    found_neg = True
                    break
            if not found_neg:
                top = int(rng.integers(0, h - size + 1))
                left = int(rng.integers(0, w - size + 1))
                patch = assembled[top : top + size, left : left + size]
            int_label = 0

        else:
            # Non-hit frame: random crop → label=0
            top = int(rng.integers(0, h - size + 1))
            left = int(rng.integers(0, w - size + 1))
            patch = assembled[top : top + size, left : left + size]
            int_label = 0

        # --- Augment ---
        patch = random_rot90(patch, rng)
        patch = random_flip(patch, rng)
        patch = gcn(patch)
        patch = lcn(patch, window=self._lcn_window)
        patch = random_cutout(
            patch, rng, n_holes=self._n_cutout_holes, hole_size=self._cutout_hole_size
        )

        tensor = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float()
        return tensor, int_label
