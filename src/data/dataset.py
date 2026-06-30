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
from src.preprocessing.pipeline import preprocess_assembled, preprocess_with_geometry


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
    """Dataset over multi-frame CXI files with embedded hit labels.

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
    ) -> None:
        self._preprocess_fn = preprocess_fn
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
        if self._preprocess_fn is not None:
            if self._use_geometry and path in self._path_to_desc:
                desc = self._path_to_desc[path]
                try:
                    pads = get_geometry(desc)
                    assembler = get_assembler(desc)
                    frame = preprocess_with_geometry(frame, pads, desc, assembler=assembler)
                except (ValueError, KeyError):
                    frame = preprocess_assembled(frame)  # e.g. Jungfrau 4M — already assembled
            else:
                frame = self._preprocess_fn(frame)
        tensor = torch.from_numpy(frame).unsqueeze(0)
        return tensor, self._labels[idx]
