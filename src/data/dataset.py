"""PyTorch Datasets for SFX diffraction images. HDF5 opened lazily in __getitem__."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.preprocessing.io import read_image


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
