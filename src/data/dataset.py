"""PyTorch Datasets for SFX diffraction images. HDF5 opened lazily in __getitem__."""

from __future__ import annotations

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
    """Dataset of labeled HDF5/CXI diffraction images for supervised training.

    Reads a plaintext split file listing one absolute image path per line.
    Detector type is always read from file metadata — never inferred.

    Label loading is not yet implemented: label format (JSON sidecar vs.
    embedded HDF5 dataset) is an open Phase 2 decision. Calling __getitem__
    raises NotImplementedError until resolved.
    """

    def __init__(self, split_file: str | Path) -> None:
        split_file = Path(split_file)
        self._paths = [
            Path(line.strip())
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = read_image(self._paths[idx])
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = self._load_label(idx)
        return tensor, label

    def _load_label(self, idx: int) -> int:
        raise NotImplementedError(
            "Label loading not yet implemented. "
            "Resolve label format (JSON sidecar vs. embedded HDF5) before Phase 3."
        )
