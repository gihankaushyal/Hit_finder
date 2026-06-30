"""DataLoader factories for SSL pretraining and supervised training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultiFrameCXIDataset, SFXDataset, UnlabeledDataset


def ssl_pretrain_loader(
    file_list: list[str | Path],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader over unlabeled images for MAE pretraining.

    Args:
        file_list: Paths to .img, .h5, or .cxi files.
        batch_size: Number of images per batch.
        num_workers: Parallel worker processes for data loading.
        shuffle: Whether to shuffle the dataset each epoch.

    Returns:
        DataLoader yielding float32 tensors of shape (B, 1, H, W).
    """
    dataset = UnlabeledDataset(file_list)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def cxi_session_loader(
    session_map: dict[str, Path],
    session_ids: list[str],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    label_key: str = "entry_1/labels/hit",
    augment: bool = False,
    n_cutout_holes: int = 3,
    cutout_hole_size: int = 32,
) -> DataLoader:
    """DataLoader over a subset of CXI sessions identified by session_id.

    Used by the LODO dataloader_factory closure in scripts/train_lodo.py.

    Args:
        session_map: Mapping from session_id to CXI file path.
        session_ids: Subset of session_ids to include in this loader.
        batch_size: Number of frames per batch.
        num_workers: Parallel worker processes.
        shuffle: Whether to shuffle each epoch.
        label_key: HDF5 key for per-frame labels; must match the key used
            in build_sessions() so frame counts and label reads are consistent.
        augment: If True, apply train-time augmentation (random crop, rot90, flip,
            cutout). If False (default), use deterministic centre-crop eval path.
        n_cutout_holes: Number of cutout patches when augment=True.
        cutout_hole_size: Side length of each cutout patch in pixels.

    Returns:
        DataLoader yielding (image, label) pairs; image shape (B, 1, 224, 224).
    """
    paths = [session_map[sid] for sid in session_ids]
    dataset = MultiFrameCXIDataset(
        paths,
        label_key=label_key,
        augment=augment,
        n_cutout_holes=n_cutout_holes,
        cutout_hole_size=cutout_hole_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def supervised_loader(
    split_file: str | Path,
    labels_file: str | Path,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader over labeled images for supervised training.

    Args:
        split_file: Plaintext file listing absolute image paths, one per line.
        labels_file: JSON file mapping absolute path strings to int labels
            (1 = hit, 0 = non-hit).
        batch_size: Number of images per batch.
        num_workers: Parallel worker processes for data loading.
        shuffle: Whether to shuffle the dataset each epoch.

    Returns:
        DataLoader yielding (image, label) pairs; image shape (B, 1, H, W).
    """
    dataset = SFXDataset(split_file, labels_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
