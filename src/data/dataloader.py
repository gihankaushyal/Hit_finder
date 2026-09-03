"""DataLoader factories for SSL pretraining and supervised training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    SSL_MIN_VALID_FRAC_DEFAULT,
    AsymmetricCXIDataset,
    MultiFrameCXIDataset,
    SSLPretrainCXIDataset,
    UnlabeledDataset,
)
from src.hitfinders.base import Hitfinder


def none_collate_fn(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Collate function that silently drops None items from the batch.

    AsymmetricCXIDataset.__getitem__ returns None when no valid miss crop can be
    found after 50 random attempts. This collate function filters those out so
    the DataLoader can continue without error. Returns None for an all-None batch
    (the training loop must skip it with ``if batch is None: continue``).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


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


def asymmetric_loader(
    session_map: dict[str, Path],
    session_ids: list[str],
    hitfinder: Hitfinder,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    label_key: str = "entry_1/labels/hit",
) -> DataLoader:
    """DataLoader for asymmetric hitfinder-guided training.

    Each item is a (1, 224, 224) float32 tensor and a crop-derived binary label
    (1 = hit crop centred on a Bragg peak, 0 = miss crop with 50 px clearance).
    None items (rare fallback when no valid miss crop is found) are filtered by
    none_collate_fn; the training loop must skip any None batch.

    Args:
        session_map: Maps session_id → CXI file path.
        session_ids: Session IDs to include.
        hitfinder: Hitfinder instance (PF8Hitfinder or GPUHitfinder).
            GPU hitfinder requires num_workers=0 (no fork-safe GPU context).
        batch_size: Crops per batch.
        num_workers: DataLoader worker processes. Must be 0 for GPU hitfinder.
        shuffle: Shuffle each epoch.
        label_key: HDF5 key for per-frame labels (used only to build the index).

    Returns:
        DataLoader yielding (tensor(B,1,224,224), label(B,)) pairs.
    """
    from src.hitfinders.gpu import GPUHitfinder

    if isinstance(hitfinder, GPUHitfinder) and num_workers > 0:
        import warnings

        warnings.warn(
            "GPUHitfinder requires num_workers=0 (CUDA context cannot be shared "
            "across forked DataLoader workers). Overriding num_workers to 0.",
            UserWarning,
            stacklevel=2,
        )
        num_workers = 0

    dataset = AsymmetricCXIDataset(
        session_ids=session_ids,
        session_map=session_map,
        hitfinder=hitfinder,
        label_key=label_key,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=none_collate_fn,
    )


def _ssl_flatten_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten per-frame multi-crop batches into a flat crop batch.

    Each dataset item is (N,1,H,W), (N,L), (N,1,H,W) where N=crops_per_frame.
    torch.cat across the batch dimension yields (B*N, 1, H, W) so the training
    loop sees the same shape as before regardless of crops_per_frame.
    """
    return (
        torch.cat([b[0] for b in batch], dim=0),
        torch.cat([b[1] for b in batch], dim=0),
        torch.cat([b[2] for b in batch], dim=0),
    )


def ssl_crop_loader(
    session_map: dict[str, Path],
    session_ids: list[str],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
    crops_per_frame: int = 1,
    hitfinder: Hitfinder | None = None,
    min_valid_frac: float = SSL_MIN_VALID_FRAC_DEFAULT,
) -> DataLoader:
    """DataLoader for MAE pretraining crops (SSLPretrainCXIDataset).

    batch_size controls the number of crops per GPU batch (after flatten collate).
    The DataLoader uses loader_batch = batch_size // crops_per_frame frames per
    batch so that collate expands each back to batch_size crops.

    GPU hitfinder backends require num_workers=0 — CUDA contexts cannot be
    forked into DataLoader worker processes (enforced by the caller, same
    convention as asymmetric_loader).
    """
    ds = SSLPretrainCXIDataset(
        session_ids=session_ids,
        session_map=session_map,
        seed=seed,
        crops_per_frame=crops_per_frame,
        hitfinder=hitfinder,
        min_valid_frac=min_valid_frac,
    )
    loader_batch = max(1, batch_size // crops_per_frame)
    return DataLoader(
        ds,
        batch_size=loader_batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_ssl_flatten_collate,
    )
