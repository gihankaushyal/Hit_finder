"""MAE ViT encoder + classification head. Track 2 — self-supervised. NOT ResNet.

Design: docs/superpowers/specs/2026-09-01-phase5-ssl-mae-design.md
Masking pipelines (YAML `ssl.masking`): "random" (default, ratio 0.6 — risk B3)
and "peak_aware" (hitfinder-centroid-biased).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MAE_MASK_RATIO_DEFAULT: float = 0.6
PEAK_MASK_FRAC_DEFAULT: float = 1.0
# Bias added to shuffle noise so selected peak patches sort into the masked tail.
_PEAK_NOISE_BIAS: float = 2.0


def _masking_from_noise(
    noise: torch.Tensor, mask_ratio: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn per-token noise into (ids_keep, mask, ids_restore).

    Tokens with the SMALLEST noise are kept (MAE convention).
    mask is (B, L) with 1 = masked, 0 = kept.
    """
    B, L = noise.shape
    len_keep = int(L * (1 - mask_ratio))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones(B, L, device=noise.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return ids_keep, mask, ids_restore


def random_masking(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = MAE_MASK_RATIO_DEFAULT,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAE-standard per-sample random masking."""
    noise = torch.rand(batch_size, seq_len, device=device, generator=generator)
    return _masking_from_noise(noise, mask_ratio)


def peak_aware_masking(
    peak_patches: torch.Tensor,
    mask_ratio: float = MAE_MASK_RATIO_DEFAULT,
    peak_mask_frac: float = PEAK_MASK_FRAC_DEFAULT,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random masking biased so hitfinder-peak patches land in the masked set.

    peak_patches: (B, L) bool — True where the token's patch contains a
    hitfinder centroid. Per sample, round(peak_mask_frac * n_peaks) peak
    patches are force-masked (capped at the mask budget); remaining budget is
    filled randomly. Samples with no peaks degrade to plain random masking.
    """
    B, L = peak_patches.shape
    device = peak_patches.device
    noise = torch.rand(B, L, device=device, generator=generator)
    budget = L - int(L * (1 - mask_ratio))
    for b in range(B):
        peak_idx = torch.nonzero(peak_patches[b], as_tuple=False).flatten()
        n_force = min(int(round(peak_mask_frac * len(peak_idx))), budget)
        if n_force == 0:
            continue
        perm = torch.randperm(len(peak_idx), generator=generator)[:n_force]
        noise[b, peak_idx[perm]] += _PEAK_NOISE_BIAS
    return _masking_from_noise(noise, mask_ratio)
