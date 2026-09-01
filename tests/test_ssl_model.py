"""Tests for src/models/ssl.py — MAE ViT-S/16, masking pipelines, loss."""

import numpy as np
import pytest
import torch

from src.models.ssl import (
    MAE_MASK_RATIO_DEFAULT,
    peak_aware_masking,
    random_masking,
)


class TestRandomMasking:
    def test_default_ratio_is_lower_than_mae_standard(self):
        # Design decision 2026-09-01: default 0.6 (risk B3), not 0.75.
        assert MAE_MASK_RATIO_DEFAULT == 0.6

    def test_shapes_and_counts(self):
        B, L = 4, 196
        g = torch.Generator().manual_seed(0)
        ids_keep, mask, ids_restore = random_masking(B, L, mask_ratio=0.6, generator=g)
        len_keep = int(L * (1 - 0.6))
        assert ids_keep.shape == (B, len_keep)
        assert mask.shape == (B, L)
        assert ids_restore.shape == (B, L)
        # mask: 1 = masked, 0 = kept
        assert torch.all(mask.sum(dim=1) == L - len_keep)

    def test_mask_consistent_with_ids_keep(self):
        B, L = 2, 196
        g = torch.Generator().manual_seed(1)
        ids_keep, mask, _ = random_masking(B, L, mask_ratio=0.5, generator=g)
        for b in range(B):
            kept = set(ids_keep[b].tolist())
            for i in range(L):
                assert (i in kept) == (mask[b, i].item() == 0)

    def test_deterministic_with_generator(self):
        a = random_masking(2, 196, 0.6, generator=torch.Generator().manual_seed(7))
        b = random_masking(2, 196, 0.6, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class TestPeakAwareMasking:
    def test_peak_patches_are_masked(self):
        B, L = 2, 196
        peak_patches = torch.zeros(B, L, dtype=torch.bool)
        peak_patches[0, [3, 50, 120]] = True
        peak_patches[1, [7]] = True
        g = torch.Generator().manual_seed(0)
        ids_keep, mask, ids_restore = peak_aware_masking(
            peak_patches, mask_ratio=0.6, peak_mask_frac=1.0, generator=g
        )
        # every peak patch must be masked
        assert torch.all(mask[peak_patches] == 1)
        # exact masked count preserved
        len_keep = int(L * (1 - 0.6))
        assert torch.all(mask.sum(dim=1) == L - len_keep)

    def test_no_peaks_falls_back_to_random(self):
        B, L = 3, 196
        empty = torch.zeros(B, L, dtype=torch.bool)
        ids_keep, mask, _ = peak_aware_masking(
            empty,
            mask_ratio=0.6,
            peak_mask_frac=1.0,
            generator=torch.Generator().manual_seed(0),
        )
        len_keep = int(L * (1 - 0.6))
        assert torch.all(mask.sum(dim=1) == L - len_keep)

    def test_peak_mask_frac_half(self):
        B, L = 1, 196
        peak_patches = torch.zeros(B, L, dtype=torch.bool)
        peak_patches[0, :10] = True  # 10 peak patches
        _, mask, _ = peak_aware_masking(
            peak_patches,
            mask_ratio=0.6,
            peak_mask_frac=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        # at least ceil(0.5 * 10) = 5 peak patches masked
        assert int(mask[0, :10].sum().item()) >= 5

    def test_more_peaks_than_budget_caps_at_budget(self):
        B, L = 1, 196
        peak_patches = torch.ones(B, L, dtype=torch.bool)  # everything is a peak
        _, mask, _ = peak_aware_masking(
            peak_patches,
            mask_ratio=0.25,
            peak_mask_frac=1.0,
            generator=torch.Generator().manual_seed(0),
        )
        len_keep = int(L * (1 - 0.25))
        assert torch.all(mask.sum(dim=1) == L - len_keep)  # never exceeds budget
