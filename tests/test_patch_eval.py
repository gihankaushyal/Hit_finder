"""Tests for patch-grid tiling and patch-aggregation evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.augment import patch_grid


class TestPatchGrid:
    def test_non_overlapping_count_exact_fit(self):
        # 448 = 2 × 224, so 2×2 = 4 patches exactly
        img = np.zeros((448, 448), dtype=np.float32)
        patches = patch_grid(img, patch_size=224, stride=224)
        assert len(patches) == 4

    def test_non_overlapping_count_partial_edge(self):
        # floor(1273 / 224) = 5 → 5×5 = 25 patches; 153 px edge discarded
        img = np.zeros((1273, 1273), dtype=np.float32)
        patches = patch_grid(img, patch_size=224, stride=224)
        assert len(patches) == 25

    def test_default_stride_equals_patch_size(self):
        img = np.zeros((1273, 1273), dtype=np.float32)
        assert len(patch_grid(img, 224)) == len(patch_grid(img, 224, 224))

    def test_each_patch_is_correct_size(self):
        img = np.zeros((1273, 1273), dtype=np.float32)
        for p in patch_grid(img, 224, 224):
            assert p.shape == (224, 224)

    def test_patches_contain_correct_pixel_values(self):
        img = np.arange(448 * 448, dtype=np.float32).reshape(448, 448)
        patches = patch_grid(img, 224, 224)
        np.testing.assert_array_equal(patches[0], img[0:224, 0:224])
        np.testing.assert_array_equal(patches[1], img[0:224, 224:448])

    def test_overlapping_stride_increases_count(self):
        img = np.zeros((500, 500), dtype=np.float32)
        assert len(patch_grid(img, 224, 100)) > len(patch_grid(img, 224, 224))

    def test_image_smaller_than_patch_returns_empty(self):
        img = np.zeros((100, 100), dtype=np.float32)
        assert patch_grid(img, 224, 224) == []

    def test_row_major_order(self):
        img = np.zeros((448, 448), dtype=np.float32)
        img[0, 0] = 1.0  # top-left → patch index 0
        img[224, 0] = 2.0  # bottom-left → patch index 2
        patches = patch_grid(img, 224, 224)
        assert patches[0][0, 0] == 1.0
        assert patches[2][0, 0] == 2.0
