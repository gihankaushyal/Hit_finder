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


from src.preprocessing.pipeline import preprocess_eval_patches


class TestPreprocessEvalPatches:
    def test_output_shape_agipd_size(self):
        # floor(1273/224)=5 → 5×5=25 patches
        img = np.random.default_rng(42).random((1273, 1273)).astype(np.float32)
        out = preprocess_eval_patches(img)
        assert out.shape == (25, 224, 224)

    def test_output_dtype_float32(self):
        img = np.zeros((500, 500), dtype=np.float32)
        out = preprocess_eval_patches(img, patch_size=224, stride=224)
        assert out.dtype == np.float32

    def test_raises_if_no_complete_patch(self):
        img = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="no complete"):
            preprocess_eval_patches(img)

    def test_custom_stride_changes_count(self):
        img = np.random.default_rng(0).random((500, 500)).astype(np.float32)
        default_out = preprocess_eval_patches(img, stride=224)
        overlap_out = preprocess_eval_patches(img, stride=112)
        assert overlap_out.shape[0] > default_out.shape[0]

    def test_each_patch_has_zero_mean_after_gcn(self):
        img = np.random.default_rng(7).random((500, 500)).astype(np.float32) * 1000
        out = preprocess_eval_patches(img)
        patch_means = out.reshape(out.shape[0], -1).mean(axis=1)
        np.testing.assert_allclose(patch_means, 0.0, atol=2e-4)

    def test_output_is_finite(self):
        img = np.random.default_rng(99).random((1273, 1273)).astype(np.float32)
        out = preprocess_eval_patches(img)
        assert np.isfinite(out).all()
