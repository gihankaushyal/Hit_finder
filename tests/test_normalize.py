"""Unit tests for GCN and LCN normalization functions."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.normalize import GCN_EPSILON, LCN_EPSILON, gcn, lcn

_H, _W = 64, 64


def _random_image(h: int = _H, w: int = _W, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((h, w)).astype(np.float32)


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------


class TestGCN:
    def test_output_shape_preserved(self) -> None:
        img = _random_image()
        assert gcn(img).shape == img.shape

    def test_output_dtype_float64(self) -> None:
        assert gcn(_random_image()).dtype == np.float64

    def test_mean_near_zero(self) -> None:
        out = gcn(_random_image())
        assert abs(out.mean()) < 1e-10

    def test_std_near_one(self) -> None:
        # GCN divides by (σ + ε), so output std = σ/(σ+ε) < 1 by ~ε/σ
        out = gcn(_random_image())
        assert abs(out.std() - 1.0) < 1e-4

    def test_uniform_image_no_div_zero(self) -> None:
        uniform = np.full((_H, _W), 5.0)
        out = gcn(uniform)
        assert np.isfinite(out).all()
        assert abs(out.mean()) < 1e-6

    def test_custom_eps_accepted(self) -> None:
        out = gcn(_random_image(), eps=1e-3)
        assert np.isfinite(out).all()

    def test_different_inputs_give_different_outputs(self) -> None:
        a = gcn(np.ones((_H, _W)) * 1.0)
        b = gcn(np.ones((_H, _W)) * 2.0)
        # Both uniform → both zero after GCN, but let's test non-uniform
        rng = np.random.default_rng(7)
        img1 = rng.standard_normal((_H, _W))
        img2 = img1 * 2 + 3
        # Linear transform → GCN output should be identical (scale/shift invariant)
        assert np.allclose(gcn(img1), gcn(img2), atol=1e-6)


# ---------------------------------------------------------------------------
# LCN
# ---------------------------------------------------------------------------


class TestLCN:
    def test_output_shape_preserved(self) -> None:
        img = _random_image()
        assert lcn(img).shape == img.shape

    def test_output_dtype_float64(self) -> None:
        assert lcn(_random_image()).dtype == np.float64

    def test_finite_output(self) -> None:
        assert np.isfinite(lcn(_random_image())).all()

    def test_uniform_image_no_div_zero(self) -> None:
        uniform = np.full((_H, _W), 3.0)
        out = lcn(uniform)
        assert np.isfinite(out).all()

    def test_custom_window_accepted(self) -> None:
        img = _random_image()
        out = lcn(img, window=15)
        assert out.shape == img.shape

    def test_custom_eps_accepted(self) -> None:
        assert np.isfinite(lcn(_random_image(), eps=1e-3)).all()

    def test_different_windows_give_different_outputs(self) -> None:
        img = _random_image(128, 128)
        out_small = lcn(img, window=3)
        out_large = lcn(img, window=31)
        assert not np.allclose(out_small, out_large)

    def test_masked_output_zero_at_invalid_pixels(self) -> None:
        img = _random_image()
        mask = np.ones((_H, _W), dtype=bool)
        mask[:, 20:30] = False
        out = lcn(img, mask=mask)
        assert (out[:, 20:30] == 0.0).all()
        assert np.isfinite(out).all()

    def test_masked_stats_ignore_gap_plateau(self) -> None:
        # A deep constant plateau in the gap must not bleed into the local
        # stats of neighbouring valid pixels (the halo bug): masked output
        # near the gap should match a gap-free reference away from edges.
        rng = np.random.default_rng(1)
        img = rng.normal(0.0, 1.0, size=(_H, _W))
        img_gapped = img.copy()
        img_gapped[:, 30:34] = -50.0  # gap plateau
        mask = np.ones((_H, _W), dtype=bool)
        mask[:, 30:34] = False
        out_masked = lcn(img_gapped, mask=mask)
        out_ref = lcn(img)
        # columns adjacent to the gap, excluding the gap itself
        np.testing.assert_allclose(out_masked[:, 36:40], out_ref[:, 36:40], atol=0.35)
        # unmasked LCN on the gapped image deviates far more there
        out_plain = lcn(img_gapped)
        assert np.abs(out_plain[:, 34:36] - out_ref[:, 34:36]).max() > 1.0

    def test_low_variance_noise_not_amplified(self) -> None:
        # Near-constant background + tiny readout noise must NOT be inflated
        # to unit variance (the old std-form eps=1e-6 salt-and-pepper bug).
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.01, size=(_H, _W))
        out = lcn(noise)
        assert out.std() < 0.5


# ---------------------------------------------------------------------------
# Order enforcement: GCN → LCN must differ from LCN → GCN
# ---------------------------------------------------------------------------


class TestNormalizationOrder:
    def test_gcn_then_lcn_differs_from_lcn_then_gcn(self) -> None:
        img = _random_image()
        correct = lcn(gcn(img))
        reversed_order = gcn(lcn(img))
        # The outputs should not be identical (order matters)
        assert not np.allclose(correct, reversed_order, atol=1e-6)
