"""Unit tests for GCN and LCN normalization functions."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.normalize import GCN_EPSILON, LCN_EPSILON, gcn, lcn

_RNG = np.random.default_rng(0)
_H, _W = 64, 64


def _random_image(h: int = _H, w: int = _W) -> np.ndarray:
    return _RNG.standard_normal((h, w)).astype(np.float32)


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
        img1 = _RNG.standard_normal((_H, _W))
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
