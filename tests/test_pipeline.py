"""Integration tests for the full preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.pipeline import _to_2d



# ---------------------------------------------------------------------------
# _to_2d
# ---------------------------------------------------------------------------


class TestTo2D:
    def test_2d_input_unchanged(self) -> None:
        img = np.ones((100, 80))
        assert _to_2d(img).shape == (100, 80)

    def test_3d_input_row_stacked(self) -> None:
        img = np.ones((16, 512, 128))
        out = _to_2d(img)
        assert out.shape == (16 * 512, 128)

    def test_1d_input_reshaped(self) -> None:
        img = np.ones(2068 * 2162)
        out = _to_2d(img, pad_ss=2162, pad_fs=2068)
        assert out.shape == (2162, 2068)

    def test_1d_input_without_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="pad_ss and pad_fs"):
            _to_2d(np.ones(100))

    def test_unexpected_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="Unexpected image ndim"):
            _to_2d(np.ones((2, 3, 4, 5)))

