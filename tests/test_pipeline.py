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


# ---------------------------------------------------------------------------
# fill_gaps_after_gcn
# ---------------------------------------------------------------------------


class TestFillGapsAfterGCN:
    def test_invalid_pixels_set_to_zero(self) -> None:
        from src.preprocessing.pipeline import fill_gaps_after_gcn

        frame = np.full((10, 10), -2.5)
        mask = np.ones((10, 10), dtype=bool)
        mask[:, 4:6] = False  # vertical gap
        out = fill_gaps_after_gcn(frame, mask=mask)
        assert (out[:, 4:6] == 0.0).all()
        assert (out[:, :4] == -2.5).all()
        assert (out[:, 6:] == -2.5).all()

    def test_no_desc_no_mask_is_noop(self) -> None:
        from src.preprocessing.pipeline import fill_gaps_after_gcn

        frame = np.full((8, 8), 1.5)
        out = fill_gaps_after_gcn(frame)
        assert (out == 1.5).all()

    def test_shape_mismatch_skips_fill_with_warning(self) -> None:
        from src.preprocessing.pipeline import fill_gaps_after_gcn

        frame = np.full((8, 8), 1.5)
        mask = np.zeros((4, 4), dtype=bool)
        with pytest.warns(UserWarning, match="gap fill skipped"):
            out = fill_gaps_after_gcn(frame, mask=mask)
        assert (out == 1.5).all()

    def test_unknown_desc_skips_fill_with_warning(self) -> None:
        from src.preprocessing.pipeline import _MASK_WARNED, fill_gaps_after_gcn

        _MASK_WARNED.discard("NotADetector 9000")
        frame = np.full((8, 8), 1.5)
        with pytest.warns(UserWarning, match="skipped"):
            out = fill_gaps_after_gcn(frame, "NotADetector 9000")
        assert (out == 1.5).all()
