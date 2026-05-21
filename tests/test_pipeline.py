"""Integration tests for the full preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.geometry import load_pad_geometry
from src.preprocessing.pipeline import TARGET_SIZE, _to_2d, preprocess

DETECTORS = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]


def _make_panel_data(pads: object) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 1000, (p.n_ss, p.n_fs), dtype=np.uint16) for p in pads]


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
# preprocess — end-to-end for all four detectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector_type", DETECTORS)
class TestPreprocess:
    def test_output_shape_is_target_size(self, detector_type: str) -> None:
        pads = load_pad_geometry(detector_type)
        panel_data = _make_panel_data(pads)
        out = preprocess(panel_data, pads)
        assert out.shape == TARGET_SIZE

    def test_output_dtype_float32(self, detector_type: str) -> None:
        pads = load_pad_geometry(detector_type)
        panel_data = _make_panel_data(pads)
        assert preprocess(panel_data, pads).dtype == np.float32

    def test_output_is_finite(self, detector_type: str) -> None:
        pads = load_pad_geometry(detector_type)
        panel_data = _make_panel_data(pads)
        assert np.isfinite(preprocess(panel_data, pads)).all()

    def test_custom_lcn_window_accepted(self, detector_type: str) -> None:
        pads = load_pad_geometry(detector_type)
        panel_data = _make_panel_data(pads)
        out = preprocess(panel_data, pads, lcn_window=15)
        assert out.shape == TARGET_SIZE

    def test_deterministic_output(self, detector_type: str) -> None:
        pads = load_pad_geometry(detector_type)
        panel_data = _make_panel_data(pads)
        out1 = preprocess(panel_data, pads)
        out2 = preprocess(panel_data, pads)
        assert np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Pipeline order: normalization before resize
# ---------------------------------------------------------------------------


class TestPipelineOrder:
    def test_resize_last_produces_224x224(self) -> None:
        """Verify final output is always TARGET_SIZE regardless of input dimensions."""
        for det in DETECTORS:
            pads = load_pad_geometry(det)
            panel_data = _make_panel_data(pads)
            out = preprocess(panel_data, pads)
            assert out.shape == TARGET_SIZE, f"Failed for {det}: got {out.shape}"

    def test_different_inputs_give_different_outputs(self) -> None:
        pads = load_pad_geometry("JUNGFRAU_4M")
        rng = np.random.default_rng(7)
        data_a = [rng.integers(0, 500, (p.n_ss, p.n_fs), dtype=np.uint16) for p in pads]
        data_b = [
            rng.integers(500, 1000, (p.n_ss, p.n_fs), dtype=np.uint16) for p in pads
        ]
        assert not np.array_equal(preprocess(data_a, pads), preprocess(data_b, pads))
