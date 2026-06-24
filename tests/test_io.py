"""Tests for src/preprocessing/io.py — unified image reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.preprocessing.io import (
    count_frames,
    read_embedded_labels,
    read_frame,
    read_image,
)

_H, _W = 32, 32
_N_FRAMES = 5
_LABEL_KEY = "entry_1/labels/hit"


@pytest.fixture()
def multiframe_cxi(tmp_path: Path) -> Path:
    """CXI file with N frames and embedded labels (Resonet format)."""
    path = tmp_path / "multiframe.cxi"
    rng = np.random.default_rng(99)
    data = rng.integers(0, 1000, (_N_FRAMES, _H, _W), dtype=np.uint16)
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=data)
        f.create_dataset(_LABEL_KEY, data=labels)
    return path


@pytest.fixture()
def h5_file(tmp_path: Path) -> Path:
    path = tmp_path / "frame.h5"
    data = np.random.default_rng(0).integers(0, 1000, (_H, _W), dtype=np.uint16)
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/data/data", data=data)
    return path


@pytest.fixture()
def img_file(tmp_path: Path) -> Path:
    fabio = pytest.importorskip("fabio")
    data = np.random.default_rng(1).integers(0, 1000, (_H, _W), dtype=np.uint16)
    path = tmp_path / "frame.img"
    img = fabio.adscimage.AdscImage(data=data)
    img.write(str(path))
    return path


class TestH5Reader:
    def test_returns_2d_float32(self, h5_file: Path) -> None:
        arr = read_image(h5_file)
        assert arr.ndim == 2
        assert arr.dtype == np.float32

    def test_shape_preserved(self, h5_file: Path) -> None:
        arr = read_image(h5_file)
        assert arr.shape == (_H, _W)

    def test_cxi_extension_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "frame.cxi"
        data = np.zeros((_H, _W), dtype=np.float32)
        with h5py.File(path, "w") as f:
            f.create_dataset("entry/data/data", data=data)
        arr = read_image(path)
        assert arr.shape == (_H, _W)


class TestImgReader:
    def test_returns_2d_float32(self, img_file: Path) -> None:
        arr = read_image(img_file)
        assert arr.ndim == 2
        assert arr.dtype == np.float32

    def test_shape_preserved(self, img_file: Path) -> None:
        arr = read_image(img_file)
        assert arr.shape == (_H, _W)


class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_image(tmp_path / "missing.h5")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "frame.tiff"
        path.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            read_image(path)


# ---------------------------------------------------------------------------
# count_frames
# ---------------------------------------------------------------------------


class TestCountFrames:
    def test_multiframe_returns_n(self, multiframe_cxi: Path) -> None:
        assert count_frames(multiframe_cxi) == _N_FRAMES

    def test_single_frame_returns_1(self, h5_file: Path) -> None:
        assert count_frames(h5_file) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            count_frames(tmp_path / "missing.cxi")


# ---------------------------------------------------------------------------
# read_frame
# ---------------------------------------------------------------------------


class TestReadFrame:
    def test_default_frame_zero(self, multiframe_cxi: Path) -> None:
        arr = read_frame(multiframe_cxi)
        assert arr.shape == (_H, _W)
        assert arr.dtype == np.float32

    def test_specific_frame_index(self, multiframe_cxi: Path) -> None:
        arr = read_frame(multiframe_cxi, frame_idx=3)
        assert arr.shape == (_H, _W)

    def test_different_frames_differ(self, multiframe_cxi: Path) -> None:
        # Frames are random; almost certainly different values.
        f0 = read_frame(multiframe_cxi, frame_idx=0)
        f1 = read_frame(multiframe_cxi, frame_idx=1)
        assert not np.array_equal(f0, f1)

    def test_out_of_range_raises(self, multiframe_cxi: Path) -> None:
        with pytest.raises(IndexError):
            read_frame(multiframe_cxi, frame_idx=_N_FRAMES)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_frame(tmp_path / "missing.cxi", frame_idx=0)


# ---------------------------------------------------------------------------
# read_embedded_labels
# ---------------------------------------------------------------------------


class TestReadEmbeddedLabels:
    def test_returns_float32_array(self, multiframe_cxi: Path) -> None:
        labels = read_embedded_labels(multiframe_cxi)
        assert labels.dtype == np.float32

    def test_shape_matches_frame_count(self, multiframe_cxi: Path) -> None:
        labels = read_embedded_labels(multiframe_cxi)
        assert labels.shape == (_N_FRAMES,)

    def test_values_are_binary(self, multiframe_cxi: Path) -> None:
        labels = read_embedded_labels(multiframe_cxi)
        assert set(labels.tolist()).issubset({0.0, 1.0})

    def test_hit_count_correct(self, multiframe_cxi: Path) -> None:
        labels = read_embedded_labels(multiframe_cxi)
        assert int(labels.sum()) == 3  # fixture has 3 hits

    def test_missing_key_raises(self, multiframe_cxi: Path) -> None:
        with pytest.raises(KeyError):
            read_embedded_labels(multiframe_cxi, label_key="nonexistent/key")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_embedded_labels(tmp_path / "missing.cxi")


# ---------------------------------------------------------------------------
# Real-data HDF5 key confirmation (skipped when files absent — CI safe)
# ---------------------------------------------------------------------------

# Update paths to real detector files on your machine before running.
_REAL_DETECTOR_FILES: dict[str, str] = {
    "AGIPD": "/Users/gketawal/Desktop/detector-images/AGIPD.cxi",
    "JUNGFRAU_4M": "/Users/gketawal/Desktop/detector-images/Jungfrau.h5",
    "ePix10k": "/Users/gketawal/Desktop/detector-images/epix10k.cxi",
    "Eiger4M": "/Users/gketawal/Desktop/detector-images/Eiger4M.h5",
}

# Confirmed by running scripts/probe_hdf5.py on each detector file.
_CONFIRMED_KEYS: dict[str, str] = {
    "AGIPD": "entry_1/instrument_1/detector_1/detector_corrected/data",
    "JUNGFRAU_4M": "entry_0000/instrument/Simulator/data",
    "ePix10k": "entry_1/data_1/data",
    "Eiger4M": "entry/data/data",
}

# Expected ndim per detector (pre-assembly raw data, opened directly via h5py).
# All files are (N, H, W) — io.py returns frame[0] as 2D, but tests open raw.
_EXPECTED_NDIM: dict[str, int] = {
    "AGIPD": 3,  # raw (367, 8192, 128) — io.py returns frame[0] as 2D
    "JUNGFRAU_4M": 3,  # raw (1000, 2164, 2068) — io.py returns frame[0] as 2D
    "ePix10k": 3,  # raw (76, 5632, 384) — io.py returns frame[0] as 2D
    "Eiger4M": 3,  # raw (500, 2167, 2070) — io.py returns frame[0] as 2D
}


@pytest.mark.parametrize("detector", list(_REAL_DETECTOR_FILES.keys()))
def test_real_hdf5_key_exists(detector: str) -> None:
    path = Path(_REAL_DETECTOR_FILES[detector])
    if not path.exists():
        pytest.skip(f"Real data file not available: {path}")
    key = _CONFIRMED_KEYS[detector]
    with h5py.File(path, "r") as f:
        assert key in f, (
            f"Key '{key}' not found in {detector} file. "
            f"Run scripts/probe_hdf5.py to find the correct key."
        )


@pytest.mark.parametrize("detector", list(_REAL_DETECTOR_FILES.keys()))
def test_real_hdf5_data_shape(detector: str) -> None:
    path = Path(_REAL_DETECTOR_FILES[detector])
    if not path.exists():
        pytest.skip(f"Real data file not available: {path}")
    key = _CONFIRMED_KEYS[detector]
    with h5py.File(path, "r") as f:
        data = f[key][()]
    expected_ndim = _EXPECTED_NDIM[detector]
    assert data.ndim == expected_ndim, (
        f"{detector}: expected {expected_ndim}D array, got {data.ndim}D "
        f"with shape {data.shape}"
    )
    assert np.issubdtype(
        data.dtype, np.number
    ), f"{detector}: data dtype {data.dtype} is not numeric"
