"""Tests for src/preprocessing/io.py — unified image reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.preprocessing.io import read_image

_H, _W = 32, 32


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
