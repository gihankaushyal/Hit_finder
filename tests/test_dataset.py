"""Tests for UnlabeledDataset, MultiFrameCXIDataset, and DataLoader factories."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.dataloader import ssl_pretrain_loader
from src.data.dataset import MultiFrameCXIDataset, UnlabeledDataset

_H, _W = 256, 256
_N = 4
_N_FRAMES = 6
_LABEL_KEY = "entry_1/labels/hit"


def _make_multiframe_cxi(
    tmp_path: Path, n_frames: int = _N_FRAMES, filename: str = "test.cxi"
) -> Path:
    """Create a Resonet-style multi-frame CXI file with embedded labels."""
    path = tmp_path / filename
    rng = np.random.default_rng(77)
    data = rng.integers(0, 1000, (n_frames, _H, _W), dtype=np.uint16)
    # Alternating hit/non-hit: [1,0,1,0,1,0]
    labels = np.array([float(i % 2) for i in range(n_frames)], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=data)
        f.create_dataset(_LABEL_KEY, data=labels)
    return path


def _make_h5_files(tmp_path: Path, n: int = _N) -> list[Path]:
    paths = []
    rng = np.random.default_rng(42)
    for i in range(n):
        p = tmp_path / f"frame_{i}.h5"
        data = rng.integers(0, 1000, (_H, _W), dtype=np.uint16)
        with h5py.File(p, "w") as f:
            f.create_dataset("entry/data/data", data=data)
        paths.append(p)
    return paths


class TestUnlabeledDataset:
    def test_len(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        ds = UnlabeledDataset(paths)
        assert len(ds) == _N

    def test_getitem_shape(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        ds = UnlabeledDataset(paths)
        item = ds[0]
        assert item.shape == (1, _H, _W)

    def test_getitem_dtype(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        ds = UnlabeledDataset(paths)
        assert ds[0].dtype == torch.float32

    def test_file_not_held_open(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path, n=2)
        ds = UnlabeledDataset(paths)
        _ = ds[0]
        _ = ds[1]
        with h5py.File(paths[0], "r") as f:
            assert "entry/data/data" in f


class TestMultiFrameCXIDataset:
    def test_len_single_file(self, tmp_path: Path) -> None:
        path = _make_multiframe_cxi(tmp_path)
        ds = MultiFrameCXIDataset([path], preprocess_fn=None)
        assert len(ds) == _N_FRAMES

    def test_len_multiple_files(self, tmp_path: Path) -> None:
        p1 = _make_multiframe_cxi(tmp_path, n_frames=4, filename="a.cxi")
        p2 = _make_multiframe_cxi(tmp_path, n_frames=6, filename="b.cxi")
        ds = MultiFrameCXIDataset([p1, p2], preprocess_fn=None)
        assert len(ds) == 10

    def test_getitem_returns_tensor_and_label(self, tmp_path: Path) -> None:
        path = _make_multiframe_cxi(tmp_path)
        ds = MultiFrameCXIDataset([path], preprocess_fn=None)
        tensor, label = ds[0]
        assert tensor.shape == (1, _H, _W)
        assert tensor.dtype == torch.float32
        assert label in (0, 1)

    def test_labels_correct_alternating(self, tmp_path: Path) -> None:
        path = _make_multiframe_cxi(tmp_path)
        ds = MultiFrameCXIDataset([path], preprocess_fn=None)
        for i in range(_N_FRAMES):
            _, label = ds[i]
            assert label == i % 2, f"Frame {i}: expected {i % 2}, got {label}"

    def test_no_preprocess_preserves_raw_shape(self, tmp_path: Path) -> None:
        path = _make_multiframe_cxi(tmp_path)
        ds = MultiFrameCXIDataset([path], preprocess_fn=None)
        tensor, _ = ds[0]
        assert tensor.shape == (1, _H, _W)

    def test_file_not_held_open(self, tmp_path: Path) -> None:
        path = _make_multiframe_cxi(tmp_path)
        ds = MultiFrameCXIDataset([path], preprocess_fn=None)
        _ = ds[0]
        _ = ds[1]
        # File should still be openable (not locked)
        with h5py.File(path, "r") as f:
            assert "entry_1/data_1/data" in f

    def test_cross_file_indexing(self, tmp_path: Path) -> None:
        p1 = _make_multiframe_cxi(tmp_path, n_frames=3, filename="a.cxi")
        p2 = _make_multiframe_cxi(tmp_path, n_frames=3, filename="b.cxi")
        ds = MultiFrameCXIDataset([p1, p2], preprocess_fn=None)
        # Indices 0-2 from p1, indices 3-5 from p2
        assert len(ds) == 6
        t3, _ = ds[3]  # first frame of second file
        assert t3.shape == (1, _H, _W)


class TestSSLPretrainLoader:
    def test_batch_shape(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        loader = ssl_pretrain_loader(paths, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        assert batch.shape == (2, 1, _H, _W)

    def test_full_epoch(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        loader = ssl_pretrain_loader(paths, batch_size=2, num_workers=0, shuffle=False)
        batches = list(loader)
        assert len(batches) == _N // 2


