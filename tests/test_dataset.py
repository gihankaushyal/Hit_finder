"""Tests for UnlabeledDataset, SFXDataset, and DataLoader factories."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.dataloader import ssl_pretrain_loader, supervised_loader
from src.data.dataset import SFXDataset, UnlabeledDataset

_H, _W = 32, 32
_N = 4


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


def _make_split_file(tmp_path: Path, paths: list[Path]) -> Path:
    split = tmp_path / "split.txt"
    split.write_text("\n".join(str(p) for p in paths))
    return split


def _make_labels_file(tmp_path: Path, paths: list[Path]) -> Path:
    labels = {str(p): i % 2 for i, p in enumerate(paths)}
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(json.dumps(labels))
    return labels_file


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


class TestSFXDataset:
    def test_len(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        split = _make_split_file(tmp_path, paths)
        labels = _make_labels_file(tmp_path, paths)
        ds = SFXDataset(split, labels)
        assert len(ds) == _N

    def test_getitem_returns_tensor_and_label(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path, n=2)
        split = _make_split_file(tmp_path, paths)
        labels = _make_labels_file(tmp_path, paths)
        ds = SFXDataset(split, labels)
        tensor, label = ds[0]
        assert tensor.shape == (1, _H, _W)
        assert tensor.dtype == torch.float32
        assert label in (0, 1)

    def test_labels_correct(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path, n=4)
        split = _make_split_file(tmp_path, paths)
        labels_dict = {str(p): i % 2 for i, p in enumerate(paths)}
        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps(labels_dict))
        ds = SFXDataset(split, labels_file)
        for i, p in enumerate(paths):
            _, label = ds[i]
            assert label == i % 2

    def test_missing_label_raises_key_error(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path, n=2)
        split = _make_split_file(tmp_path, paths)
        # labels file only covers the first path
        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps({str(paths[0]): 1}))
        ds = SFXDataset(split, labels_file)
        ds[0]  # should succeed
        with pytest.raises(KeyError):
            ds[1]  # missing label

    def test_blank_lines_in_split_ignored(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path, n=2)
        split = tmp_path / "split.txt"
        split.write_text(f"{paths[0]}\n\n{paths[1]}\n")
        labels = _make_labels_file(tmp_path, paths)
        ds = SFXDataset(split, labels)
        assert len(ds) == 2


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


class TestSupervisedLoader:
    def test_batch_shape_and_labels(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        split = _make_split_file(tmp_path, paths)
        labels = _make_labels_file(tmp_path, paths)
        loader = supervised_loader(
            split, labels, batch_size=2, num_workers=0, shuffle=False
        )
        images, lbls = next(iter(loader))
        assert images.shape == (2, 1, _H, _W)
        assert lbls.shape == (2,)

    def test_full_epoch(self, tmp_path: Path) -> None:
        paths = _make_h5_files(tmp_path)
        split = _make_split_file(tmp_path, paths)
        labels = _make_labels_file(tmp_path, paths)
        loader = supervised_loader(
            split, labels, batch_size=2, num_workers=0, shuffle=False
        )
        assert len(list(loader)) == _N // 2
