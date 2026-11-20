"""atomic_torch_save: a kill mid-write never leaves a truncated checkpoint."""

from __future__ import annotations

import pytest
import torch

from src.training import checkpoint_io
from src.training.checkpoint_io import atomic_torch_save


def test_writes_the_file_and_no_tmp(tmp_path):
    path = tmp_path / "last.pt"
    atomic_torch_save({"epoch": 3}, path)
    assert torch.load(path, weights_only=True) == {"epoch": 3}
    assert list(tmp_path.iterdir()) == [path]


def test_a_failed_write_keeps_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_torch_save({"epoch": 3}, path)

    def _dies_mid_write(obj, f):
        f.write(b"trunc")
        raise OSError("preempted")

    monkeypatch.setattr(checkpoint_io.torch, "save", _dies_mid_write)
    with pytest.raises(OSError, match="preempted"):
        atomic_torch_save({"epoch": 4}, path)
    monkeypatch.undo()
    assert torch.load(path, weights_only=True) == {"epoch": 3}
    assert list(tmp_path.iterdir()) == [path]


def test_the_data_is_fsynced_before_the_rename(tmp_path, monkeypatch):
    order: list[str] = []
    real_save = checkpoint_io.torch.save
    real_fsync, real_replace = checkpoint_io.os.fsync, checkpoint_io.os.replace
    monkeypatch.setattr(
        checkpoint_io.torch,
        "save",
        lambda obj, f: (order.append("save"), real_save(obj, f)),
    )
    monkeypatch.setattr(
        checkpoint_io.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))
    )
    monkeypatch.setattr(
        checkpoint_io.os,
        "replace",
        lambda a, b: (order.append("replace"), real_replace(a, b)),
    )
    atomic_torch_save({"epoch": 1}, tmp_path / "last.pt")
    assert order == ["save", "fsync", "replace"]
