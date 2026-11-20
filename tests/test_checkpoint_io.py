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
        with open(f, "wb") as fh:
            fh.write(b"trunc")
        raise OSError("preempted")

    monkeypatch.setattr(checkpoint_io.torch, "save", _dies_mid_write)
    with pytest.raises(OSError, match="preempted"):
        atomic_torch_save({"epoch": 4}, path)
    monkeypatch.undo()
    assert torch.load(path, weights_only=True) == {"epoch": 3}
    assert list(tmp_path.iterdir()) == [path]
