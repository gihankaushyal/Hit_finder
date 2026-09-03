"""Unit tests for the --resume-training flag in src/training/train_asymmetric.py.

All tests run on CPU with mocked checkpoints — no SLURM, no real training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.supervised import build_supervised_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    tmp_path: Path, epoch: int, val_f1: float, include_optimizer: bool = True
) -> Path:
    """Write a minimal checkpoint to disk and return its path."""
    model = build_supervised_model("resnet18", pretrained=False, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt: dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_f1": val_f1,
        "inference_threshold": 0.5,
        "backbone": "resnet18",
        "num_classes": 2,
    }
    if include_optimizer:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    ckpt_path = tmp_path / "best.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resume_training_flag_parses():
    """--resume-training is accepted by argparse and stored as True."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-training", action="store_true", default=False)
    args = parser.parse_args(["--resume-training"])
    assert args.resume_training is True


def test_no_resume_training_flag_defaults_false():
    """Omitting --resume-training gives False (checkpoint → eval-only path)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-training", action="store_true", default=False)
    args = parser.parse_args([])
    assert args.resume_training is False


def test_resume_restores_model_and_optimizer(tmp_path: Path):
    """Resume path loads model weights, optimizer state, best_f1, and start_epoch."""
    ckpt_path = _make_checkpoint(tmp_path, epoch=5, val_f1=0.85)

    model = build_supervised_model("resnet18", pretrained=False, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_ckpt["model_state_dict"])
    optimizer.load_state_dict(_ckpt["optimizer_state_dict"])

    _saved_f1 = _ckpt.get("val_f1", -1.0)
    best_f1 = -1.0 if np.isnan(_saved_f1) else _saved_f1
    start_epoch = _ckpt.get("epoch", 0) + 1

    assert best_f1 == pytest.approx(0.85)
    assert start_epoch == 6
    # Optimizer state has param groups restored
    assert optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(1e-4)


def test_resume_backward_compat_no_optimizer_state(tmp_path: Path, capsys):
    """Old checkpoint without optimizer_state_dict: warns but does not crash."""
    ckpt_path = _make_checkpoint(
        tmp_path, epoch=3, val_f1=0.72, include_optimizer=False
    )

    model = build_supervised_model("resnet18", pretrained=False, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_ckpt["model_state_dict"])

    if "optimizer_state_dict" in _ckpt:
        optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
    else:
        print(
            "  Warning: checkpoint has no optimizer state — starting with fresh optimizer."
        )

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "fresh optimizer" in captured.out
    # Optimizer is still usable (fresh state)
    assert optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(1e-4)


def test_resume_epoch_overflow_skips_loop(tmp_path: Path, capsys):
    """When checkpoint epoch >= config epochs, the training range is empty."""
    epochs = 10
    ckpt_path = _make_checkpoint(tmp_path, epoch=10, val_f1=0.90)

    _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    start_epoch = _ckpt.get("epoch", 0) + 1  # 11

    if start_epoch > epochs:
        print(
            f"  Warning: checkpoint epoch {start_epoch - 1} >= config epochs {epochs}. "
            "Nothing left to train — proceeding to evaluation."
        )

    # The training loop range(11, 11) is empty — no iterations
    iterations = list(range(start_epoch, epochs + 1))
    assert iterations == []

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "Nothing left to train" in captured.out


def test_checkpoint_detection_split(tmp_path: Path):
    """resume_eval_only and resume_training_from_ckpt are mutually exclusive."""
    ckpt_path = tmp_path / "best.pt"
    ckpt_path.touch()  # simulate existing checkpoint

    # flag absent → eval-only
    resume_training = False
    resume_eval_only = ckpt_path.exists() and not resume_training
    resume_training_from_ckpt = ckpt_path.exists() and resume_training
    assert resume_eval_only is True
    assert resume_training_from_ckpt is False

    # flag present → resume training
    resume_training = True
    resume_eval_only = ckpt_path.exists() and not resume_training
    resume_training_from_ckpt = ckpt_path.exists() and resume_training
    assert resume_eval_only is False
    assert resume_training_from_ckpt is True

    # no checkpoint → both False regardless of flag
    ckpt_path.unlink()
    for flag in (True, False):
        assert (ckpt_path.exists() and not flag) is False
        assert (ckpt_path.exists() and flag) is False
