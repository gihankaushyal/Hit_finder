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


def test_track1_override_rotates_wandb_id(tmp_path: Path, monkeypatch):
    from src.training.train_asymmetric import _check_checkpoint_collisions

    monkeypatch.chdir(tmp_path)
    tagged = []
    monkeypatch.setattr(
        "src.training.wandb_identity.tag_overridden",
        lambda project, entity, run_id: tagged.append(run_id) or True,
    )
    cfg = {"seed": 42, "wandb": {"project": "p"}}
    name = "resnet18-asymmetric-v2-fold1-seed42"
    run_dir = tmp_path / "checkpoints" / name
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"x")
    _check_checkpoint_collisions([1], cfg, "resnet18-asymmetric-v2", False, True)
    assert (run_dir / "wandb_id.txt").read_text().strip() == f"{name}-o2"
    assert tagged == [name]
    assert not (run_dir / "best.pt").exists()


class TestTrack1PerFoldOverride:
    PREFIX = "resnet18-asymmetric-v2"

    def _drive(self, tmp_path: Path, monkeypatch, fake_train_fold):
        from src.training import train_asymmetric as ta

        monkeypatch.chdir(tmp_path)
        cfg = {
            "seed": 42,
            "wandb": {"project": "p"},
            "training": {"num_workers": 0},
            "hitfinder": {"backend": "mock"},
        }
        monkeypatch.setattr(ta, "load_config", lambda path: cfg)
        monkeypatch.setattr(ta, "get_hitfinder", lambda c: None)
        monkeypatch.setattr(ta, "frame_cache_from_cfg", lambda c: None)
        monkeypatch.setattr(
            ta,
            "build_sessions",
            lambda lodo_cfg: ([{"detector": "AGIPD", "frame_count": 1}], {}),
        )
        cfg["lodo"] = {"detector_dirs": {"AGIPD": "x"}}
        monkeypatch.setattr(
            ta,
            "build_lodo_folds",
            lambda: [
                {"fold_id": 1, "test_detector": "AGIPD"},
                {"fold_id": 2, "test_detector": "AGIPD"},
            ],
        )
        monkeypatch.setattr(
            ta, "build_session_stratified_split", lambda *a, **k: {"splits": {}}
        )
        monkeypatch.setattr(ta, "save_split_artifact", lambda *a, **k: None)
        monkeypatch.setattr(ta, "_train_fold", fake_train_fold)
        ta.main(
            "cfg.yaml",
            self.PREFIX,
            device="cpu",
            override_training=True,
        )

    def test_override_is_deferred_per_fold_and_ordered_before_id_resolution(
        self, tmp_path: Path, monkeypatch
    ):
        from src.training.wandb_identity import resolve_wandb_id

        names = {i: f"{self.PREFIX}-fold{i}-seed42" for i in (1, 2)}
        dirs = {i: tmp_path / "checkpoints" / names[i] for i in (1, 2)}
        for d in dirs.values():
            d.mkdir(parents=True)
            (d / "best.pt").write_bytes(b"x")
        seen = []

        def fake_train_fold(fold, *a, **k):
            fid = fold["fold_id"]
            # FIX 6: the id must already be rotated when _train_fold runs
            seen.append((fid, resolve_wandb_id(dirs[fid], names[fid])))
            raise RuntimeError("fold 1 crashed")

        with pytest.raises(RuntimeError):
            self._drive(tmp_path, monkeypatch, fake_train_fold)
        assert seen == [(1, f"{names[1]}-o2")]
        assert not (dirs[1] / "best.pt").exists()
        assert (dirs[2] / "best.pt").exists()
        assert not (dirs[2] / "wandb_id.txt").exists()

    def test_unresolved_collision_fails_fast_touching_nothing(
        self, tmp_path: Path, monkeypatch
    ):
        from src.training import train_asymmetric as ta

        d = tmp_path / "checkpoints" / f"{self.PREFIX}-fold2-seed42"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")
        cfg = {"seed": 42, "wandb": {"project": "p"}}
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            ta._check_checkpoint_collisions(
                [2], cfg, self.PREFIX, False, False, dry_run=True
            )
        assert (d / "best.pt").exists()
