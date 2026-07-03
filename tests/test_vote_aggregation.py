"""Tests for vote aggregation mode in run_patch_agg."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.benchmark import run_patch_agg


class _FixedScoreModel(torch.nn.Module):
    """Returns fixed softmax scores for every patch."""

    def __init__(self, hit_score: float) -> None:
        super().__init__()
        self._score = hit_score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        score = self._score
        logit = np.log(score / (1 - score + 1e-9))
        logits = torch.zeros(batch, 2)
        logits[:, 1] = logit
        return logits


def _make_single_frame_session(tmp_path, frame: np.ndarray, label: int):
    """Write one-frame CXI and return session_map, session_ids."""
    import h5py

    cxi = tmp_path / "test.cxi"
    with h5py.File(cxi, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=frame[np.newaxis])
        f.create_dataset("entry_1/labels/hit", data=np.array([label], dtype=np.float32))
    return {"s0": cxi}, ["s0"]


def test_vote_aggregation_all_patches_hit(tmp_path):
    # Frame 896×896 → 4×4=16 tiles of 224×224
    # Model always scores 0.8 → all tiles are "hit" → score = 16/16 = 1.0
    frame = np.zeros((896, 896), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=1)
    model = _FixedScoreModel(hit_score=0.8)
    result = run_patch_agg(
        model, session_map, session_ids, aggregation="vote", device="cpu"
    )
    assert result["ap"] == pytest.approx(1.0, abs=1e-5)


def test_vote_aggregation_no_hits(tmp_path):
    # Model always scores 0.2 → no tiles hit → score = 0.0
    frame = np.zeros((896, 896), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=0)
    model = _FixedScoreModel(hit_score=0.2)
    result = run_patch_agg(
        model, session_map, session_ids, aggregation="vote", device="cpu"
    )
    # Only non-hit frames → AP=0 (no positives in session)
    assert result["ap"] == pytest.approx(0.0, abs=1e-5)


def test_max_aggregation_backward_compat(tmp_path):
    frame = np.zeros((448, 448), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=1)
    model = _FixedScoreModel(hit_score=0.9)
    result = run_patch_agg(
        model, session_map, session_ids, aggregation="max", device="cpu"
    )
    assert result["ap"] == pytest.approx(1.0, abs=1e-5)


def test_run_patch_agg_default_aggregation_is_max(tmp_path):
    # Calling without aggregation= keyword must not raise
    frame = np.zeros((448, 448), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=1)
    model = _FixedScoreModel(hit_score=0.9)
    result = run_patch_agg(model, session_map, session_ids, device="cpu")
    assert "ap" in result
