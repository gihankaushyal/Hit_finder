"""Tests for vote aggregation mode in run_patch_agg."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.benchmark import run_patch_agg
from src.evaluation.metrics import RECALL_FPR_KEYS


@pytest.fixture(autouse=True)
def _fake_jungfrau_assembly(fake_jungfrau_assembly_via_pipeline: None) -> None:
    pass


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
    """Write one-frame CXI and return session_map, session_ids.

    desc="Jungfrau 4M": an arbitrary-shape synthetic frame is not a real
    detector canvas for any of the 4 detector types, so the
    _fake_jungfrau_assembly fixture (above) is required to keep this
    shape-agnostic — real PADAssembler needs a full (2164, 2068) canvas.
    """
    import h5py

    cxi = tmp_path / "test.cxi"
    with h5py.File(cxi, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=frame[np.newaxis])
        f.create_dataset("entry_1/labels/hit", data=np.array([label], dtype=np.float32))
        f.create_dataset(
            "entry_1/instrument_1/detector_1/description", data=b"Jungfrau 4M"
        )
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


def _two_frame_session(tmp_path):
    """One session, two frames (labels 1 and 0), Jungfrau desc as above."""
    import h5py

    cxi = tmp_path / "two.cxi"
    frames = np.zeros((2, 448, 448), dtype=np.float32)
    with h5py.File(cxi, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=frames)
        f.create_dataset("entry_1/labels/hit", data=np.array([1, 0], dtype=np.float32))
        f.create_dataset(
            "entry_1/instrument_1/detector_1/description", data=b"Jungfrau 4M"
        )
    return {"s0": cxi}, ["s0"]


def test_run_patch_agg_reports_recall_at_fixed_fpr(tmp_path):
    session_map, ids = _two_frame_session(tmp_path)
    plain = run_patch_agg(_FixedScoreModel(0.8), session_map, ids, device="cpu")
    # opt-in: validation pays for neither the recall keys nor the scores
    assert not set(RECALL_FPR_KEYS) & set(plain) and "scores" not in plain
    out = run_patch_agg(
        _FixedScoreModel(0.8), session_map, ids, device="cpu", return_test_metrics=True
    )
    for key in RECALL_FPR_KEYS:
        assert key in out
    # both frames score 1.0 (tied): recall 0 at FPR 0
    assert out["recall_at_fpr_1e-2"] == 0.0 and out["fpr_at_fpr_1e-2"] == 0.0
    assert out["max_recall_at_fpr_1e-3"] == 0.0


def test_run_patch_agg_returns_per_frame_scores_on_request(tmp_path):
    session_map, ids = _two_frame_session(tmp_path)
    out = run_patch_agg(
        _FixedScoreModel(0.8), session_map, ids, device="cpu", return_test_metrics=True
    )
    sc = out["scores"]
    assert sc["y_true"].tolist() == [1, 0]
    assert sc["vote"].tolist() == [1.0, 1.0]
    assert sc["vote"].dtype == np.float32 and sc["y_true"].dtype == np.int8
    assert sc["max"].shape == (2,) and sc["session"].tolist() == ["s0", "s0"]


def test_run_patch_agg_with_no_frames_keeps_every_recall_key_nan(tmp_path):
    out = run_patch_agg(
        _FixedScoreModel(0.8),
        {"s0": tmp_path / "missing.cxi"},
        ["s0"],
        device="cpu",
        return_test_metrics=True,
    )
    assert all(np.isnan(out[k]) for k in RECALL_FPR_KEYS)


def test_recall_keys_use_the_aggregation_score_and_the_max_score_separately():
    from src.evaluation.benchmark import _recall_metrics
    from src.evaluation.metrics import recall_at_fpr

    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # The two scores rank the frames differently: vote separates, max does not.
    vote = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    max_score = np.array([0.2, 0.9, 0.95, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    out = _recall_metrics(y, vote, max_score)
    assert out["recall_at_fpr_1e-2"] == recall_at_fpr(y, vote, 1e-2)[0] == 1.0
    assert out["max_recall_at_fpr_1e-2"] == recall_at_fpr(y, max_score, 1e-2)[0]
    assert out["max_recall_at_fpr_1e-2"] == 0.0


class _PerFrameScoreModel(torch.nn.Module):
    """Patch scores taken in order from a list of per-frame score lists, one per call."""

    def __init__(self, frames: list[list[float]]) -> None:
        super().__init__()
        self._frames = iter(frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tensor(next(self._frames))
        assert len(scores) == x.shape[0]  # one forward per frame (4 patches)
        logits = torch.zeros(x.shape[0], 2)
        logits[:, 1] = torch.log(scores / (1 - scores))
        return logits


@pytest.mark.parametrize("aggregation", ["vote", "max"])
def test_headline_recall_follows_the_configured_aggregation(tmp_path, aggregation):
    from src.evaluation.metrics import recall_at_fpr

    session_map, ids = _two_frame_session(tmp_path)  # labels [1, 0], 4 patches each
    # Hit frame: one strong tile (vote 0.25, max 0.99). Non-hit: four mid tiles
    # (vote 1.0, max 0.6). Vote ranks the non-hit first, max the hit.
    model = _PerFrameScoreModel([[0.99, 0.1, 0.1, 0.1], [0.6, 0.6, 0.6, 0.6]])
    out = run_patch_agg(
        model, session_map, ids, device="cpu", aggregation=aggregation,
        return_test_metrics=True,
    )  # fmt: skip
    y, sc = out["scores"]["y_true"], out["scores"]
    expected = recall_at_fpr(y, sc[aggregation], 1e-2)[0]
    assert out["recall_at_fpr_1e-2"] == expected
    assert out["recall_at_fpr_1e-2"] == (0.0 if aggregation == "vote" else 1.0)
    assert out["max_recall_at_fpr_1e-2"] == 1.0  # always the max-tile score


def test_scores_keep_session_ids_aligned_with_labels(tmp_path):
    import h5py

    session_map = {}
    for sid, labels in (("s0", [1, 0]), ("s1", [0, 0])):
        cxi = tmp_path / f"{sid}.cxi"
        with h5py.File(cxi, "w") as f:
            f.create_dataset(
                "entry_1/data_1/data", data=np.zeros((2, 448, 448), np.float32)
            )
            f.create_dataset("entry_1/labels/hit", data=np.array(labels, np.float32))
            f.create_dataset(
                "entry_1/instrument_1/detector_1/description", data=b"Jungfrau 4M"
            )
        session_map[sid] = cxi
    out = run_patch_agg(
        _FixedScoreModel(0.8), session_map, ["s0", "s1"], device="cpu",
        return_test_metrics=True,
    )  # fmt: skip
    assert out["scores"]["session"].tolist() == ["s0", "s0", "s1", "s1"]
    assert out["scores"]["y_true"].tolist() == [1, 0, 0, 0]
