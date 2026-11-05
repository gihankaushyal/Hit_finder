import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "measure_padding_effect",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_padding_effect.py",
)
mpe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mpe)


def test_threshold_drops_gap_heavy_tiles_and_keeps_all_when_none_qualify():
    scores = [np.array([0.9, 0.9, 0.1]), np.array([0.9, 0.1, 0.1])]
    valid = [np.array([0.9, 0.8, 0.3]), np.array([0.2, 0.2, 0.2])]
    # thr 0: both frames use all 3 tiles; thr 0.5: frame 0 keeps its 2 valid tiles,
    # frame 1 has no qualifying tile so keeps all 3
    np.testing.assert_allclose(mpe.vote_scores(scores, valid, 0.0), [2 / 3, 1 / 3])
    np.testing.assert_allclose(mpe.vote_scores(scores, valid, 0.5), [1.0, 1 / 3])


def test_rescore_reports_every_threshold_and_dropped_tiles():
    rng = np.random.default_rng(0)
    scores = [rng.random(9) for _ in range(20)]
    valid = [rng.random(9) for _ in range(20)]
    labels = np.array([0, 1] * 10)
    out = mpe.rescore(scores, valid, labels)
    assert set(out) == {"valid>=0.0", "valid>=0.5", "valid>=0.75"}
    assert out["valid>=0.0"]["mean_tiles_dropped"] == 0.0
    assert (
        out["valid>=0.75"]["mean_tiles_dropped"]
        > out["valid>=0.5"]["mean_tiles_dropped"]
    )
    assert all(0.0 <= m["ap_tieaware"] <= 1.0 for m in out.values())


def test_frame_smaller_than_a_tile_is_skipped_not_fatal():
    import torch

    small = np.zeros((100, 100), dtype=np.float32)
    assert mpe.score_frame(torch.nn.Identity(), small, small > 0, "cpu") is None
