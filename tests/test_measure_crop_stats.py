import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "measure_crop_stats",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_crop_stats.py",
)
mcs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcs)

H = W = 600


def _frame():
    valid = np.ones((H, W), dtype=bool)
    valid[:, 290:310] = False  # a vertical panel gap
    peaks = np.array([[5.0, 5.0], [300.0, 300.0], [320.0, 310.0], [590.0, 590.0]])
    return valid, peaks


def test_modes_differ_in_pad_exposure_and_always_contain_a_peak():
    valid, peaks = _frame()
    out = mcs.frame_crop_stats(valid, peaks, np.random.default_rng(0), draws=200)
    pad = {m: np.mean([r[1] for r in out[m]["rows"]]) for m in mcs.POSITIVE_MODES}
    assert pad["centred"] > 0  # corner peaks put the zero band in a centred crop
    assert pad["uniform+within_frame"] == 0.0
    assert out["hard_neg+within_frame"]["rows"], "negatives must still be found"
    assert all(r[1] == 0.0 for r in out["hard_neg+within_frame"]["rows"])
    for m in mcs.POSITIVE_MODES:
        assert all(r[2] >= 1 for r in out[m]["rows"]), f"{m}: crop without its peak"


def test_peaks_per_crop_counts_neighbours_and_eval_tiles_only_count_peak_tiles():
    valid, peaks = _frame()
    # window covering both central peaks only
    assert (
        mcs.peaks_in_window(peaks + mcs.PAD_BORDER_DEFAULT, 250 + 112, 250 + 112) == 2
    )
    tiles = mcs.eval_tile_stats(valid, peaks)
    assert tiles and all(t[2] >= 1 for t in tiles)
    s = mcs.summarise(tiles, positive=True)
    assert s["n"] == len(tiles) and set(s["peaks_hist"]) == {"1", "2", "3-4", "5+"}
    assert mcs.summarise([], positive=True) == {"n": 0}
