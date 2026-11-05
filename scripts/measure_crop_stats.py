"""Measure what the training crops look like under each crop-placement mode.

For hit frames of one split, draws many positive crops per frame under
    centred                (v2: peak in the middle of the crop)
    uniform                (positive_jitter: uniform; may still overlap the pad band)
    uniform+within_frame   (positive_jitter: uniform and crop_within_frame: true)
and hard-negative crops with and without crop_within_frame, then reports per detector

    valid_frac   mean fraction of crop pixels covered by real panels (gaps and pad count as invalid)
    pad_frac     mean fraction of crop pixels in the 112 px zero pad band; P(pad>0) is how often
    peaks        peaks per positive crop: mean and histogram (1, 2, 3-4, 5+)

next to the same statistics for the evaluation tile grid (stride-224 tiles that contain at
least one peak). Only masks and peak lists are used, so no image data is cropped or scored.

Usage:
    python scripts/measure_crop_stats.py --config configs/ssl/mae_finetune.yaml --fold 4 \\
        --frames-per-detector 60 --out docs/figures/crop_stats/fold4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (  # noqa: E402
    HARD_NEG_MARGIN_DEFAULT as HARD_NEG_MARGIN,
    POSITIVE_JITTER_EDGE_PAD,
    _crop_bounds,
    _load_gcn_frame,
    _path_a_crop,
    _sample_clear_crop,
)
from src.evaluation.benchmark import (  # noqa: E402
    SPLIT_TRAIN,
    build_lodo_folds,
    build_session_stratified_split,
)
from src.hitfinders import get_hitfinder  # noqa: E402
from src.preprocessing.augment import PAD_BORDER_DEFAULT, pad_border  # noqa: E402
from src.preprocessing.io import (
    read_detector_description,
    read_embedded_labels,
)  # noqa: E402
from src.training.lodo import build_sessions  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CROP = 224
HARD_NEG_TRIES = 50
PEAK_BINS = ((1, 1), (2, 2), (3, 4), (5, 10**9))
# channels of the stacked map the crop helpers cut from
_VALID, _INFRAME, _ROW, _COL = range(4)

POSITIVE_MODES = {
    "centred": (None, False),
    "uniform": (POSITIVE_JITTER_EDGE_PAD, False),
    "uniform+within_frame": (POSITIVE_JITTER_EDGE_PAD, True),
}
NEGATIVE_MODES = {"hard_neg": False, "hard_neg+within_frame": True}


def stack_maps(valid_mask: np.ndarray) -> np.ndarray:
    """(H, W, 4) padded map: valid pixels, in-frame indicator, row index, column index.

    The index channels let a cut crop report where it came from (crop[0, 0, _ROW/_COL]).
    """
    v = pad_border(valid_mask.astype(np.float64))
    inframe = pad_border(np.ones(valid_mask.shape, dtype=np.float64))
    rows, cols = np.indices(v.shape)
    return np.dstack([v, inframe, rows.astype(np.float64), cols.astype(np.float64)])


def _crop_summary(crop: np.ndarray) -> tuple[float, float, int, int]:
    return (
        float(crop[:, :, _VALID].mean()),
        float(1.0 - crop[:, :, _INFRAME].mean()),
        int(crop[0, 0, _ROW]),
        int(crop[0, 0, _COL]),
    )


def peaks_in_window(centroids: np.ndarray, top: int, left: int) -> int:
    """Number of [x, y] centroids inside the CROP x CROP window at (top, left)."""
    x, y = centroids[:, 0], centroids[:, 1]
    return int(((x >= left) & (x < left + CROP) & (y >= top) & (y < top + CROP)).sum())


def frame_crop_stats(
    valid_mask: np.ndarray, centroids: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, dict[str, list]]:
    """Per-mode lists of (valid_frac, pad_frac, n_peaks) for one frame.

    `centroids` are [x, y] in the un-padded frame; at least one is required.
    """
    maps = stack_maps(valid_mask)
    ph, pw = maps.shape[:2]
    shifted = centroids + PAD_BORDER_DEFAULT
    out: dict[str, dict[str, list]] = {}
    for name, (jitter, within) in POSITIVE_MODES.items():
        bounds = _crop_bounds(ph, pw, CROP, within)
        rows = []
        for _ in range(draws):
            crop, _ = _path_a_crop(maps, shifted, rng, ph, pw, CROP, jitter, bounds)
            valid, pad, top, left = _crop_summary(crop)
            rows.append((valid, pad, peaks_in_window(shifted, top, left)))
        out[name] = {"rows": rows}
    for name, within in NEGATIVE_MODES.items():
        bounds = _crop_bounds(ph, pw, CROP, within)
        rows = []
        for _ in range(draws):
            crop = _sample_clear_crop(
                maps,
                shifted,
                rng,
                ph,
                pw,
                CROP,
                HARD_NEG_MARGIN,
                HARD_NEG_TRIES,
                bounds,
            )
            if crop is not None:
                valid, pad, _, _ = _crop_summary(crop)
                rows.append((valid, pad, 0))
        out[name] = {"rows": rows, "failed": draws - len(rows)}
    return out


def eval_tile_stats(
    valid_mask: np.ndarray, centroids: np.ndarray
) -> list[tuple[float, float, int]]:
    """(valid_frac, pad_frac=0, n_peaks) for each stride-224 eval tile holding >= 1 peak."""
    h, w = valid_mask.shape
    rows = []
    for top in range(0, h - CROP + 1, CROP):
        for left in range(0, w - CROP + 1, CROP):
            n = peaks_in_window(centroids, top, left)
            if n:
                rows.append(
                    (
                        float(valid_mask[top : top + CROP, left : left + CROP].mean()),
                        0.0,
                        n,
                    )
                )
    return rows


def summarise(rows: list[tuple[float, float, int]], positive: bool) -> dict:
    """Aggregate (valid_frac, pad_frac, n_peaks) rows into means and a peaks histogram."""
    if not rows:
        return {"n": 0}
    a = np.asarray(rows, dtype=np.float64)
    out = {
        "n": len(rows),
        "valid_frac_mean": float(a[:, 0].mean()),
        "valid_frac_ge_0.8": float((a[:, 0] >= 0.8).mean()),
        "pad_frac_mean": float(a[:, 1].mean()),
        "p_pad_gt_0": float((a[:, 1] > 0).mean()),
    }
    if positive:
        out["peaks_mean"] = float(a[:, 2].mean())
        out["peaks_hist"] = {
            (
                f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**6 else f"{lo}-{hi}")
            ): float(((a[:, 2] >= lo) & (a[:, 2] <= hi)).mean())
            for lo, hi in PEAK_BINS
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--split", default=SPLIT_TRAIN)
    p.add_argument("--frames-per-detector", type=int, default=60)
    p.add_argument("--draws", type=int, default=20, help="crops per frame per mode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    hitfinder = get_hitfinder(cfg)
    sessions, session_map = build_sessions(cfg["lodo"])
    fold = next(f for f in build_lodo_folds() if f["fold_id"] == args.fold)
    split = build_session_stratified_split(
        sessions,
        test_detector=fold["test_detector"],
        fold=fold["fold_id"],
        seed=cfg["seed"],
    )
    ids = [sid for sid, s in split["splits"].items() if s == args.split]
    label_key = cfg["lodo"].get("label_key", "entry_1/labels/hit")

    by_desc: dict[str, list[tuple[Path, int]]] = {}
    for sid in ids:
        path = session_map[sid]
        hits = np.flatnonzero(np.round(read_embedded_labels(path, label_key)) == 1)
        by_desc.setdefault(read_detector_description(path), []).extend(
            (path, int(i)) for i in hits
        )

    geom_cache: dict[Path, dict[str, float]] = {}
    holder: list = [None]
    report: dict[str, dict] = {}
    for desc, frames in sorted(by_desc.items()):
        pick = rng.permutation(len(frames))[: args.frames_per_detector]
        acc: dict[str, list] = {}
        failed: dict[str, int] = {}
        eval_rows: list = []
        n_used = 0
        for k in pick:
            path, idx = frames[int(k)]
            _, valid_mask, centroids = _load_gcn_frame(
                path, idx, desc, geom_cache, hitfinder, holder
            )
            if centroids.shape[0] == 0:
                continue
            n_used += 1
            for mode, d in frame_crop_stats(
                valid_mask, centroids, rng, args.draws
            ).items():
                acc.setdefault(mode, []).extend(d["rows"])
                failed[mode] = failed.get(mode, 0) + d.get("failed", 0)
            eval_rows.extend(eval_tile_stats(valid_mask, centroids))
        report[desc] = {"frames_used": n_used}
        for mode in list(POSITIVE_MODES) + list(NEGATIVE_MODES):
            report[desc][mode] = summarise(acc.get(mode, []), mode in POSITIVE_MODES)
            if mode in NEGATIVE_MODES:
                # a failed draw (no clear window) is dropped, so report how often it happens
                tried = len(acc.get(mode, [])) + failed.get(mode, 0)
                report[desc][mode]["draw_failure_rate"] = (
                    failed.get(mode, 0) / tried if tried else 0.0
                )
        report[desc]["eval_tiles_with_peak"] = summarise(eval_rows, True)
        print(f"{desc}: {n_used} frames")
        for mode, s in report[desc].items():
            if isinstance(s, dict) and s.get("n"):
                print(
                    f"  {mode:24s} valid={s['valid_frac_mean']:.3f} "
                    f">=0.8:{s['valid_frac_ge_0.8']:.2f} pad={s['pad_frac_mean']:.3f} "
                    f"P(pad>0)={s['p_pad_gt_0']:.2f}"
                    + (
                        f" peaks={s['peaks_mean']:.2f} {s['peaks_hist']}"
                        if "peaks_mean" in s
                        else ""
                    )
                )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
