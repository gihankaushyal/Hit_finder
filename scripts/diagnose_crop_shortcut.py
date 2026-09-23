"""Measure how much a trained classifier relies on crop-construction shortcuts.

The asymmetric training pipeline (src/data/dataset.py:380) builds a positive crop
by centring it on a randomly chosen Bragg peak, and a negative crop by taking a
random window from a frame whose hitfinder returned *zero* peaks. Two shortcuts
are therefore available to the model and neither is diffraction physics:

  1. Positional — every positive has a peak at the centre pixel. rot90 and flip
     both map the centre to itself, so augmentation does not break it.
  2. Frame-level — negatives only ever come from blank frames, so global
     background statistics separate the classes without looking at a peak.

This script scores one checkpoint under three crop distributions:

  A (control)   positives centred on a peak, negatives from peak-free frames
                — the training distribution; reproduces the reported val AP.
  B (jitter)    positives contain a peak at a random off-centre offset,
                negatives as in A. AP drop relative to A isolates shortcut 1.
  C (hard-neg)  positives as in A, negatives drawn from peak-free *regions of
                hit frames* with 50 px clearance. Drop isolates shortcut 2.

Conditions share their frame pool and differ only in crop geometry, so the
deltas are attributable to the crop construction rather than to sampling.

Usage:
    python -m scripts.diagnose_crop_shortcut \
        --config configs/ssl/mae_finetune.yaml \
        --checkpoint checkpoints/vits16-mae-finetune-fold4-seed42-v2/best.pt \
        --fold 4 --max-frames 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import _crop_within_margin, _load_gcn_frame  # noqa: E402
from src.evaluation.benchmark import (  # noqa: E402
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
)
from src.evaluation.metrics import average_precision, auc_roc  # noqa: E402
from src.hitfinders import get_hitfinder  # noqa: E402
from src.models.ssl import build_ssl_classifier  # noqa: E402
from src.preprocessing.augment import PAD_BORDER_DEFAULT, pad_border  # noqa: E402
from src.preprocessing.io import read_detector_description  # noqa: E402
from src.preprocessing.normalize import lcn  # noqa: E402
from src.training.lodo import build_sessions  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CROP = 224
HARD_NEG_MARGIN = 50
# Keep the peak clear of the crop border so condition B still shows a whole
# peak; only its position changes. 32 px each side leaves a 160 px jitter box.
JITTER_EDGE_PAD = 32


def _stack(assembled: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Pad image and mask into the (H, W, 2) layout the crop helpers expect."""
    return np.dstack([pad_border(assembled), pad_border(valid_mask.astype(np.float64))])


def _finalise(crop: np.ndarray) -> np.ndarray:
    """Masked LCN, matching training minus the train-only cutout step."""
    return lcn(crop[:, :, 0], mask=crop[:, :, 1] > 0.5)


def _centred_crop(padded: np.ndarray, peak: np.ndarray) -> np.ndarray:
    ph, pw = padded.shape[:2]
    left = int(np.clip(int(round(float(peak[0]))) - CROP // 2, 0, pw - CROP))
    top = int(np.clip(int(round(float(peak[1]))) - CROP // 2, 0, ph - CROP))
    return padded[top : top + CROP, left : left + CROP]


def _jittered_crop(
    padded: np.ndarray, peak: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Crop containing the peak at a random offset rather than dead centre."""
    ph, pw = padded.shape[:2]
    px, py = int(round(float(peak[0]))), int(round(float(peak[1])))
    # Offset of the peak within the crop, drawn away from the crop border.
    off_x = int(rng.integers(JITTER_EDGE_PAD, CROP - JITTER_EDGE_PAD))
    off_y = int(rng.integers(JITTER_EDGE_PAD, CROP - JITTER_EDGE_PAD))
    left = int(np.clip(px - off_x, 0, pw - CROP))
    top = int(np.clip(py - off_y, 0, ph - CROP))
    return padded[top : top + CROP, left : left + CROP]


def _random_crop(
    padded: np.ndarray,
    centroids: np.ndarray,
    rng: np.random.Generator,
    margin: int | None,
) -> np.ndarray | None:
    """Random window; when margin is set, reject windows near any centroid."""
    ph, pw = padded.shape[:2]
    for _ in range(50):
        top = int(rng.integers(0, ph - CROP + 1))
        left = int(rng.integers(0, pw - CROP + 1))
        if margin is not None and _crop_within_margin(
            top, left, CROP, centroids, margin=margin
        ):
            continue
        return padded[top : top + CROP, left : left + CROP]
    return None


@torch.no_grad()
def _score(model: torch.nn.Module, crops: list[np.ndarray], device: str) -> np.ndarray:
    """P(hit) for each crop, batched."""
    out: list[np.ndarray] = []
    for start in range(0, len(crops), 64):
        batch = np.stack(crops[start : start + 64])[:, None, :, :]
        x = torch.from_numpy(batch).float().to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1]
        out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument(
        "--split",
        default=SPLIT_CROSS_DETECTOR,
        choices=[SPLIT_VAL, SPLIT_IN_DOMAIN_TEST, SPLIT_CROSS_DETECTOR],
        help="Which sessions to draw frames from. The held-out detector "
        "(cross_detector_eval, the default) is where AP has headroom and a "
        "shortcut would show; in-domain splits are near-saturated.",
    )
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="Write results JSON here.")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    val_ids = [sid for sid, s in split["splits"].items() if s == args.split]
    if not val_ids:
        raise SystemExit(f"fold {args.fold}: no sessions in the '{args.split}' split")
    print(
        f"fold {args.fold}: {len(val_ids)} '{args.split}' sessions, "
        f"held out {fold['test_detector']}"
    )

    model = build_ssl_classifier(cfg, mae_checkpoint=None)
    # weights_only=True is sufficient: we read only model_state_dict and epoch.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Round-robin the val sessions so no single session dominates the pool.
    frames: list[tuple[Path, int, str | None]] = []
    descs: dict[str, str | None] = {}
    for sid in val_ids:
        path = session_map[sid]
        descs[sid] = read_detector_description(path)
    per_session = max(1, args.max_frames // len(val_ids))
    for sid in val_ids:
        path = session_map[sid]
        with __import__("h5py").File(path, "r") as f:
            n = int(f[cfg["lodo"].get("label_key", "entry_1/labels/hit")].shape[0])
        for i in rng.choice(n, size=min(per_session, n), replace=False):
            frames.append((path, int(i), descs[sid]))
    rng.shuffle(frames)
    frames = frames[: args.max_frames]
    print(f"scoring {len(frames)} frames")

    geom_cache: dict[Path, dict[str, float]] = {}
    holder: list = [None]

    centred, jittered, blank_neg, hard_neg = [], [], [], []
    n_hit = n_blank = 0

    for k, (path, idx, desc) in enumerate(frames):
        if k and k % 50 == 0:
            print(f"  {k}/{len(frames)}  hit={n_hit} blank={n_blank}", flush=True)
        assembled, valid_mask, centroids = _load_gcn_frame(
            path, idx, desc, geom_cache, hitfinder, holder
        )
        padded = _stack(assembled, valid_mask)

        if centroids.shape[0] > 0:
            n_hit += 1
            shifted = centroids + PAD_BORDER_DEFAULT
            peak = shifted[int(rng.integers(0, len(shifted)))]
            centred.append(_finalise(_centred_crop(padded, peak)))
            jittered.append(_finalise(_jittered_crop(padded, peak, rng)))
            neg = _random_crop(padded, shifted, rng, margin=HARD_NEG_MARGIN)
            if neg is not None:
                hard_neg.append(_finalise(neg))
        else:
            n_blank += 1
            neg = _random_crop(padded, centroids, rng, margin=None)
            if neg is not None:
                blank_neg.append(_finalise(neg))

    print(
        f"pools: centred={len(centred)} jittered={len(jittered)} "
        f"blank_neg={len(blank_neg)} hard_neg={len(hard_neg)}"
    )
    if not centred or not blank_neg:
        raise SystemExit("empty positive or negative pool — raise --max-frames")

    s_centred = _score(model, centred, device)
    s_jittered = _score(model, jittered, device)
    s_blank = _score(model, blank_neg, device)
    s_hard = _score(model, hard_neg, device) if hard_neg else np.empty(0)

    def metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        s = np.concatenate([pos, neg])
        return {
            "ap": average_precision(y, s),
            "auc": auc_roc(y, s),
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "mean_pos_score": float(pos.mean()) if len(pos) else float("nan"),
            "mean_neg_score": float(neg.mean()) if len(neg) else float("nan"),
        }

    results = {
        "fold": args.fold,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "A_control": metrics(s_centred, s_blank),
        "B_jitter": metrics(s_jittered, s_blank),
        "C_hard_neg": metrics(s_centred, s_hard) if len(s_hard) else None,
    }
    results["delta_B_positional"] = (
        results["B_jitter"]["ap"] - results["A_control"]["ap"]
    )
    if results["C_hard_neg"]:
        results["delta_C_frame_level"] = (
            results["C_hard_neg"]["ap"] - results["A_control"]["ap"]
        )

    print("\n=== crop shortcut ablation ===")
    for key in ("A_control", "B_jitter", "C_hard_neg"):
        r = results[key]
        if r is None:
            print(f"{key:12s}  (no samples)")
            continue
        print(
            f"{key:12s}  AP={r['ap']:.4f}  AUC={r['auc']:.4f}  "
            f"pos={r['n_pos']:4d} neg={r['n_neg']:4d}  "
            f"mean(pos)={r['mean_pos_score']:.3f} mean(neg)={r['mean_neg_score']:.3f}"
        )
    print(f"\ndelta B (positional shortcut) : {results['delta_B_positional']:+.4f} AP")
    if "delta_C_frame_level" in results:
        print(
            f"delta C (frame-level shortcut): {results['delta_C_frame_level']:+.4f} AP"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
