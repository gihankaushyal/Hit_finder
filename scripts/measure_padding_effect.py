"""Does the share of invalid pixels in an evaluation tile move the frame-level AP?

Evaluation tiles are never zero-padded, but panel gaps leave some tiles mostly invalid
(AGIPD: 6 of 25 tiles under half valid). The frame score is the fraction of ALL tiles that vote
"hit", so a gap-heavy tile counts as much as a full one. This script runs one inference pass over
frames of a held-out split, keeps every tile's score and valid-pixel fraction, then re-scores the
frames offline with only the tiles that are at least `thr` valid (a frame with no such tile keeps
all of its tiles). It reports AP/AUC for each threshold, so the effect of gap-heavy tiles can be
read off without retraining and without changing the evaluation protocol.

Tiles are scored with the cached-path semantics of run_patch_agg (masked torch LCN on the
stride-224 grid of the GCN'd, gap-filled frame).

Usage:
    python scripts/measure_padding_effect.py --arch vit --config configs/ssl/mae_finetune.yaml \\
        --checkpoint checkpoints/vits16-mae-finetune-v2-fold4-seed42/best.pt --fold 4 \\
        --out docs/figures/padding_effect/fold4_finetune.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import _load_gcn_frame  # noqa: E402
from src.evaluation.benchmark import (  # noqa: E402
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_VAL,
    _tie_aware_ap,
    _tie_aware_auc,
    build_lodo_folds,
    build_session_stratified_split,
)
from src.models.ssl import build_ssl_classifier  # noqa: E402
from src.models.supervised import build_supervised_model  # noqa: E402
from src.preprocessing.augment import patch_grid  # noqa: E402
from src.preprocessing.io import read_detector_description  # noqa: E402
from src.preprocessing.normalize import lcn_torch  # noqa: E402
from src.training.lodo import build_sessions  # noqa: E402
from src.utils.config import load_config  # noqa: E402

PATCH = 224
VOTE_THRESHOLD = 0.5
VALID_THRESHOLDS = (
    0.0,
    0.5,
    0.75,
)  # 0.0 = all tiles (the evaluation protocol as it is)


def vote_scores(
    tile_scores: list[np.ndarray], tile_valid: list[np.ndarray], thr: float
) -> np.ndarray:
    """Fraction of kept tiles voting hit, per frame.

    Kept = tiles with valid fraction >= thr; a frame with none keeps all its tiles.
    """
    out = []
    for s, v in zip(tile_scores, tile_valid):
        keep = v >= thr
        if not keep.any():
            keep = np.ones_like(keep)
        out.append(float((s[keep] > VOTE_THRESHOLD).mean()))
    return np.asarray(out)


def rescore(
    tile_scores: list[np.ndarray], tile_valid: list[np.ndarray], labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """Tie-aware AP/AUC of the vote score for each valid-fraction threshold."""
    result = {}
    for thr in VALID_THRESHOLDS:
        vote = vote_scores(tile_scores, tile_valid, thr)
        result[f"valid>={thr}"] = {
            "ap_tieaware": _tie_aware_ap(labels, vote),
            "auc_tieaware": _tie_aware_auc(labels, vote),
            "n_distinct_scores": int(len(np.unique(vote))),
            "mean_tiles_dropped": float(np.mean([(v < thr).sum() for v in tile_valid])),
        }
    return result


@torch.no_grad()
def score_frame(
    model: torch.nn.Module, gcn_frame: np.ndarray, valid_mask: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """P(hit) and valid-pixel fraction of every stride-224 tile of one frame."""
    tiles = patch_grid(gcn_frame, PATCH, PATCH)
    masks = patch_grid(valid_mask, PATCH, PATCH)
    x = lcn_torch(
        torch.from_numpy(np.stack(tiles)).to(device),
        masks=torch.from_numpy(np.stack(masks)).to(device),
    ).unsqueeze(1)
    scores = []
    for start in range(0, len(x), 64):
        scores.append(torch.softmax(model(x[start : start + 64]), dim=1)[:, 1].cpu())
    frac = np.asarray([float(np.asarray(m).mean()) for m in masks])
    return torch.cat(scores).numpy(), frac


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument(
        "--split",
        default=SPLIT_CROSS_DETECTOR,
        choices=[SPLIT_VAL, SPLIT_IN_DOMAIN_TEST, SPLIT_CROSS_DETECTOR],
    )
    p.add_argument("--arch", default="vit", choices=["vit", "resnet"])
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    sessions, session_map = build_sessions(cfg["lodo"])
    fold = next(f for f in build_lodo_folds() if f["fold_id"] == args.fold)
    split = build_session_stratified_split(
        sessions,
        test_detector=fold["test_detector"],
        fold=fold["fold_id"],
        seed=cfg["seed"],
    )
    ids = [sid for sid, s in split["splits"].items() if s == args.split]
    if not ids:
        raise SystemExit(f"fold {args.fold}: no sessions in the '{args.split}' split")

    if args.arch == "resnet":
        model = build_supervised_model(
            cfg["model"]["backbone"], pretrained=False, num_classes=2
        )
    else:
        model = build_ssl_classifier(cfg, mae_checkpoint=None)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    label_key = cfg["lodo"].get("label_key", "entry_1/labels/hit")
    per_session = max(1, args.max_frames // len(ids))
    frames: list[tuple[Path, int, str | None, int]] = []
    for sid in ids:
        path = session_map[sid]
        desc = read_detector_description(path)
        with h5py.File(path, "r") as f:
            labels = np.round(f[label_key][()]).astype(int)
        for i in rng.choice(
            len(labels), size=min(per_session, len(labels)), replace=False
        ):
            frames.append((path, int(i), desc, int(labels[i])))
    rng.shuffle(frames)
    frames = frames[: args.max_frames]
    print(f"scoring {len(frames)} frames, checkpoint epoch {ckpt.get('epoch', '?')}")

    geom_cache: dict[Path, dict[str, float]] = {}
    holder: list = [None]
    tile_scores, tile_valid, labels_out = [], [], []
    for k, (path, idx, desc, label) in enumerate(frames):
        if k and k % 100 == 0:
            print(f"  {k}/{len(frames)}", flush=True)
        gcn_frame, valid_mask, _ = _load_gcn_frame(
            path, idx, desc, geom_cache, None, holder  # peaks are not needed here
        )
        s, v = score_frame(model, gcn_frame, valid_mask, device)
        tile_scores.append(s)
        tile_valid.append(v)
        labels_out.append(label)

    y = np.asarray(labels_out)
    result = {
        "checkpoint": args.checkpoint,
        "fold": args.fold,
        "split": args.split,
        "n_frames": int(len(y)),
        "n_tiles_per_frame": int(np.median([len(s) for s in tile_scores])),
        "tile_valid_fraction_quartiles": np.percentile(
            np.concatenate(tile_valid), [25, 50, 75]
        ).tolist(),
        "rescored": rescore(tile_scores, tile_valid, y),
    }
    for name, m in result["rescored"].items():
        print(
            f"{name:14s} AP={m['ap_tieaware']:.4f} AUC={m['auc_tieaware']:.4f} "
            f"distinct={m['n_distinct_scores']} dropped/frame={m['mean_tiles_dropped']:.2f}"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
