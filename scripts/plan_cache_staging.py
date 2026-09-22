#!/usr/bin/env python
"""Emit frame-cache entry directories for one LODO fold in staging priority order.

Train sessions are read once per epoch, so they go to NVMe first. Val sessions
are also read once per epoch but are 8x smaller, so they follow. In-domain and
cross-detector sessions are read exactly once, after training, and are never
staging candidates — streaming them from NFS costs seconds, once.

The split is recomputed here rather than read from checkpoints/asymmetric_splits/,
because staging runs before the training job that writes that artifact. Both use
build_session_stratified_split with the same seed, so they agree by construction.

Usage:
    python scripts/plan_cache_staging.py --config configs/ssl/mae_finetune.yaml --fold 1
Output (stdout, one per line, highest priority first):
    agipd_20k/compressed_000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark import (  # noqa: E402
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
)
from src.training.lodo import build_sessions  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Read-frequency order: every-epoch splits only, largest consumer first.
STAGING_PRIORITY = (SPLIT_TRAIN, SPLIT_VAL)


def staging_order(
    split_artifact: dict, session_map: dict[str, Path]
) -> list[tuple[str, str]]:
    """Return (detector_dir_name, cxi_stem) pairs in staging priority order."""
    order: list[tuple[str, str]] = []
    for split in STAGING_PRIORITY:
        ids = sorted(sid for sid, s in split_artifact["splits"].items() if s == split)
        for sid in ids:
            path = Path(session_map[sid])
            order.append((path.parent.name, path.stem))
    return order


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    sessions, session_map = build_sessions(cfg["lodo"])
    fold = next((f for f in build_lodo_folds() if f["fold_id"] == args.fold), None)
    if fold is None:
        raise ValueError(f"no such fold: {args.fold}")
    split_artifact = build_session_stratified_split(
        sessions,
        test_detector=fold["test_detector"],
        fold=fold["fold_id"],
        seed=cfg["seed"],
    )
    for det, stem in staging_order(split_artifact, session_map):
        print(f"{det}/{stem}")


if __name__ == "__main__":
    main()
