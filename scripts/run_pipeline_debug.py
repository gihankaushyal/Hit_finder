# scripts/run_pipeline_debug.py
"""Preprocessing pipeline debug runner.

Samples ~10 frames across all four detectors and executes the full
SFX preprocessing pipeline with per-step console logging.

NOTE: Augmentation order intentionally differs from the training pipeline.
Training order: rot90 → flip → cutout → LCN.
This runner's order: rot90 → flip → LCN → cutout (cutout applied after LCN).
This is a deliberate deviation for debugging purposes — results from this
runner are NOT directly representative of the training data distribution.

Usage:
    python scripts/run_pipeline_debug.py [--config <path>] [--seed <int>] [--device <str>]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from src.preprocessing.io import (
    count_frames,
    read_embedded_labels,
    read_detector_description,
    read_frame,
)
from src.preprocessing.geometry import get_assembler, get_geometry
from src.preprocessing.normalize import gcn, lcn
from src.preprocessing.pipeline import assemble_only
from src.hitfinders.gpu import GPUHitfinder
from src.data.dataset import _crop_within_margin
from src.utils.config import load_config

# Path to the pyFAI GPU hitfinder script consumed by GPUHitfinder
_GPU_PF8_SCRIPT = Path(__file__).parent.parent / "src" / "hitfinders" / "gpu_pf8.py"

# Default number of frames to sample per detector (total = 10)
_DEFAULT_N: dict[str, int] = {"AGIPD": 3, "JUNGFRAU_4M": 3, "ePix10k": 2, "Eiger4M": 2}


def _build_frame_list(
    lodo_cfg: dict,
    n_per_detector: dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[str, Path, int]]:
    """Sample CXI files and frame indices for each detector.

    Returns a list of (detector_name, cxi_path, frame_idx) tuples.
    Picks one random CXI file per detector, then samples frame indices
    without replacement (seeded).
    """
    pattern = lodo_cfg.get("cxi_pattern", "compressed*.cxi")
    frame_list: list[tuple[str, Path, int]] = []

    for det_name, n in n_per_detector.items():
        det_dir = Path(lodo_cfg["detector_dirs"][det_name])
        files = sorted(det_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {det_dir}"
            )
        cxi_path = files[int(rng.integers(0, len(files)))]
        n_frames = count_frames(cxi_path)
        indices = rng.choice(n_frames, size=min(n, n_frames), replace=False)
        for idx in indices.tolist():
            frame_list.append((det_name, cxi_path, int(idx)))

    return frame_list
