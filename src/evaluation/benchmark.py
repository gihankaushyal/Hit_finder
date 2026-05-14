"""Cross-detector leave-one-detector-out evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold

DETECTORS: list[str] = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_IN_DOMAIN_TEST = "in_domain_test"
SPLIT_CROSS_DETECTOR = "cross_detector_eval"


def build_lodo_folds() -> list[dict]:
    """Return the 4 leave-one-detector-out fold definitions.

    Each fold: {"fold_id": int, "test_detector": str, "train_detectors": list[str]}
    """
    return [
        {
            "fold_id": i + 1,
            "test_detector": detector,
            "train_detectors": [d for d in DETECTORS if d != detector],
        }
        for i, detector in enumerate(DETECTORS)
    ]
