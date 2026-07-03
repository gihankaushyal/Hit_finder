"""Tests for AsymmetricCXIDataset and _crop_contains_centroid.

Uses a synthetic CXI fixture (h5py) — no real detector data required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.dataset import AsymmetricCXIDataset, _crop_contains_centroid
from src.hitfinders import MockHitfinder

# ---------------------------------------------------------------------------
# Synthetic CXI fixture
# ---------------------------------------------------------------------------

H, W = 512, 512
N_FRAMES = 8
N_HITS = 4
LABEL_KEY = "entry_1/labels/hit"
DATA_KEY = "entry_1/data_1/data"


@pytest.fixture(scope="module")
def synthetic_cxi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny 8-frame CXI file with 4 hits and 4 misses."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "synthetic.cxi"
    rng = np.random.default_rng(42)
    frames = rng.random((N_FRAMES, H, W)).astype(np.float32)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset(DATA_KEY, data=frames)
        f.create_dataset(LABEL_KEY, data=labels)
    return path


# ---------------------------------------------------------------------------
# _crop_contains_centroid tests
# ---------------------------------------------------------------------------


def test_crop_contains_centroid_inside() -> None:
    """Centroid at the image centre falls inside a centred crop."""
    centroids = np.array([[256.0, 256.0]], dtype=np.float32)  # [x, y]
    # crop starting at (100, 100) with size 224 covers [100..324) × [100..324)
    assert _crop_contains_centroid(top=100, left=100, size=224, centroids=centroids)


def test_crop_contains_centroid_boundary_excluded() -> None:
    """Centroid exactly at top+size (exclusive upper bound) is outside."""
    size = 224
    top, left = 100, 100
    # x = left + size = 324 → outside (exclusive)
    centroids = np.array([[float(left + size), float(top + size)]], dtype=np.float32)
    assert not _crop_contains_centroid(top=top, left=left, size=size, centroids=centroids)


def test_crop_contains_centroid_empty() -> None:
    """Empty centroid array (0, 2) → False."""
    centroids = np.zeros((0, 2), dtype=np.float32)
    assert not _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


# ---------------------------------------------------------------------------
# AsymmetricCXIDataset structural tests
# ---------------------------------------------------------------------------


def test_len(synthetic_cxi: Path) -> None:
    """Dataset length equals total frames in the CXI file."""
    hf = MockHitfinder()
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
    )
    assert len(ds) == N_FRAMES


def test_getitem_returns_correct_shape_dtype(synthetic_cxi: Path) -> None:
    """Each item is a (1, 224, 224) float32 tensor and an int label 0 or 1."""
    hf = MockHitfinder()
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
    )
    for idx in range(len(ds)):
        tensor, label = ds[idx]
        assert isinstance(tensor, torch.Tensor), f"item {idx}: expected Tensor"
        assert tensor.shape == (1, 224, 224), f"item {idx}: wrong shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"item {idx}: wrong dtype {tensor.dtype}"
        assert label in (0, 1), f"item {idx}: label out of range {label}"


# ---------------------------------------------------------------------------
# Crop-labelling logic tests
# ---------------------------------------------------------------------------


def test_hit_patch_label_with_centroids(synthetic_cxi: Path) -> None:
    """With centroids scattered across the frame and hit_frac=1.0, at least 1/20
    samples from a hit frame gets label=1."""
    peaks = np.array(
        [[100.0, 100.0], [300.0, 300.0], [450.0, 450.0]], dtype=np.float32
    )
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=1.0,
    )
    # Indices 0–3 are hit frames
    hit_frame_indices = [i for i in range(N_FRAMES) if ds._labels[i] == 1]
    assert hit_frame_indices, "Expected at least one hit frame"

    label_ones = 0
    n_samples = 20
    rng = np.random.default_rng(1)
    for _ in range(n_samples):
        idx = int(rng.choice(hit_frame_indices))
        _, label = ds[idx]
        if label == 1:
            label_ones += 1

    assert label_ones >= 1, (
        f"Expected >=1 label=1 out of {n_samples} samples from hit frames, got {label_ones}"
    )


def test_hard_negative_label_zero(synthetic_cxi: Path) -> None:
    """With hit_frac=0.0, crops from hit frames are always hard negatives → label=0."""
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=0.0,
    )
    hit_frame_indices = [i for i in range(N_FRAMES) if ds._labels[i] == 1]
    for idx in hit_frame_indices:
        _, label = ds[idx]
        assert label == 0, f"Expected hard-negative label=0 for hit frame idx={idx}, got {label}"


def test_miss_frame_always_label_zero(synthetic_cxi: Path) -> None:
    """Non-hit frames always produce label=0 regardless of centroids."""
    peaks = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=1.0,
    )
    miss_frame_indices = [i for i in range(N_FRAMES) if ds._labels[i] == 0]
    assert miss_frame_indices, "Expected at least one non-hit frame"
    for idx in miss_frame_indices:
        _, label = ds[idx]
        assert label == 0, f"Non-hit frame idx={idx} produced label={label}"


def test_no_centroids_fallback(synthetic_cxi: Path) -> None:
    """MockHitfinder returning empty (0,2) → hit frame falls back to label=0 without crash."""
    hf = MockHitfinder()  # default: no peaks
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=1.0,
    )
    hit_frame_indices = [i for i in range(N_FRAMES) if ds._labels[i] == 1]
    for idx in hit_frame_indices:
        tensor, label = ds[idx]
        assert label == 0, (
            f"With no centroids, hit frame idx={idx} should fall back to label=0, got {label}"
        )
        assert tensor.shape == (1, 224, 224)
