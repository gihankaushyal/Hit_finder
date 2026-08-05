"""Tests for AsymmetricCXIDataset and crop helper functions.

Uses a synthetic CXI fixture (h5py) — no real detector data required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.dataset import (
    AsymmetricCXIDataset,
    _crop_contains_centroid,
    _crop_within_margin,
)
from src.hitfinders import MockHitfinder
from src.preprocessing.augment import PAD_BORDER_DEFAULT, pad_border

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
        det_grp = f.require_group("entry_1/instrument_1/detector_1")
        det_grp.create_dataset("distance", data=np.float64(0.1))
        det_grp.create_dataset("x_pixel_size", data=np.float64(1e-4))
        src_grp = f.require_group("entry_1/instrument_1/source_1")
        src_grp.create_dataset("wavelength", data=np.float64(1.3e-10))
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


def test_crop_within_margin_near() -> None:
    """Centroid just outside the crop but within the margin is rejected."""
    # crop at (100, 100) size 224; centroid at x=325, y=200 — 1 px outside right edge
    centroids = np.array([[325.0, 200.0]], dtype=np.float32)
    assert _crop_within_margin(top=100, left=100, size=224, centroids=centroids, margin=50)


def test_crop_within_margin_far() -> None:
    """Centroid well outside the margin is accepted (returns False)."""
    # crop at (0, 0) size 224; centroid at x=400, y=400 — well outside
    centroids = np.array([[400.0, 400.0]], dtype=np.float32)
    assert not _crop_within_margin(top=0, left=0, size=224, centroids=centroids, margin=50)


def test_crop_within_margin_empty() -> None:
    """Empty centroid array → False (any position is safe)."""
    centroids = np.zeros((0, 2), dtype=np.float32)
    assert not _crop_within_margin(top=0, left=0, size=224, centroids=centroids)


def test_pad_border_centroid_shift() -> None:
    """After pad_border, centroids shifted by PAD_BORDER_DEFAULT land in the same
    relative position as the original centroids did in the unpadded image."""
    p = PAD_BORDER_DEFAULT
    img = np.random.default_rng(0).random((512, 512)).astype(np.float32)
    padded = pad_border(img)

    # A centroid at (r, c) in the original maps to (r+p, c+p) in the padded image.
    r, c = 100, 200
    original_value = img[r, c]
    shifted_value = padded[r + p, c + p]
    assert original_value == shifted_value

    # Verify shifted centroids are valid for _crop_contains_centroid.
    centroids_orig = np.array([[float(c), float(r)]], dtype=np.float32)  # (x, y)
    centroids_shifted = centroids_orig + p
    # A crop at the shifted position should contain the shifted centroid.
    assert _crop_contains_centroid(top=r, left=c, size=224, centroids=centroids_shifted)


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


def test_hit_path_returns_crop_shape_and_label_one(synthetic_cxi: Path) -> None:
    """When hitfinder returns peaks, __getitem__ takes Path A: (1,224,224) tensor, label=1."""
    # Peak at centre of 512×512 frame — well within padded bounds
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
    )
    for idx in range(len(ds)):
        result = ds[idx]
        assert result is not None, f"item {idx}: unexpected None from Path A"
        tensor, label = result
        assert tensor.shape == (1, 224, 224), f"item {idx}: wrong shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"item {idx}: wrong dtype {tensor.dtype}"
        assert label == 1, f"item {idx}: expected label=1 (hit crop), got {label}"


def test_miss_path_returns_crop_shape_and_label_zero(synthetic_cxi: Path) -> None:
    """When hitfinder finds no peaks, __getitem__ takes Path B: (1,224,224) tensor, label=0."""
    hf = MockHitfinder()  # returns empty (0, 2) centroid array
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
    )
    for idx in range(len(ds)):
        result = ds[idx]
        assert result is not None, f"item {idx}: unexpected None — empty centroids always yield a valid miss crop"
        tensor, label = result
        assert tensor.shape == (1, 224, 224), f"item {idx}: wrong shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"item {idx}: wrong dtype {tensor.dtype}"
        assert label == 0, f"item {idx}: expected label=0 (miss crop), got {label}"


def test_crop_is_normalised(synthetic_cxi: Path) -> None:
    """Returned tensor values are not in raw detector range — GCN+LCN has been applied."""
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
    )
    tensor, _ = ds[0]
    arr = tensor.numpy()
    # Raw synthetic frames are uniform random in [0, 1); after GCN+LCN the mean
    # should be near 0 and values span well beyond [0, 1).
    assert arr.mean() == pytest.approx(0.0, abs=0.5), "GCN should shift mean toward 0"
    assert arr.max() > 1.0 or arr.min() < 0.0, "LCN should produce values outside [0,1]"


def test_hitfinder_runs_before_gcn(synthetic_cxi: Path) -> None:
    """find_peaks must be called on the raw assembled frame, before gcn()."""
    from unittest.mock import patch

    call_order: list[str] = []

    class OrderTrackingHitfinder:
        def find_peaks(self, frame: np.ndarray) -> np.ndarray:
            call_order.append("find_peaks")
            return np.zeros((0, 2), dtype=np.float32)

    def recording_gcn(frame: np.ndarray) -> np.ndarray:
        call_order.append("gcn")
        return frame

    with patch("src.data.dataset.gcn", side_effect=recording_gcn):
        ds = AsymmetricCXIDataset(
            session_ids=["s0"],
            session_map={"s0": synthetic_cxi},
            hitfinder=OrderTrackingHitfinder(),
        )
        ds[0]

    assert "find_peaks" in call_order
    assert "gcn" in call_order
    assert call_order.index("find_peaks") < call_order.index("gcn"), (
        "find_peaks must run before gcn"
    )


def test_set_geometry_called_with_cxi_params(synthetic_cxi: Path) -> None:
    """AsymmetricCXIDataset calls set_geometry with dist/wavelength/pixel_size."""
    set_geom_calls: list[dict] = []

    class GeomCapturingHitfinder:
        def set_geometry(self, **kwargs: float) -> None:
            set_geom_calls.append(kwargs)

        def find_peaks(self, frame: np.ndarray) -> np.ndarray:
            return np.zeros((0, 2), dtype=np.float32)

    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=GeomCapturingHitfinder(),
    )
    ds[0]

    assert len(set_geom_calls) >= 1
    call = set_geom_calls[0]
    assert "dist" in call
    assert "wavelength" in call
    assert "pixel_size" in call
    assert call["dist"] == pytest.approx(0.1)
    assert call["wavelength"] == pytest.approx(1.3e-10)
    assert call["pixel_size"] == pytest.approx(1e-4)
