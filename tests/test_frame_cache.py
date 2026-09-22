"""Tests for the _compute_gcn_frame / _load_gcn_frame split (Task 1 of frame-cache plan).

Uses a synthetic CXI fixture (h5py) — no real detector data required. Same
fixture style as tests/test_asymmetric_dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.dataset import _compute_gcn_frame, _load_gcn_frame
from src.hitfinders import MockHitfinder

H, W = 512, 512
N_FRAMES = 4
LABEL_KEY = "entry_1/labels/hit"
DATA_KEY = "entry_1/data_1/data"


@pytest.fixture(scope="module")
def synthetic_cxi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny 4-frame CXI file with detector/geometry metadata."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "synthetic.cxi"
    rng = np.random.default_rng(0)
    frames = rng.random((N_FRAMES, H, W)).astype(np.float32)
    labels = np.array([1, 1, 0, 0], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset(DATA_KEY, data=frames)
        f.create_dataset(LABEL_KEY, data=labels)
        det_grp = f.require_group("entry_1/instrument_1/detector_1")
        det_grp.create_dataset("description", data=np.bytes_(b"Jungfrau 4M"))
        det_grp.create_dataset("distance", data=np.float64(0.1))
        det_grp.create_dataset("x_pixel_size", data=np.float64(1e-4))
        src_grp = f.require_group("entry_1/instrument_1/source_1")
        src_grp.create_dataset("wavelength", data=np.float64(1.3e-10))
    return path


def test_compute_gcn_frame_exists_and_is_callable() -> None:
    """_compute_gcn_frame must be importable from src.data.dataset (no ImportError)."""
    assert callable(_compute_gcn_frame)


def test_load_gcn_frame_delegates_to_compute_gcn_frame(synthetic_cxi: Path) -> None:
    """_load_gcn_frame must now be a thin wrapper producing identical output to
    calling _compute_gcn_frame directly with the same arguments."""
    hitfinder = MockHitfinder()

    args_load = (synthetic_cxi, 0, "Jungfrau 4M", {}, hitfinder, [None])
    args_compute = (synthetic_cxi, 0, "Jungfrau 4M", {}, hitfinder, [None])

    frame_load, mask_load, centroids_load = _load_gcn_frame(*args_load)
    frame_compute, mask_compute, centroids_compute = _compute_gcn_frame(*args_compute)

    np.testing.assert_array_equal(frame_load, frame_compute)
    np.testing.assert_array_equal(mask_load, mask_compute)
    np.testing.assert_array_equal(centroids_load, centroids_compute)
