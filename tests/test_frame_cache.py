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


# ---------------------------------------------------------------------------
# FrameCache + manifest
# ---------------------------------------------------------------------------

from src.data.frame_cache import (  # noqa: E402
    CACHE_DTYPE,
    CacheMissError,
    CacheStaleError,
    FrameCache,
    build_manifest,
    cache_key,
    entry_dir,
    verify_manifest,
    write_manifest,
)

CFG = {
    "hitfinder": {
        "backend": "pf8",
        "pf8_threshold": 800.0,
        "pf8_min_snr": 5.0,
        "pf8_min_pix_count": 2,
        "pf8_max_pix_count": 200,
        "pf8_local_bg_radius": 3,
        "pf8_min_res": 0,
        "pf8_max_res": 0,
        "pf8_use_saturated": False,
    }
}


def _write_cache_entry(
    root: Path, det: str, stem: str, frames: np.ndarray, centroids: list[np.ndarray]
) -> None:
    """Write one detector's mask and one CXI entry into a cache root."""
    (root / det).mkdir(parents=True, exist_ok=True)
    np.save(root / det / "valid_mask.npy", np.ones(frames.shape[1:], dtype=bool))
    d = root / det / stem
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "frames.npy", frames.astype(CACHE_DTYPE))
    np.savez(
        d / "centroids.npz",
        **{f"c_{i:05d}": c.astype(np.float32) for i, c in enumerate(centroids)},
    )


def test_cache_key_uses_parent_dir_and_stem(tmp_path: Path) -> None:
    """Keys are (detector_dir_name, cxi_stem) so NFS and NVMe paths collide."""
    nfs = tmp_path / "production" / "agipd_20k" / "compressed_000.cxi"
    nvme = tmp_path / "stage" / "agipd_20k" / "compressed_000.cxi"
    assert cache_key(nfs) == ("agipd_20k", "compressed_000")
    assert cache_key(nfs) == cache_key(nvme)


def test_entry_dir_layout(tmp_path: Path) -> None:
    p = tmp_path / "agipd_20k" / "compressed_007.cxi"
    assert entry_dir(tmp_path / "cache", p) == (
        tmp_path / "cache" / "agipd_20k" / "compressed_007"
    )


def test_get_returns_frame_mask_centroids(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    frames = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    cents = [np.array([[1.0, 2.0]], np.float32) for _ in range(3)]
    _write_cache_entry(root, "agipd_20k", "compressed_000", frames, cents)

    fc = FrameCache([root])
    frame, mask, cent = fc.get(tmp_path / "agipd_20k" / "compressed_000.cxi", 1)
    assert frame.dtype == np.float32
    np.testing.assert_allclose(frame, frames[1].astype(CACHE_DTYPE), rtol=0, atol=0.5)
    assert mask.shape == (4, 5) and mask.all()
    np.testing.assert_array_equal(cent, cents[1])


def test_first_root_wins(tmp_path: Path) -> None:
    """NVMe tier shadows the NFS tier for the same entry."""
    nvme, nfs = tmp_path / "nvme", tmp_path / "nfs"
    a = np.full((2, 4, 4), 1.0, np.float32)
    b = np.full((2, 4, 4), 9.0, np.float32)
    cents = [np.zeros((0, 2), np.float32)] * 2
    _write_cache_entry(nvme, "agipd_20k", "c0", a, cents)
    _write_cache_entry(nfs, "agipd_20k", "c0", b, cents)

    fc = FrameCache([nvme, nfs])
    frame, _, _ = fc.get(tmp_path / "agipd_20k" / "c0.cxi", 0)
    assert frame[0, 0] == pytest.approx(1.0)


def test_falls_through_to_second_root(tmp_path: Path) -> None:
    """A partially staged NVMe tier is normal — missing entries resolve on NFS."""
    nvme, nfs = tmp_path / "nvme", tmp_path / "nfs"
    nvme.mkdir()
    frames = np.full((2, 4, 4), 7.0, np.float32)
    _write_cache_entry(
        nfs, "agipd_20k", "c0", frames, [np.zeros((0, 2), np.float32)] * 2
    )

    fc = FrameCache([nvme, nfs])
    frame, _, _ = fc.get(tmp_path / "agipd_20k" / "c0.cxi", 0)
    assert frame[0, 0] == pytest.approx(7.0)


def test_total_miss_raises_cache_miss_error(tmp_path: Path) -> None:
    fc = FrameCache([tmp_path / "nvme", tmp_path / "nfs"])
    with pytest.raises(CacheMissError):
        fc.get(tmp_path / "agipd_20k" / "nope.cxi", 0)


def test_frame_index_out_of_range_raises_cache_miss_error(tmp_path: Path) -> None:
    """A truncated entry must miss loudly, not return the wrong frame."""
    root = tmp_path / "cache"
    _write_cache_entry(
        root,
        "agipd_20k",
        "c0",
        np.zeros((2, 4, 4), np.float32),
        [np.zeros((0, 2), np.float32)] * 2,
    )
    fc = FrameCache([root])
    with pytest.raises(CacheMissError):
        fc.get(tmp_path / "agipd_20k" / "c0.cxi", 5)


def test_returned_mask_is_read_only(tmp_path: Path) -> None:
    """The per-detector mask is shared across calls, so callers must not mutate it."""
    root = tmp_path / "cache"
    _write_cache_entry(
        root,
        "agipd_20k",
        "c0",
        np.zeros((1, 4, 4), np.float32),
        [np.zeros((0, 2), np.float32)],
    )
    fc = FrameCache([root])
    _, mask, _ = fc.get(tmp_path / "agipd_20k" / "c0.cxi", 0)
    with pytest.raises(ValueError):
        mask[0, 0] = False


def test_manifest_roundtrip_verifies(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    write_manifest(root, CFG)
    verify_manifest(root, CFG)  # must not raise


def test_manifest_detects_changed_pf8_param(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    write_manifest(root, CFG)
    changed = {"hitfinder": dict(CFG["hitfinder"], pf8_threshold=900.0)}
    with pytest.raises(CacheStaleError, match="pf8_threshold"):
        verify_manifest(root, changed)


def test_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CacheStaleError, match="no manifest"):
        verify_manifest(tmp_path / "empty", CFG)


def test_manifest_git_sha_is_recorded_but_not_compared(tmp_path: Path) -> None:
    """A new commit must not invalidate a 469 GB cache."""
    root = tmp_path / "cache"
    root.mkdir()
    write_manifest(root, CFG)
    import json

    m = json.loads((root / "cache_manifest.json").read_text())
    assert "git_sha" in m
    m["git_sha"] = "deadbeef"
    (root / "cache_manifest.json").write_text(json.dumps(m))
    verify_manifest(root, CFG)  # must not raise


def test_build_manifest_includes_pipeline_constants() -> None:
    params = build_manifest(CFG)["params"]
    assert params["lcn_window"] == 9
    assert params["lcn_eps"] == pytest.approx(1e-2)
    assert params["edge_erosion_px"] == 2
    assert params["cache_dtype"] == "float16"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_builder_writes_readable_entry(synthetic_cxi: Path, tmp_path: Path) -> None:
    """Building one CXI produces an entry FrameCache.get() can read back."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_frame_cache import build_one_cxi

    root = tmp_path / "cache"
    n = build_one_cxi(synthetic_cxi, root, CFG, backend="mock")
    assert n == N_FRAMES

    fc = FrameCache([root])
    frame, mask, cent = fc.get(synthetic_cxi, 0)
    assert frame.shape == (H, W)
    assert mask.shape == (H, W)
    assert cent.ndim == 2 and cent.shape[1] == 2


def test_builder_is_atomic_no_tmp_left_behind(
    synthetic_cxi: Path, tmp_path: Path
) -> None:
    """No .tmp directory survives a successful build."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_frame_cache import build_one_cxi

    root = tmp_path / "cache"
    build_one_cxi(synthetic_cxi, root, CFG, backend="mock")
    assert not list(root.rglob("*.tmp"))


def test_builder_matches_live_pipeline(synthetic_cxi: Path, tmp_path: Path) -> None:
    """Cached frame equals _compute_gcn_frame output within fp16 tolerance.

    This is the bit-exactness gate for the builder: mask and centroids must
    match exactly; the frame only loses fp16 precision.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_frame_cache import build_one_cxi

    root = tmp_path / "cache"
    build_one_cxi(synthetic_cxi, root, CFG, backend="mock")
    fc = FrameCache([root])

    for idx in range(N_FRAMES):
        want_f, want_m, want_c = _compute_gcn_frame(
            synthetic_cxi, idx, "Jungfrau 4M", {}, MockHitfinder(), [None]
        )
        got_f, got_m, got_c = fc.get(synthetic_cxi, idx)
        np.testing.assert_array_equal(got_m, want_m)
        np.testing.assert_array_equal(got_c, want_c)
        scale = max(float(np.abs(want_f).max()), 1e-6)
        assert np.abs(got_f - want_f).max() / scale < 1e-3
