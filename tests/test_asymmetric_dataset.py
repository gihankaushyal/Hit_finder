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
        det_grp.create_dataset("description", data=np.bytes_(b"Jungfrau 4M"))
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
    assert not _crop_contains_centroid(
        top=top, left=left, size=size, centroids=centroids
    )


def test_crop_contains_centroid_empty() -> None:
    """Empty centroid array (0, 2) → False."""
    centroids = np.zeros((0, 2), dtype=np.float32)
    assert not _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


def test_crop_within_margin_near() -> None:
    """Centroid just outside the crop but within the margin is rejected."""
    # crop at (100, 100) size 224; centroid at x=325, y=200 — 1 px outside right edge
    centroids = np.array([[325.0, 200.0]], dtype=np.float32)
    assert _crop_within_margin(
        top=100, left=100, size=224, centroids=centroids, margin=50
    )


def test_crop_within_margin_far() -> None:
    """Centroid well outside the margin is accepted (returns False)."""
    # crop at (0, 0) size 224; centroid at x=400, y=400 — well outside
    centroids = np.array([[400.0, 400.0]], dtype=np.float32)
    assert not _crop_within_margin(
        top=0, left=0, size=224, centroids=centroids, margin=50
    )


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


def test_path_a_crop_centres_on_peak() -> None:
    """_path_a_crop returns a crop centred on the given peak and label=1."""
    from src.data.dataset import _path_a_crop

    padded = np.arange(736 * 736, dtype=np.float32).reshape(736, 736)
    centroids = np.array([[400.0, 300.0]], dtype=np.float32)  # [x, y]
    rng = np.random.default_rng(0)

    crop, label = _path_a_crop(padded, centroids, rng, ph=736, pw=736, size=224)

    assert label == 1
    assert crop.shape == (224, 224)
    # Crop top-left should place the peak at the crop centre: cx-112, cy-112.
    expected_left = 400 - 112
    expected_top = 300 - 112
    np.testing.assert_array_equal(
        crop,
        padded[expected_top : expected_top + 224, expected_left : expected_left + 224],
    )


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
    """Metadata-hit frames with a peak found return a valid (1,224,224) crop.

    Only indices 0-3 (metadata hit, per the synthetic_cxi fixture) are
    exercised here: indices 4-7 are metadata non-hit and never reach the
    hitfinder at all (see test_non_hit_frame_never_calls_hitfinder). Label is
    not asserted to be strictly 1 here because metadata-hit frames with
    centroids now go through a 50/50 coin toss between a peak-centred crop
    (label=1) and a hard-negative crop (label=0) — see
    test_hit_frame_coin_toss_produces_both_labels for that behavior.
    """
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
    )
    for idx in range(N_HITS):  # indices 0-3 are metadata hit
        result = ds[idx]
        assert result is not None, f"item {idx}: unexpected None"
        tensor, label = result
        assert tensor.shape == (1, 224, 224), f"item {idx}: wrong shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"item {idx}: wrong dtype {tensor.dtype}"
        assert label in (0, 1), f"item {idx}: unexpected label {label}"


def test_hit_path_a_forced_returns_label_one_for_all_hit_indices(
    synthetic_cxi: Path,
) -> None:
    """hit_frac=1.0 forces Path A (rng.random() < 1.0 is always True, since
    np.random.Generator.random() draws from [0, 1)), so every metadata-hit
    index with a peak present must return label=1 individually — not just
    index 0 (which test_hit_frame_coin_toss_produces_both_labels happens to
    exercise via seed-hunting)."""
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=1.0,
    )
    for idx in range(
        N_HITS
    ):  # indices 0-3 are metadata hit (see synthetic_cxi fixture)
        result = ds[idx]
        assert result is not None, f"item {idx}: unexpected None"
        tensor, label = result
        assert tensor.shape == (1, 224, 224), f"item {idx}: wrong shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"item {idx}: wrong dtype {tensor.dtype}"
        assert (
            label == 1
        ), f"item {idx}: hit_frac=1.0 must force Path A (label=1), got {label}"


def test_hit_path_b_coin_toss_crop_satisfies_margin(synthetic_cxi: Path) -> None:
    """A successful coin-toss-triggered Path B crop must respect the 50px
    margin from every centroid.

    hit_frac=0.0 forces the coin toss to always take the Path B branch for
    metadata-hit frames with centroids present (rng.random() < 0.0 is never
    true). The peak is placed near a corner of the 736x736 padded frame
    (512 + 2*112) so plenty of clear area remains for _sample_clear_crop to
    find a valid position. We wrap _crop_within_margin to record every
    (top, left, result) it is asked to evaluate inside _sample_clear_crop's
    search loop, then confirm the position that was ultimately accepted
    (i.e. the last call before ds[idx] returns, whose result was False)
    truly clears the margin.
    """
    from unittest.mock import patch

    from src.data.dataset import _crop_within_margin as _real_crop_within_margin

    # Peak near a corner in un-padded assembled coords; _load_gcn_frame shifts
    # centroids by PAD_BORDER_DEFAULT before the crop search runs.
    peaks = np.array([[50.0, 50.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=0.0,
        hard_neg_max_attempts=50,
    )

    recorded: list[tuple[int, int, bool]] = []

    def spy(top: int, left: int, size: int, centroids: np.ndarray, margin: int = 50):
        result = _real_crop_within_margin(
            top=top, left=left, size=size, centroids=centroids, margin=margin
        )
        recorded.append((top, left, result))
        return result

    with patch("src.data.dataset._crop_within_margin", side_effect=spy):
        result = ds[0]  # index 0 is metadata hit, centroids present

    assert result is not None
    tensor, label = result
    assert tensor.shape == (1, 224, 224)
    assert label == 0, f"hit_frac=0.0 must choose Path B (label=0), got {label}"

    # The accepted position is the last recorded call — _sample_clear_crop
    # returns immediately once _crop_within_margin reports False.
    assert recorded, "expected _crop_within_margin to be called at least once"
    accepted_top, accepted_left, accepted_result = recorded[-1]
    assert accepted_result is False, (
        "the last _crop_within_margin call before ds[idx] returned must have "
        "been the accepted (clear) position"
    )

    # Independently re-verify the geometric clearance property.
    padded_peaks = peaks + PAD_BORDER_DEFAULT
    assert not _crop_within_margin(
        top=accepted_top,
        left=accepted_left,
        size=224,
        centroids=padded_peaks,
        margin=50,
    ), "accepted Path B crop must clear the 50px margin from every centroid"


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
        assert (
            result is not None
        ), f"item {idx}: unexpected None — empty centroids always yield a valid miss crop"
        tensor, label = result
        assert tensor.shape == (1, 224, 224), f"item {idx}: wrong shape {tensor.shape}"
        assert tensor.dtype == torch.float32, f"item {idx}: wrong dtype {tensor.dtype}"
        assert label == 0, f"item {idx}: expected label=0 (miss crop), got {label}"


def test_non_hit_frame_never_calls_hitfinder(synthetic_cxi: Path) -> None:
    """Metadata NON-HIT frames (indices 4-7) must never invoke find_peaks."""

    class RaisingHitfinder:
        def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
            raise AssertionError("find_peaks must not be called for a non-hit frame")

    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=RaisingHitfinder(),
        label_key=LABEL_KEY,
    )
    for idx in range(N_HITS, N_FRAMES):  # indices 4-7 are metadata non-hit
        result = ds[idx]
        assert result is not None
        tensor, label = result
        assert label == 0, f"item {idx}: metadata non-hit must yield label=0"
        assert tensor.shape == (1, 224, 224)


def test_hit_frame_coin_toss_produces_both_labels(synthetic_cxi: Path) -> None:
    """Across many seeds, a metadata-hit frame with a peak yields both labels."""
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    labels_seen: set[int] = set()
    for seed in range(30):
        ds = AsymmetricCXIDataset(
            session_ids=["s0"],
            session_map={"s0": synthetic_cxi},
            hitfinder=hf,
            label_key=LABEL_KEY,
            seed=seed,
        )
        result = ds[0]  # index 0 is metadata hit
        assert result is not None
        _, label = result
        labels_seen.add(label)
        if labels_seen == {0, 1}:
            break
    assert labels_seen == {
        0,
        1,
    }, f"expected both labels across 30 seeds from the coin toss, got {labels_seen}"


def test_hit_path_b_falls_back_to_path_a_when_margin_search_fails(
    synthetic_cxi: Path,
) -> None:
    """When peaks blanket the frame, no 50px-clear crop exists — must fall back
    to a peak-centred crop (label=1) instead of ever returning None."""
    # Dense grid of peaks covering the full 736x736 padded frame (512 + 2*112)
    # at 80px spacing, well under the 2*50=100px margin needed for any gap.
    xs = np.arange(0, 736, 80)
    ys = np.arange(0, 736, 80)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    hf = MockHitfinder(peaks=grid)
    for seed in range(10):
        ds = AsymmetricCXIDataset(
            session_ids=["s0"],
            session_map={"s0": synthetic_cxi},
            hitfinder=hf,
            label_key=LABEL_KEY,
            seed=seed,
        )
        result = ds[0]  # index 0 is metadata hit
        assert result is not None, f"seed {seed}: fallback must never return None"
        tensor, label = result
        assert tensor.shape == (1, 224, 224)
        assert label == 1, (
            f"seed {seed}: dense peak grid leaves no clear region, so every "
            f"draw (Path A directly, or Path B falling back) must land on "
            f"label=1, got {label}"
        )


def test_hit_frac_zero_forces_path_b(synthetic_cxi: Path) -> None:
    """hit_frac=0.0 means rng.random() < 0.0 is never true, so the coin toss
    always takes the Path B (hard-negative) branch for metadata-hit frames."""
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    for seed in range(5):
        ds = AsymmetricCXIDataset(
            session_ids=["s0"],
            session_map={"s0": synthetic_cxi},
            hitfinder=hf,
            label_key=LABEL_KEY,
            seed=seed,
            hit_frac=0.0,
        )
        result = ds[0]  # index 0 is metadata hit
        assert result is not None
        tensor, label = result
        assert tensor.shape == (1, 224, 224)
        assert label == 0, (
            f"seed {seed}: hit_frac=0.0 must always choose Path B "
            f"(a clear crop exists far from the single peak), got label={label}"
        )


def test_hard_neg_max_attempts_zero_always_falls_back_to_path_a(
    synthetic_cxi: Path,
) -> None:
    """hard_neg_max_attempts=0 means _sample_clear_crop's loop runs zero times
    and immediately returns None, forcing the Path A fallback (label=1)."""
    peaks = np.array([[256.0, 256.0]], dtype=np.float32)
    hf = MockHitfinder(peaks=peaks)
    ds = AsymmetricCXIDataset(
        session_ids=["s0"],
        session_map={"s0": synthetic_cxi},
        hitfinder=hf,
        label_key=LABEL_KEY,
        hit_frac=0.0,  # always attempt Path B first
        hard_neg_max_attempts=0,  # ...but immediately fail and fall back
    )
    result = ds[0]  # index 0 is metadata hit, centroids present
    assert result is not None
    tensor, label = result
    assert tensor.shape == (1, 224, 224)
    assert label == 1, (
        "hard_neg_max_attempts=0 must exhaust the margin search instantly and "
        f"fall back to _path_a_crop (label=1), got label={label}"
    )


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
    assert call_order.index("find_peaks") < call_order.index(
        "gcn"
    ), "find_peaks must run before gcn"


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


@pytest.fixture(scope="module")
def malformed_label_cxi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create an 8-frame CXI file where one embedded label is out of range.

    Same layout as synthetic_cxi, but frame index 2 (nominally a hit) is
    corrupted to -1 to exercise the out-of-range label warning.
    """
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "malformed.cxi"
    rng = np.random.default_rng(42)
    frames = rng.random((N_FRAMES, H, W)).astype(np.float32)
    labels = np.array([1, 1, -1, 1, 0, 0, 0, 0], dtype=np.float32)
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


def test_out_of_range_label_warns_and_still_functions(
    malformed_label_cxi: Path,
) -> None:
    """An out-of-range embedded label (e.g. -1) triggers a UserWarning but the
    dataset still builds correctly and the frame is treated as non-hit."""
    hf = MockHitfinder()
    with pytest.warns(UserWarning, match="out-of-range embedded label"):
        ds = AsymmetricCXIDataset(
            session_ids=["s0"],
            session_map={"s0": malformed_label_cxi},
            hitfinder=hf,
            label_key=LABEL_KEY,
        )
    assert len(ds) == N_FRAMES

    result = ds[2]  # the corrupted frame (raw label -1)
    assert result is not None
    tensor, label = result
    assert tensor.shape == (1, 224, 224)
    assert label == 0, f"out-of-range label must be treated as non-hit, got {label}"
