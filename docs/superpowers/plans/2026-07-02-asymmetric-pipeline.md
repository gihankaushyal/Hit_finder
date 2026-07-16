# Asymmetric Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LODO training pipeline with an asymmetric strategy where PeakFinder8 (or a GPU hitfinder) generates peak centroids on-the-fly during training to assign patch-level labels, while validation uses a blind 224×224 sliding-window grid with vote-count aggregation.

**Architecture:** A new `src/hitfinders/` module defines a pluggable `Hitfinder` Protocol (PF8 C++ wrapper and GPU stub), swapped at config time. `AsymmetricCXIDataset` uses the hitfinder in `__getitem__` to assign labels based on crop content (≥1 centroid inside → label 1). Validation calls the existing `run_patch_agg` with a new `aggregation="vote"` parameter: frame score = hit-tile-count / n-tiles; frame binary = count ≥ 3.

**Tech Stack:** PyTorch, h5py, NumPy, Reborn geometry, existing `gcn`/`lcn`/`random_crop`/`random_rot90`/`random_flip`/`random_cutout` from `src/preprocessing/`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/hitfinders/__init__.py` | Re-exports + `get_hitfinder(cfg)` factory |
| Create | `src/hitfinders/base.py` | `Hitfinder` Protocol + `MockHitfinder` for tests |
| Create | `src/hitfinders/pf8.py` | `PF8Hitfinder` — C++ PF8 wrapper stub |
| Create | `src/hitfinders/gpu.py` | `GPUHitfinder` — GPU hitfinder stub |
| Create | `tests/test_hitfinders.py` | Protocol + factory + mock tests |
| Modify | `src/evaluation/benchmark.py` | Add `aggregation` param to `run_patch_agg` |
| Create | `tests/test_vote_aggregation.py` | Vote aggregation tests |
| Modify | `src/data/dataset.py` | Add `AsymmetricCXIDataset`; deprecate `MultiFrameCXIDataset` |
| Modify | `src/data/dataloader.py` | Add `asymmetric_loader`; deprecate `cxi_session_loader` |
| Create | `tests/test_asymmetric_dataset.py` | Dataset + dataloader tests |
| Create | `configs/supervised/resnet18_asymmetric.yaml` | Asymmetric pipeline config |
| Create | `scripts/train_asymmetric.py` | Main training entry point |

---

## Task 1: Hitfinder Module

**Files:**
- Create: `src/hitfinders/__init__.py`
- Create: `src/hitfinders/base.py`
- Create: `src/hitfinders/pf8.py`
- Create: `src/hitfinders/gpu.py`
- Create: `tests/test_hitfinders.py`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/test_hitfinders.py
from __future__ import annotations

import numpy as np
import pytest

from src.hitfinders.base import MockHitfinder
from src.hitfinders import get_hitfinder


def test_mock_hitfinder_returns_correct_shape():
    hf = MockHitfinder(peaks=np.array([[100.0, 200.0], [300.0, 400.0]]))
    frame = np.zeros((512, 512), dtype=np.float32)
    result = hf.find_peaks(frame)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def test_mock_hitfinder_empty_peaks():
    hf = MockHitfinder(peaks=np.zeros((0, 2), dtype=np.float32))
    result = hf.find_peaks(np.zeros((512, 512), dtype=np.float32))
    assert result.shape == (0, 2)


def test_get_hitfinder_mock():
    cfg = {"hitfinder": {"backend": "mock"}}
    hf = get_hitfinder(cfg)
    assert isinstance(hf, MockHitfinder)


def test_get_hitfinder_unknown_backend_raises():
    cfg = {"hitfinder": {"backend": "unknown"}}
    with pytest.raises(ValueError, match="Unknown hitfinder backend"):
        get_hitfinder(cfg)


def test_pf8_hitfinder_find_peaks_raises_not_implemented():
    from src.hitfinders.pf8 import PF8Hitfinder
    hf = PF8Hitfinder(threshold_snr=5.0, min_peaks=1)
    with pytest.raises(NotImplementedError):
        hf.find_peaks(np.zeros((512, 512), dtype=np.float32))


def test_gpu_hitfinder_find_peaks_raises_not_implemented():
    from src.hitfinders.gpu import GPUHitfinder
    hf = GPUHitfinder()
    with pytest.raises(NotImplementedError):
        hf.find_peaks(np.zeros((512, 512), dtype=np.float32))
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd /data/bioxfel/user/gihan/Hit_finder
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_hitfinders.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'src.hitfinders'`

- [ ] **Step 1.3: Create `src/hitfinders/base.py`**

```python
# src/hitfinders/base.py
"""Hitfinder Protocol and MockHitfinder for testing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Hitfinder(Protocol):
    """Interface for peak-finding algorithms.

    Implementations must be stateless with respect to frame content —
    the same assembled frame must always produce the same centroids
    (deterministic). Implementations may hold configuration state
    (thresholds, device handles) set at __init__ time.
    """

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Locate Bragg peak centroids in an assembled detector frame.

        Args:
            assembled: 2D float32 array (H, W) — assembled detector image.

        Returns:
            float32 array of shape (N_peaks, 2) where each row is [x, y]
            in assembled-frame pixel coordinates (x=column, y=row).
            Returns shape (0, 2) when no peaks are found.
        """
        ...


class MockHitfinder:
    """Deterministic hitfinder that returns a fixed set of centroids.

    Used in unit tests where real peak detection is not needed.
    """

    def __init__(self, peaks: np.ndarray | None = None) -> None:
        if peaks is None:
            peaks = np.zeros((0, 2), dtype=np.float32)
        self._peaks = np.asarray(peaks, dtype=np.float32)

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._peaks.copy()
```

- [ ] **Step 1.4: Create `src/hitfinders/pf8.py`**

```python
# src/hitfinders/pf8.py
"""PeakFinder8 C++ wrapper.

PF8Hitfinder is a stub. Implement find_peaks() once the C++ wrapper
interface is confirmed (subprocess args or ctypes binding).

Worker safety: PF8 C++ subprocess is safe across DataLoader fork workers.
"""

from __future__ import annotations

import numpy as np


class PF8Hitfinder:
    """Wraps the PeakFinder8 C++ binary installed on Sol HPC.

    Args:
        threshold_snr: Signal-to-noise threshold for peak acceptance.
        min_peaks: Minimum number of peaks required to report any peaks.
    """

    def __init__(self, threshold_snr: float = 5.0, min_peaks: int = 1) -> None:
        self.threshold_snr = threshold_snr
        self.min_peaks = min_peaks

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """Call PF8 C++ wrapper and return centroids.

        Not yet implemented — awaiting C++ wrapper interface details.
        Replace this body with the actual subprocess/ctypes call.
        """
        raise NotImplementedError(
            "PF8Hitfinder.find_peaks is not yet implemented. "
            "Provide the C++ PF8 wrapper interface to complete this method."
        )
```

- [ ] **Step 1.5: Create `src/hitfinders/gpu.py`**

```python
# src/hitfinders/gpu.py
"""GPU-accelerated hitfinder stub.

Worker safety: GPU hitfinder CANNOT run inside DataLoader workers with
num_workers > 0 (no shared CUDA context across forked processes).
When using GPUHitfinder, set num_workers=0 in asymmetric_loader().
"""

from __future__ import annotations

import numpy as np


class GPUHitfinder:
    """GPU-accelerated peak finder.

    Implementation to be provided by user. Replace find_peaks() body
    with the actual GPU inference call once the hitfinder script is
    integrated.

    WARNING: num_workers must be 0 when using this backend — GPU context
    is not shareable across forked DataLoader worker processes.
    """

    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "GPUHitfinder.find_peaks is not yet implemented. "
            "Integrate the GPU hitfinder script to complete this method. "
            "Remember to set num_workers=0 in asymmetric_loader()."
        )
```

- [ ] **Step 1.6: Create `src/hitfinders/__init__.py`**

```python
# src/hitfinders/__init__.py
"""Pluggable hitfinder backends for the asymmetric training pipeline."""

from __future__ import annotations

from src.hitfinders.base import Hitfinder, MockHitfinder
from src.hitfinders.gpu import GPUHitfinder
from src.hitfinders.pf8 import PF8Hitfinder

__all__ = ["Hitfinder", "MockHitfinder", "PF8Hitfinder", "GPUHitfinder", "get_hitfinder"]


def get_hitfinder(cfg: dict) -> Hitfinder:
    """Instantiate the hitfinder specified in the config dict.

    Config key: cfg["hitfinder"]["backend"] — one of "pf8", "gpu", "mock".

    Args:
        cfg: Config dict as returned by load_config().

    Returns:
        A Hitfinder instance.

    Raises:
        ValueError: If the backend name is unrecognised.
    """
    hf_cfg = cfg.get("hitfinder", {})
    backend = hf_cfg.get("backend", "pf8")

    if backend == "pf8":
        return PF8Hitfinder(
            threshold_snr=hf_cfg.get("pf8_threshold_snr", 5.0),
            min_peaks=hf_cfg.get("pf8_min_peaks", 1),
        )
    if backend == "gpu":
        return GPUHitfinder()
    if backend == "mock":
        return MockHitfinder()
    raise ValueError(
        f"Unknown hitfinder backend {backend!r}. "
        "Valid options: 'pf8', 'gpu', 'mock'."
    )
```

- [ ] **Step 1.7: Run tests — expect pass**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_hitfinders.py -v
```

Expected: 6 PASSED

- [ ] **Step 1.8: Commit**

```bash
git add src/hitfinders/ tests/test_hitfinders.py
git commit -m "feat: add pluggable Hitfinder module with PF8/GPU stubs and MockHitfinder"
```

---

## Task 2: Vote Aggregation in `run_patch_agg`

**Files:**
- Modify: `src/evaluation/benchmark.py` (lines 169–176, 241–242)
- Create: `tests/test_vote_aggregation.py`

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_vote_aggregation.py
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.benchmark import run_patch_agg


class _FixedScoreModel(torch.nn.Module):
    """Returns fixed softmax scores for every patch."""

    def __init__(self, hit_score: float) -> None:
        super().__init__()
        self._score = hit_score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return [1-score, score] logits that produce desired softmax[:,1]
        # softmax([0, log(score/(1-score))]) ≈ [1-score, score]
        batch = x.shape[0]
        score = self._score
        logit = np.log(score / (1 - score + 1e-9))
        logits = torch.zeros(batch, 2)
        logits[:, 1] = logit
        return logits


def _make_single_frame_session(tmp_path, frame: np.ndarray, label: int):
    """Write one-frame CXI and return session_map, session_ids."""
    import h5py

    cxi = tmp_path / "test.cxi"
    with h5py.File(cxi, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=frame[np.newaxis])
        f.create_dataset("entry_1/labels/hit", data=np.array([label], dtype=np.float32))
    return {"s0": cxi}, ["s0"]


def test_vote_aggregation_three_of_four_patches(tmp_path):
    # Frame 896×896 → 4×4=16 tiles of 224×224
    # Model always scores 0.8 → all tiles are "hit" → score = 16/16 = 1.0
    frame = np.zeros((896, 896), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=1)
    model = _FixedScoreModel(hit_score=0.8)
    result = run_patch_agg(
        model, session_map, session_ids, aggregation="vote", device="cpu"
    )
    assert result["ap"] == pytest.approx(1.0, abs=1e-5)


def test_vote_aggregation_no_hits(tmp_path):
    # Model always scores 0.2 → no tiles hit → score = 0/n = 0.0
    frame = np.zeros((896, 896), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=0)
    model = _FixedScoreModel(hit_score=0.2)
    result = run_patch_agg(
        model, session_map, session_ids, aggregation="vote", device="cpu"
    )
    # Only one frame (label=0), score=0.0 — AP is 0 (no positives in this session)
    assert result["ap"] == pytest.approx(0.0, abs=1e-5)


def test_max_aggregation_backward_compat(tmp_path):
    # Default aggregation="max" still returns max softmax score
    frame = np.zeros((448, 448), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=1)
    model = _FixedScoreModel(hit_score=0.9)
    result = run_patch_agg(
        model, session_map, session_ids, aggregation="max", device="cpu"
    )
    assert result["ap"] == pytest.approx(1.0, abs=1e-5)


def test_run_patch_agg_default_aggregation_is_max(tmp_path):
    # Calling without aggregation= keyword must not raise
    frame = np.zeros((448, 448), dtype=np.float32)
    session_map, session_ids = _make_single_frame_session(tmp_path, frame, label=1)
    model = _FixedScoreModel(hit_score=0.9)
    result = run_patch_agg(model, session_map, session_ids, device="cpu")
    assert "ap" in result
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_vote_aggregation.py -v 2>&1 | head -20
```

Expected: `TypeError: run_patch_agg() got an unexpected keyword argument 'aggregation'`

- [ ] **Step 2.3: Edit `benchmark.py` — add `aggregation` parameter**

In `src/evaluation/benchmark.py`, change the `run_patch_agg` signature from:

```python
def run_patch_agg(
    model: torch.nn.Module,
    session_map: dict[str, Path],
    session_ids: list[str],
    label_key: str = "entry_1/labels/hit",
    patch_size: int = 224,
    patch_stride: int = 224,
    min_hit_patches: int = 3,
    device: str = "cpu",
    inference_batch_size: int = 64,
) -> dict[str, float]:
```

to:

```python
def run_patch_agg(
    model: torch.nn.Module,
    session_map: dict[str, Path],
    session_ids: list[str],
    label_key: str = "entry_1/labels/hit",
    patch_size: int = 224,
    patch_stride: int = 224,
    min_hit_patches: int = 3,
    device: str = "cpu",
    inference_batch_size: int = 64,
    aggregation: str = "max",
) -> dict[str, float]:
```

Update the docstring for `min_hit_patches`:
```
min_hit_patches: Vote mode only — frame is logged as binary hit if
    vote_count >= min_hit_patches. Not used in AP/AUC/F1 computation.
aggregation: Frame-score reduction over patches. "max" (default, backward
    compat): max softmax across patches. "vote": hit_count/n_patches where
    hit_count = number of patches with softmax[:,1] > 0.5.
```

- [ ] **Step 2.4: Edit `benchmark.py` — replace aggregation line 242**

Replace:
```python
            patch_scores = np.concatenate(patch_scores_list)
            all_scores.append(float(patch_scores.max()))
```

With:
```python
            patch_scores = np.concatenate(patch_scores_list)
            n_patches = len(patch_scores)
            if aggregation == "vote":
                hit_count = int((patch_scores > 0.5).sum())
                frame_score = float(hit_count) / max(n_patches, 1)
            else:
                frame_score = float(patch_scores.max())
            all_scores.append(frame_score)
```

- [ ] **Step 2.5: Run tests — expect pass**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_vote_aggregation.py tests/test_evaluation.py -v
```

Expected: all PASSED (existing `test_evaluation.py` tests still pass because default is `"max"`)

- [ ] **Step 2.6: Commit**

```bash
git add src/evaluation/benchmark.py tests/test_vote_aggregation.py
git commit -m "feat: add vote aggregation to run_patch_agg (aggregation='vote'|'max')"
```

---

## Task 3: `AsymmetricCXIDataset`

**Files:**
- Modify: `src/data/dataset.py`
- Create: `tests/test_asymmetric_dataset.py`

- [ ] **Step 3.1: Write failing tests**

```python
# tests/test_asymmetric_dataset.py
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.hitfinders.base import MockHitfinder
from src.data.dataset import AsymmetricCXIDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cxi(tmp_path: Path, n_frames: int = 8, h: int = 512, w: int = 512) -> Path:
    """Synthetic CXI: alternating hit/non-hit frames."""
    cxi = tmp_path / "test.cxi"
    rng = np.random.default_rng(0)
    data = rng.random((n_frames, h, w)).astype(np.float32)
    labels = np.array([i % 2 for i in range(n_frames)], dtype=np.float32)
    with h5py.File(cxi, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=data)
        f.create_dataset("entry_1/labels/hit", data=labels)
    return cxi


@pytest.fixture()
def cxi_path(tmp_path):
    return _make_cxi(tmp_path)


@pytest.fixture()
def hit_hitfinder():
    """Returns two known peak centroids, always."""
    return MockHitfinder(peaks=np.array([[100.0, 100.0], [300.0, 300.0]], dtype=np.float32))


@pytest.fixture()
def empty_hitfinder():
    """Returns no peaks."""
    return MockHitfinder(peaks=np.zeros((0, 2), dtype=np.float32))


# ── Shape / dtype ─────────────────────────────────────────────────────────────

def test_getitem_returns_correct_shape(cxi_path, hit_hitfinder):
    ds = AsymmetricCXIDataset([cxi_path], hitfinder=hit_hitfinder)
    tensor, label = ds[0]
    assert tensor.shape == (1, 224, 224)
    assert tensor.dtype == torch.float32


def test_getitem_label_is_int(cxi_path, hit_hitfinder):
    ds = AsymmetricCXIDataset([cxi_path], hitfinder=hit_hitfinder)
    _, label = ds[0]
    assert isinstance(label, int)


def test_dataset_len(cxi_path, hit_hitfinder):
    ds = AsymmetricCXIDataset([cxi_path], hitfinder=hit_hitfinder)
    assert len(ds) == 8


# ── Crop-content label logic ──────────────────────────────────────────────────

def test_hit_frame_with_peaks_can_produce_label1(cxi_path, hit_hitfinder):
    """A hit frame with peaks should eventually produce a label-1 sample."""
    ds = AsymmetricCXIDataset(
        [cxi_path], hitfinder=hit_hitfinder, hard_neg_fraction=0.0, rng_seed=42
    )
    # Frame 1 is a hit (label=1.0). With hard_neg_fraction=0, always tries hit crop.
    found_hit = False
    for _ in range(20):
        _, lbl = ds[1]  # index 1 = hit frame
        if lbl == 1:
            found_hit = True
            break
    assert found_hit, "Expected at least one label-1 sample from a hit frame with peaks"


def test_non_hit_frame_always_produces_label0(cxi_path, hit_hitfinder):
    ds = AsymmetricCXIDataset([cxi_path], hitfinder=hit_hitfinder)
    # Frame 0 is non-hit. Must always be label 0 regardless of hitfinder output.
    for _ in range(10):
        _, lbl = ds[0]
        assert lbl == 0


def test_hard_negative_is_label0(cxi_path, hit_hitfinder):
    """When hard_neg_fraction=1.0, hit frames always produce hard-neg (label 0)."""
    ds = AsymmetricCXIDataset(
        [cxi_path], hitfinder=hit_hitfinder, hard_neg_fraction=1.0
    )
    for _ in range(10):
        _, lbl = ds[1]  # hit frame
        assert lbl == 0


# ── Crop-contains-centroid helper ─────────────────────────────────────────────

def test_crop_contains_centroid_true():
    from src.data.dataset import _crop_contains_centroid
    centroids = np.array([[112.0, 112.0]], dtype=np.float32)  # centre of 224×224 crop at (0,0)
    assert _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


def test_crop_contains_centroid_false():
    from src.data.dataset import _crop_contains_centroid
    centroids = np.array([[500.0, 500.0]], dtype=np.float32)
    assert not _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


def test_crop_contains_centroid_on_left_edge():
    from src.data.dataset import _crop_contains_centroid
    centroids = np.array([[0.0, 100.0]], dtype=np.float32)  # x=0 is left edge of crop
    assert _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


def test_crop_contains_centroid_on_right_boundary_excluded():
    from src.data.dataset import _crop_contains_centroid
    centroids = np.array([[224.0, 0.0]], dtype=np.float32)  # x=224 is outside [0, 224)
    assert not _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


def test_crop_contains_centroid_empty_centroids():
    from src.data.dataset import _crop_contains_centroid
    centroids = np.zeros((0, 2), dtype=np.float32)
    assert not _crop_contains_centroid(top=0, left=0, size=224, centroids=centroids)


# ── Fallbacks ─────────────────────────────────────────────────────────────────

def test_hit_frame_with_no_peaks_fallback_to_label0(cxi_path, empty_hitfinder):
    """Hit frame where hitfinder finds 0 peaks → falls back to random crop, label=0."""
    ds = AsymmetricCXIDataset(
        [cxi_path], hitfinder=empty_hitfinder, hard_neg_fraction=0.0
    )
    tensor, lbl = ds[1]  # hit frame, but no peaks found
    assert tensor.shape == (1, 224, 224)
    assert lbl == 0


# ── DataLoader multiprocess ───────────────────────────────────────────────────

def test_dataloader_two_workers(cxi_path, hit_hitfinder):
    ds = AsymmetricCXIDataset([cxi_path], hitfinder=hit_hitfinder)
    loader = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    images, labels = batches[0]
    assert images.shape == (4, 1, 224, 224)


# ── No open file handles ──────────────────────────────────────────────────────

def test_no_open_file_after_getitem(cxi_path, hit_hitfinder):
    import gc
    ds = AsymmetricCXIDataset([cxi_path], hitfinder=hit_hitfinder)
    ds[0]
    gc.collect()
    # If HDF5 files were left open this would error on repeated access
    ds[0]  # should not raise
```

- [ ] **Step 3.2: Run failing tests**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_asymmetric_dataset.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'AsymmetricCXIDataset' from 'src.data.dataset'`

- [ ] **Step 3.3: Add imports to `src/data/dataset.py`**

After the existing imports block (after line 28), add:

```python
from src.preprocessing.augment import (
    random_crop,
    random_cutout,
    random_flip,
    random_rot90,
)
from src.preprocessing.normalize import LCN_WINDOW_DEFAULT, gcn, lcn
```

- [ ] **Step 3.4: Add `_crop_contains_centroid` to `src/data/dataset.py`**

Add as a module-level function after the imports:

```python
def _crop_contains_centroid(
    top: int,
    left: int,
    size: int,
    centroids: np.ndarray,
) -> bool:
    """Return True if any centroid (x, y) falls inside the crop region.

    Crop region: columns [left, left+size), rows [top, top+size).
    centroids: float32 array of shape (N, 2) where col 0 is x, col 1 is y.
    """
    if len(centroids) == 0:
        return False
    in_col = (centroids[:, 0] >= left) & (centroids[:, 0] < left + size)
    in_row = (centroids[:, 1] >= top) & (centroids[:, 1] < top + size)
    return bool((in_col & in_row).any())
```

- [ ] **Step 3.5: Add deprecation comment to `MultiFrameCXIDataset`**

Replace the first line of the `MultiFrameCXIDataset` docstring:

```python
class MultiFrameCXIDataset(Dataset):
    """DEPRECATED: use AsymmetricCXIDataset for new training runs.

    Dataset over multi-frame CXI files with embedded hit labels.
    ...
    """
```

- [ ] **Step 3.6: Add `AsymmetricCXIDataset` to `src/data/dataset.py`**

Append after `MultiFrameCXIDataset`:

```python
class AsymmetricCXIDataset(Dataset):
    """Asymmetric training dataset: hitfinder-guided crops for training.

    For each frame per epoch, calls a hitfinder to locate Bragg peak
    centroids, then samples a 224×224 crop whose label is determined by
    crop content — not the frame-level label:

    - Frame label=1 (hit), coin < (1 - hard_neg_fraction):
        Sample random crops until one contains ≥1 centroid → label=1.
        If max_attempts exhausted with no hit crop, fallback → label=0.
    - Frame label=1 (hit), coin ≥ (1 - hard_neg_fraction):
        Hard negative: sample crop with 0 centroids → label=0.
    - Frame label=0 (non-hit):
        Random crop → label=0 (no centroids present).

    Augmentation (applied to every crop regardless of label):
        random_rot90 → random_flip → gcn(patch) → lcn(patch) → random_cutout

    HDF5 files are opened lazily in __getitem__ (multiprocessing safe).

    Args:
        cxi_paths: CXI/HDF5 files to include.
        hitfinder: Hitfinder instance (PF8Hitfinder, GPUHitfinder, MockHitfinder).
        label_key: HDF5 key for per-frame labels.
        patch_size: Side length of output patches (default 224).
        lcn_window: Window size for local contrast normalization.
        hard_neg_fraction: Fraction of hit-frame items used as hard negatives
            (crops with no centroids, label=0). Range [0, 1].
        hard_neg_max_attempts: Rejection-sampling attempts before fallback.
        n_cutout_holes: Number of cutout regions per patch.
        cutout_hole_size: Side length of each cutout region in pixels.
        rng_seed: Seed for the per-worker numpy Generator.
    """

    def __init__(
        self,
        cxi_paths: list[str | Path],
        hitfinder: object,
        label_key: str = "entry_1/labels/hit",
        patch_size: int = 224,
        lcn_window: int = LCN_WINDOW_DEFAULT,
        hard_neg_fraction: float = 0.5,
        hard_neg_max_attempts: int = 50,
        n_cutout_holes: int = 3,
        cutout_hole_size: int = 32,
        rng_seed: int = 42,
    ) -> None:
        self._hitfinder = hitfinder
        self._patch_size = patch_size
        self._lcn_window = lcn_window
        self._hard_neg_fraction = hard_neg_fraction
        self._hard_neg_max_attempts = hard_neg_max_attempts
        self._n_cutout_holes = n_cutout_holes
        self._cutout_hole_size = cutout_hole_size
        self._rng_seed = rng_seed

        # Read detector descriptions eagerly (same pattern as MultiFrameCXIDataset).
        unique_paths = {Path(p) for p in cxi_paths}
        self._path_to_desc: dict[Path, str] = {}
        for p in unique_paths:
            try:
                self._path_to_desc[p] = read_detector_description(p)
            except (ValueError, KeyError, OSError):
                pass

        # Build flat (path, frame_idx) index and cache labels eagerly.
        self._index: list[tuple[Path, int]] = []
        self._labels: list[int] = []
        for p in cxi_paths:
            p = Path(p)
            arr = read_embedded_labels(p, label_key)
            for i, raw in enumerate(arr):
                self._index.append((p, i))
                self._labels.append(int(round(float(raw))))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, frame_idx = self._index[idx]
        frame_label = self._labels[idx]

        # Fresh per-call RNG (reproducible under workers via idx + seed).
        rng = np.random.default_rng(self._rng_seed + idx)

        # --- Assemble frame ---
        raw = read_frame(path, frame_idx)
        if path in self._path_to_desc:
            desc = self._path_to_desc[path]
            try:
                pads = get_geometry(desc)
                assembler = get_assembler(desc)
                assembled = assemble_only(raw, pads, desc, assembler=assembler)
            except (ValueError, KeyError, OSError):
                assembled = _to_2d(raw)
        else:
            assembled = _to_2d(raw)
        assembled = assembled.astype(np.float32)

        # --- Hitfinder ---
        centroids = self._hitfinder.find_peaks(assembled)  # (N, 2) float32

        h, w = assembled.shape
        size = self._patch_size

        # --- Sample crop + assign label ---
        patch, patch_label = self._sample_patch(
            assembled, centroids, frame_label, h, w, size, rng
        )

        # --- Augment: rotate → flip → GCN → LCN → cutout ---
        patch = random_rot90(patch, rng)
        patch = random_flip(patch, rng)
        patch = gcn(patch.copy())         # .copy() — rot90 returns non-contiguous view
        patch = lcn(patch, window=self._lcn_window)
        patch = random_cutout(
            patch.astype(np.float32), rng,
            n_holes=self._n_cutout_holes,
            hole_size=self._cutout_hole_size,
        )

        tensor = torch.from_numpy(np.ascontiguousarray(patch).astype(np.float32)).unsqueeze(0)
        return tensor, patch_label

    def _sample_patch(
        self,
        assembled: np.ndarray,
        centroids: np.ndarray,
        frame_label: int,
        h: int,
        w: int,
        size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Sample a crop and return (patch, patch_label)."""
        if frame_label == 0:
            # Non-hit frame: any random crop is label 0.
            return random_crop(assembled, size, rng), 0

        # Hit frame: decide strategy.
        coin = float(rng.random())
        want_hard_neg = coin < self._hard_neg_fraction

        if len(centroids) == 0:
            # No peaks found — fall back to random crop, label 0.
            return random_crop(assembled, size, rng), 0

        if want_hard_neg:
            return self._sample_background_crop(assembled, centroids, h, w, size, rng)
        else:
            return self._sample_hit_crop(assembled, centroids, h, w, size, rng)

    def _sample_hit_crop(
        self,
        assembled: np.ndarray,
        centroids: np.ndarray,
        h: int,
        w: int,
        size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Sample a crop containing ≥1 centroid (label 1). Falls back to label 0."""
        for _ in range(self._hard_neg_max_attempts):
            top = int(rng.integers(0, max(1, h - size + 1)))
            left = int(rng.integers(0, max(1, w - size + 1)))
            if _crop_contains_centroid(top, left, size, centroids):
                return assembled[top : top + size, left : left + size], 1
        # Exhausted attempts — return random crop, label 0.
        return random_crop(assembled, size, rng), 0

    def _sample_background_crop(
        self,
        assembled: np.ndarray,
        centroids: np.ndarray,
        h: int,
        w: int,
        size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Sample a crop containing 0 centroids (hard negative, label 0)."""
        for _ in range(self._hard_neg_max_attempts):
            top = int(rng.integers(0, max(1, h - size + 1)))
            left = int(rng.integers(0, max(1, w - size + 1)))
            if not _crop_contains_centroid(top, left, size, centroids):
                return assembled[top : top + size, left : left + size], 0
        # Exhausted — just return any random crop as label 0.
        return random_crop(assembled, size, rng), 0
```

- [ ] **Step 3.7: Run tests — expect pass**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_asymmetric_dataset.py -v
```

Expected: all PASSED

- [ ] **Step 3.8: Run full test suite to confirm nothing broken**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 3.9: Commit**

```bash
git add src/data/dataset.py tests/test_asymmetric_dataset.py
git commit -m "feat: add AsymmetricCXIDataset with hitfinder-guided patch labeling"
```

---

## Task 4: `asymmetric_loader` + Deprecate `cxi_session_loader`

**Files:**
- Modify: `src/data/dataloader.py`
- Test: covered by `tests/test_asymmetric_dataset.py` (DataLoader test already passes)

- [ ] **Step 4.1: Edit `src/data/dataloader.py` — add import**

Add to top-of-file imports:

```python
from src.data.dataset import AsymmetricCXIDataset, MultiFrameCXIDataset, SFXDataset, UnlabeledDataset
```

(Replace the existing import of `MultiFrameCXIDataset, SFXDataset, UnlabeledDataset`.)

- [ ] **Step 4.2: Deprecate `cxi_session_loader`**

Add one line to `cxi_session_loader` docstring (first line of the docstring):

```python
def cxi_session_loader(...) -> DataLoader:
    """DEPRECATED: use asymmetric_loader for new training runs.

    DataLoader over a subset of CXI sessions identified by session_id.
    ...
    """
```

- [ ] **Step 4.3: Add `asymmetric_loader` to `src/data/dataloader.py`**

Append after `cxi_session_loader`:

```python
def asymmetric_loader(
    session_map: dict[str, Path],
    session_ids: list[str],
    hitfinder: object,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    label_key: str = "entry_1/labels/hit",
    patch_size: int = 224,
    lcn_window: int = 9,
    hard_neg_fraction: float = 0.5,
    hard_neg_max_attempts: int = 50,
    n_cutout_holes: int = 3,
    cutout_hole_size: int = 32,
    rng_seed: int = 42,
) -> DataLoader:
    """DataLoader for the asymmetric pipeline.

    Each item is a hitfinder-guided 224×224 patch with a crop-content label.
    See AsymmetricCXIDataset for label assignment logic.

    Args:
        session_map: Mapping from session_id to CXI file path.
        session_ids: Subset of session_ids to include.
        hitfinder: Hitfinder instance (PF8Hitfinder / GPUHitfinder / MockHitfinder).
            GPU backend requires num_workers=0.
        batch_size: Patches per batch.
        num_workers: DataLoader worker processes (set 0 for GPU hitfinder).
        shuffle: Shuffle each epoch.
        label_key: HDF5 key for per-frame labels.
        patch_size: Crop side length in pixels.
        lcn_window: LCN neighbourhood window size.
        hard_neg_fraction: Fraction of hit-frame samples used as hard negatives.
        hard_neg_max_attempts: Rejection-sampling cap before fallback.
        n_cutout_holes: Number of cutout regions per patch.
        cutout_hole_size: Cutout region side length in pixels.
        rng_seed: Base seed for per-item RNGs.

    Returns:
        DataLoader yielding (image, label) pairs; image shape (B, 1, 224, 224).
    """
    paths = [session_map[sid] for sid in session_ids]
    dataset = AsymmetricCXIDataset(
        cxi_paths=paths,
        hitfinder=hitfinder,
        label_key=label_key,
        patch_size=patch_size,
        lcn_window=lcn_window,
        hard_neg_fraction=hard_neg_fraction,
        hard_neg_max_attempts=hard_neg_max_attempts,
        n_cutout_holes=n_cutout_holes,
        cutout_hole_size=cutout_hole_size,
        rng_seed=rng_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
```

- [ ] **Step 4.4: Run full tests**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 4.5: Commit**

```bash
git add src/data/dataloader.py
git commit -m "feat: add asymmetric_loader; deprecate cxi_session_loader"
```

---

## Task 5: Config File

**Files:**
- Create: `configs/supervised/resnet18_asymmetric.yaml`

- [ ] **Step 5.1: Create the config**

```yaml
# configs/supervised/resnet18_asymmetric.yaml
# Asymmetric pipeline config — inherits base.yaml via load_config() deep-merge.

model:
  backbone: resnet18
  pretrained: true
  num_classes: 2

training:
  epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  batch_size: 32
  num_workers: 4
  early_stopping_patience: 10

data:
  label_key: entry_1/labels/hit

# Fill in actual paths on Sol before running.
lodo:
  detector_dirs:
    AGIPD: /data/bioxfel/agipd
    JUNGFRAU_4M: /data/bioxfel/jungfrau4m
    ePix10k: /data/bioxfel/epix10k
    Eiger4M: /data/bioxfel/eiger4m
  cxi_pattern: "compressed*.cxi"
  label_key: entry_1/labels/hit

hitfinder:
  backend: pf8          # "pf8" | "gpu" | "mock"
  pf8_threshold_snr: 5.0
  pf8_min_peaks: 1

asymmetric:
  hard_neg_fraction: 0.5
  hard_neg_max_attempts: 50
  n_cutout_holes: 3
  cutout_hole_size: 32
  rng_seed: 42

benchmark:
  aggregation: vote     # "vote" uses hit_count/n_tiles as frame score
  min_hit_patches: 3    # binary frame decision threshold (informational only)
  patch_stride: 224

wandb:
  project: sfx-hitfinder
  tags:
    - supervised
    - resnet18
    - asymmetric-pipeline
```

- [ ] **Step 5.2: Verify load_config parses it without error**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -c "
from src.utils.config import load_config
cfg = load_config('configs/supervised/resnet18_asymmetric.yaml')
print('model:', cfg['model'])
print('hitfinder:', cfg['hitfinder'])
print('benchmark:', cfg['benchmark'])
"
```

Expected: prints all three dicts without error.

- [ ] **Step 5.3: Commit**

```bash
git add configs/supervised/resnet18_asymmetric.yaml
git commit -m "feat: add resnet18_asymmetric.yaml config"
```

---

## Task 6: `scripts/train_asymmetric.py`

**Files:**
- Create: `scripts/train_asymmetric.py`

- [ ] **Step 6.1: Create the training script**

```python
# scripts/train_asymmetric.py
"""Asymmetric pipeline training entry point.

Trains ResNet18 using hitfinder-guided patch crops for supervision
and blind-grid sliding-window aggregation for validation.

Usage:
    /home/gketawal/.conda/envs/sfx-hitfinder/bin/python \\
        scripts/train_asymmetric.py \\
        --config configs/supervised/resnet18_asymmetric.yaml

    # Single LODO fold (smoke test):
    scripts/train_asymmetric.py --config ... --folds 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import asymmetric_loader
from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
    format_results_table,
    run_patch_agg,
    save_split_artifact,
)
from src.hitfinders import get_hitfinder
from src.models.supervised import build_supervised_model
from src.training.train_supervised import _set_seeds, train_one_epoch
from src.utils.config import load_config


def build_sessions(lodo_cfg: dict) -> tuple[list[dict], dict[str, Path]]:
    """Discover CXI files and build session records (mirrors train_lodo.py)."""
    sessions: list[dict] = []
    session_map: dict[str, Path] = {}
    pattern = lodo_cfg.get("cxi_pattern", "compressed*.cxi")
    label_key = lodo_cfg.get("label_key", "entry_1/labels/hit")

    for detector, dir_str in lodo_cfg["detector_dirs"].items():
        det_dir = Path(dir_str)
        for cxi in sorted(det_dir.glob(pattern)):
            with h5py.File(cxi, "r") as f:
                n_frames = int(f[label_key].shape[0])
            sid = f"{detector}_{cxi.stem}"
            sessions.append(
                {"session_id": sid, "detector": detector, "frame_count": n_frames}
            )
            session_map[sid] = cxi

    return sessions, session_map


def _make_train_loader(
    split_artifact: dict,
    session_map: dict[str, Path],
    hitfinder: object,
    cfg: dict,
):
    ids = [sid for sid, s in split_artifact["splits"].items() if s == SPLIT_TRAIN]
    asym = cfg.get("asymmetric", {})
    return asymmetric_loader(
        session_map=session_map,
        session_ids=ids,
        hitfinder=hitfinder,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
        label_key=cfg["data"]["label_key"],
        hard_neg_fraction=asym.get("hard_neg_fraction", 0.5),
        hard_neg_max_attempts=asym.get("hard_neg_max_attempts", 50),
        n_cutout_holes=asym.get("n_cutout_holes", 3),
        cutout_hole_size=asym.get("cutout_hole_size", 32),
        rng_seed=asym.get("rng_seed", cfg.get("seed", 42)),
    )


def _train_fold(
    fold: dict,
    split_artifact: dict,
    session_map: dict[str, Path],
    hitfinder: object,
    cfg: dict,
    device: str,
) -> dict:
    import wandb

    backbone = cfg["model"]["backbone"]
    seed = cfg["seed"]
    fold_id = fold["fold_id"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stopping_patience", 10)
    run_name = f"{backbone}-asymmetric-fold{fold_id}-seed{seed}"

    label_key = cfg["data"]["label_key"]
    bench_cfg = cfg.get("benchmark", {})
    aggregation = bench_cfg.get("aggregation", "vote")
    patch_stride = bench_cfg.get("patch_stride", 224)
    min_hit_patches = bench_cfg.get("min_hit_patches", 3)

    val_ids = [sid for sid, s in split_artifact["splits"].items() if s == SPLIT_VAL]
    in_domain_ids = [
        sid for sid, s in split_artifact["splits"].items() if s == SPLIT_IN_DOMAIN_TEST
    ]
    cross_ids = [
        sid for sid, s in split_artifact["splits"].items() if s == SPLIT_CROSS_DETECTOR
    ]

    print(
        f"\n{'='*60}\n"
        f"Fold {fold_id}  |  held-out: {fold['test_detector']}\n"
        f"  val={len(val_ids)}  in_domain={len(in_domain_ids)}  cross={len(cross_ids)}\n"
        f"{'='*60}"
    )

    _set_seeds(seed)
    model = build_supervised_model(
        backbone=backbone,
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    resume_eval_only = ckpt_path.exists()

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        id=run_name,
        name=run_name,
        config={**cfg, "fold_id": fold_id, "test_detector": fold["test_detector"]},
        tags=cfg["wandb"].get("tags", []),
        resume="allow",
    )

    if not resume_eval_only:
        train_dl = _make_train_loader(split_artifact, session_map, hitfinder, cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"].get("weight_decay", 1e-4),
        )
        criterion = nn.CrossEntropyLoss()
        best_f1 = -1.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_m = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_m = run_patch_agg(
                model,
                session_map,
                val_ids,
                label_key=label_key,
                patch_stride=patch_stride,
                min_hit_patches=min_hit_patches,
                device=device,
                aggregation=aggregation,
            )
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val_AP={val_m['ap']:.4f}  val_F1={val_m['f1']:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_m["loss"],
                    "val/ap": val_m["ap"],
                    "val/auc": val_m["auc_roc"],
                    "val/f1": val_m["f1"],
                }
            )

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_f1": best_f1,
                        "backbone": backbone,
                        "num_classes": cfg["model"]["num_classes"],
                    },
                    ckpt_path,
                )
                print(f"    → checkpoint saved (val F1={best_f1:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
    else:
        print(f"  Checkpoint found — skipping training, running eval only.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    in_domain_m = run_patch_agg(
        model, session_map, in_domain_ids,
        label_key=label_key, patch_stride=patch_stride,
        min_hit_patches=min_hit_patches, device=device, aggregation=aggregation,
    )
    cross_m = run_patch_agg(
        model, session_map, cross_ids,
        label_key=label_key, patch_stride=patch_stride,
        min_hit_patches=min_hit_patches, device=device, aggregation=aggregation,
    )

    print(
        f"  In-domain test: AP={in_domain_m['ap']:.4f}  F1={in_domain_m['f1']:.4f}\n"
        f"  Cross-detector: AP={cross_m['ap']:.4f}  F1={cross_m['f1']:.4f}"
    )

    wandb.log(
        {
            "in_domain/ap": in_domain_m["ap"],
            "in_domain/auc": in_domain_m["auc_roc"],
            "in_domain/f1": in_domain_m["f1"],
            "cross/ap": cross_m["ap"],
            "cross/auc": cross_m["auc_roc"],
            "cross/f1": cross_m["f1"],
        }
    )
    wandb.finish()

    result = {
        "fold_id": fold_id,
        "test_detector": fold["test_detector"],
        "cross": {
            "ap": cross_m["ap"],
            "auc_roc": cross_m["auc_roc"],
            "f1": cross_m["f1"],
            "threshold": cross_m["threshold"],
        },
        "in_domain": {
            "ap": in_domain_m["ap"],
            "auc_roc": in_domain_m["auc_roc"],
            "f1": in_domain_m["f1"],
            "threshold": in_domain_m["threshold"],
        },
    }
    results_path = ckpt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results → {results_path}")

    return {
        "test_detector": fold["test_detector"],
        "ap": cross_m["ap"],
        "in_domain_ap": in_domain_m["ap"],
        "auc_roc": cross_m["auc_roc"],
        "f1": cross_m["f1"],
    }


def main(config_path: str | Path, folds: list[int] | None = None) -> None:
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Config: {config_path}")

    hitfinder = get_hitfinder(cfg)
    print(f"Hitfinder: {type(hitfinder).__name__}")

    sessions, session_map = build_sessions(cfg["lodo"])
    print(f"Sessions: {len(sessions)}")

    all_folds = build_lodo_folds()
    if folds is not None:
        all_folds = [f for f in all_folds if f["fold_id"] in folds]

    known_detectors = {s["detector"] for s in sessions}
    for fold in all_folds:
        if fold["test_detector"] not in known_detectors:
            raise ValueError(
                f"Fold {fold['fold_id']} test_detector={fold['test_detector']!r} "
                f"not in sessions (have: {sorted(known_detectors)})."
            )

    artifacts_dir = Path("checkpoints") / "asymmetric_splits"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fold_results: dict[str, dict] = {}
    for fold in all_folds:
        split_artifact = build_session_stratified_split(
            sessions,
            test_detector=fold["test_detector"],
            fold=fold["fold_id"],
            seed=cfg["seed"],
        )
        save_split_artifact(
            split_artifact, artifacts_dir / f"fold_{fold['fold_id']}.json"
        )
        result = _train_fold(fold, split_artifact, session_map, hitfinder, cfg, device)
        fold_results[f"fold_{fold['fold_id']}"] = result

    results_for_table: dict = {}
    ap_values = []
    for key, r in fold_results.items():
        results_for_table[key] = {"ap": r["ap"], "test_detector": r["test_detector"]}
        ap_values.append(r["ap"])

    if len(ap_values) > 1:
        results_for_table["mean_ap"] = float(np.mean(ap_values))
        results_for_table["std_ap"] = float(np.std(ap_values, ddof=1))
    elif ap_values:
        results_for_table["mean_ap"] = ap_values[0]
        results_for_table["std_ap"] = float("nan")

    print("\n" + format_results_table(results_for_table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asymmetric pipeline training")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Fold IDs to run (1–4). Omit to run all four.",
    )
    args = parser.parse_args()
    main(args.config, folds=args.folds)
```

- [ ] **Step 6.2: Verify script imports cleanly**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -c "
import sys; sys.path.insert(0, '.')
import scripts.train_asymmetric as ta
print('imports OK')
"
```

Expected: `imports OK`

- [ ] **Step 6.3: Run full test suite**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 6.4: Commit**

```bash
git add scripts/train_asymmetric.py
git commit -m "feat: add train_asymmetric.py — asymmetric pipeline training entry point"
```

---

## Task 7: Final Verification

- [ ] **Step 7.1: Run complete test suite**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all PASSED, 0 FAILED

- [ ] **Step 7.2: Verify deprecated imports still work**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -c "
from src.data.dataset import MultiFrameCXIDataset
from src.data.dataloader import cxi_session_loader
print('deprecated imports OK')
"
```

Expected: `deprecated imports OK`

- [ ] **Step 7.3: Smoke-test script dry run (mock hitfinder)**

Edit `configs/supervised/resnet18_asymmetric.yaml` temporarily: set `hitfinder.backend: mock`. Then check it parses:

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -c "
from src.utils.config import load_config
from src.hitfinders import get_hitfinder
cfg = load_config('configs/supervised/resnet18_asymmetric.yaml')
hf = get_hitfinder(cfg)
print('hitfinder:', type(hf).__name__)
"
```

Expected: `hitfinder: MockHitfinder`

- [ ] **Step 7.4: Final commit**

```bash
git add -p  # stage any remaining changes
git commit -m "chore: asymmetric pipeline — final integration and verification"
```

---

## Notes for PF8 Integration (pending)

Once the C++ PF8 wrapper interface is confirmed, implement `PF8Hitfinder.find_peaks` in `src/hitfinders/pf8.py`. The method must:

1. Accept `assembled: np.ndarray` (2D float32, shape H×W)
2. Return `np.ndarray` of shape `(N_peaks, 2)` float32, columns `[x, y]`  
3. Return `np.zeros((0, 2), dtype=np.float32)` when no peaks found
4. Be safe to call from multiple DataLoader worker processes simultaneously

Update `tests/test_hitfinders.py` to add a real integration test once the binary is accessible.
