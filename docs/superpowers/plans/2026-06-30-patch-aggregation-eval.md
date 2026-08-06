> **Saved to:** `docs/superpowers/plans/2026-06-30-patch-aggregation-eval.md`
> **Spec saved to:** `docs/superpowers/specs/2026-06-30-patch-aggregation-eval.md`

# Patch-Aggregation Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every non-training evaluation step (validation loop, in-domain test, cross-detector test) with a patch-grid aggregation strategy that tiles the full assembled frame, runs all patches through ResNet18, and reduces to a single frame-level score.

**Architecture:** Add `patch_grid` to `augment.py` and `preprocess_eval_patches` to `pipeline.py` as pure numpy helpers. Add `run_patch_agg` to `benchmark.py` as a standalone evaluation function that assembles frames directly from CXI files (bypasses the DataLoader). In `train_lodo.py`, eliminate the `val_dl` / `in_domain_dl` / `cross_dl` DataLoaders for metric purposes — replace all three `run_on_loader` / `evaluate` calls with `run_patch_agg`. The gradient-update loop (`train_one_epoch`) is unchanged: it still uses the augmented `train_dl`. Early stopping now monitors val F1 from `run_patch_agg`.

**Tech Stack:** numpy, PyTorch (CPU/GPU mini-batch inference), existing Reborn geometry stack (`assemble_only`, `get_geometry`, `get_assembler`), `preprocess_eval_patches` → GCN → LCN per patch.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/preprocessing/augment.py` | Add `patch_grid` |
| Modify | `src/preprocessing/pipeline.py` | Add `preprocess_eval_patches` |
| Modify | `src/evaluation/benchmark.py` | Add `run_patch_agg`, update `__all__` |
| Modify | `scripts/train_lodo.py` | Replace all eval calls (val + final) with `run_patch_agg`; remove DataLoaders that existed only for eval |
| Modify | `configs/supervised/resnet18_lodo.yaml` | Add `evaluation` section |
| Create | `tests/test_patch_eval.py` | Tests for all new functions |

---

## Task 1: `patch_grid` — tile a 2D image into a list of patches

**Files:**
- Modify: `src/preprocessing/augment.py`
- Test: `tests/test_patch_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_patch_eval.py`:

```python
"""Tests for patch-grid tiling and patch-aggregation evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.augment import patch_grid


class TestPatchGrid:
    def test_non_overlapping_count_exact_fit(self):
        # 448 = 2 × 224, so 2×2 = 4 patches exactly
        img = np.zeros((448, 448), dtype=np.float32)
        patches = patch_grid(img, patch_size=224, stride=224)
        assert len(patches) == 4

    def test_non_overlapping_count_partial_edge(self):
        # floor(1273 / 224) = 5 → 5×5 = 25 patches; 153 px edge discarded
        img = np.zeros((1273, 1273), dtype=np.float32)
        patches = patch_grid(img, patch_size=224, stride=224)
        assert len(patches) == 25

    def test_default_stride_equals_patch_size(self):
        img = np.zeros((1273, 1273), dtype=np.float32)
        assert len(patch_grid(img, 224)) == len(patch_grid(img, 224, 224))

    def test_each_patch_is_correct_size(self):
        img = np.zeros((1273, 1273), dtype=np.float32)
        for p in patch_grid(img, 224, 224):
            assert p.shape == (224, 224)

    def test_patches_contain_correct_pixel_values(self):
        img = np.arange(448 * 448, dtype=np.float32).reshape(448, 448)
        patches = patch_grid(img, 224, 224)
        np.testing.assert_array_equal(patches[0], img[0:224, 0:224])
        np.testing.assert_array_equal(patches[1], img[0:224, 224:448])

    def test_overlapping_stride_increases_count(self):
        img = np.zeros((500, 500), dtype=np.float32)
        assert len(patch_grid(img, 224, 100)) > len(patch_grid(img, 224, 224))

    def test_image_smaller_than_patch_returns_empty(self):
        img = np.zeros((100, 100), dtype=np.float32)
        assert patch_grid(img, 224, 224) == []

    def test_row_major_order(self):
        img = np.zeros((448, 448), dtype=np.float32)
        img[0, 0] = 1.0    # top-left → patch index 0
        img[224, 0] = 2.0  # bottom-left → patch index 2
        patches = patch_grid(img, 224, 224)
        assert patches[0][0, 0] == 1.0
        assert patches[2][0, 0] == 2.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_patch_eval.py::TestPatchGrid -v
```

Expected: `ImportError` — `patch_grid` not yet defined.

- [ ] **Step 3: Implement `patch_grid` in `augment.py`**

Append to `src/preprocessing/augment.py`:

```python
def patch_grid(
    image: np.ndarray,
    patch_size: int = 224,
    stride: int | None = None,
) -> list[np.ndarray]:
    """Tile a 2D image into a grid of (patch_size × patch_size) patches.

    Partial patches at the right/bottom edge are discarded — no padding.
    Patches are returned in row-major order (left-to-right, top-to-bottom).

    Args:
        image: 2D float32 array (H, W).
        patch_size: Side length of each square patch in pixels.
        stride: Step between patch origins. Defaults to patch_size
            (non-overlapping). A stride < patch_size produces overlapping
            patches for better edge coverage at the cost of more forward passes.

    Returns:
        List of float32 arrays each of shape (patch_size, patch_size).
        Empty list if the image is smaller than patch_size in either dim.
    """
    if stride is None:
        stride = patch_size
    h, w = image.shape
    patches = []
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patches.append(image[top : top + patch_size, left : left + patch_size])
    return patches
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_patch_eval.py::TestPatchGrid -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/augment.py tests/test_patch_eval.py
git commit -m "feat: add patch_grid for sliding-window image tiling"
```

---

## Task 2: `preprocess_eval_patches` — GCN → LCN per patch

**Files:**
- Modify: `src/preprocessing/pipeline.py`
- Test: `tests/test_patch_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patch_eval.py`:

```python
from src.preprocessing.pipeline import preprocess_eval_patches


class TestPreprocessEvalPatches:
    def test_output_shape_agipd_size(self):
        # floor(1273/224)=5 → 5×5=25 patches
        img = np.random.default_rng(42).random((1273, 1273)).astype(np.float32)
        out = preprocess_eval_patches(img)
        assert out.shape == (25, 224, 224)

    def test_output_dtype_float32(self):
        img = np.zeros((500, 500), dtype=np.float32)
        out = preprocess_eval_patches(img, patch_size=224, stride=224)
        assert out.dtype == np.float32

    def test_raises_if_no_complete_patch(self):
        img = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="no complete"):
            preprocess_eval_patches(img)

    def test_custom_stride_changes_count(self):
        img = np.random.default_rng(0).random((500, 500)).astype(np.float32)
        default_out = preprocess_eval_patches(img, stride=224)
        overlap_out = preprocess_eval_patches(img, stride=112)
        assert overlap_out.shape[0] > default_out.shape[0]

    def test_each_patch_has_zero_mean_after_gcn(self):
        img = np.random.default_rng(7).random((500, 500)).astype(np.float32) * 1000
        out = preprocess_eval_patches(img)
        patch_means = out.reshape(out.shape[0], -1).mean(axis=1)
        np.testing.assert_allclose(patch_means, 0.0, atol=1e-5)

    def test_output_is_finite(self):
        img = np.random.default_rng(99).random((1273, 1273)).astype(np.float32)
        out = preprocess_eval_patches(img)
        assert np.isfinite(out).all()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_patch_eval.py::TestPreprocessEvalPatches -v
```

Expected: `ImportError` — `preprocess_eval_patches` not yet defined.

- [ ] **Step 3: Implement `preprocess_eval_patches` in `pipeline.py`**

Append after `preprocess_eval` in `src/preprocessing/pipeline.py`:

```python
def preprocess_eval_patches(
    assembled: np.ndarray,
    patch_size: int = TARGET_SIZE[0],
    stride: int | None = None,
    lcn_window: int = LCN_WINDOW_DEFAULT,
) -> np.ndarray:
    """GCN → LCN each patch from a patch_grid tiling of the assembled image.

    Used for all evaluation paths (validation, in-domain test, cross-detector
    test). The full native-resolution assembled frame is tiled into complete
    (patch_size × patch_size) patches; each patch is normalised independently.
    The caller runs all patches through the model and aggregates to a single
    frame-level score.

    Args:
        assembled: float32 array (H, W) at native detector resolution.
        patch_size: Patch side length in pixels (default 224).
        stride: Step between patch origins (default = patch_size, non-overlapping).
        lcn_window: LCN neighbourhood size (default 9, Phase 3 ablation).

    Returns:
        float32 array of shape (N, patch_size, patch_size) where N ≥ 1.

    Raises:
        ValueError: If the image produces zero complete patches.
    """
    from src.preprocessing.augment import patch_grid

    patches = patch_grid(assembled.astype(np.float32), patch_size, stride)
    if not patches:
        raise ValueError(
            f"preprocess_eval_patches: no complete {patch_size}×{patch_size} "
            f"patch fits in image of shape {assembled.shape}."
        )
    normed = [lcn(gcn(p), window=lcn_window) for p in patches]
    return np.stack(normed, axis=0).astype(np.float32)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_patch_eval.py::TestPreprocessEvalPatches -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/pipeline.py tests/test_patch_eval.py
git commit -m "feat: add preprocess_eval_patches — GCN→LCN per patch_grid tile"
```

---

## Task 3: `run_patch_agg` — frame-level score via patch inference

**Files:**
- Modify: `src/evaluation/benchmark.py`
- Test: `tests/test_patch_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patch_eval.py`:

```python
import torch
import torch.nn as nn
from pathlib import Path
from src.evaluation.benchmark import run_patch_agg


class _ConstantModel(nn.Module):
    """Always predicts the same 2-class logits for all inputs."""

    def __init__(self, hit_logit: float = 10.0):
        super().__init__()
        self.hit_logit = hit_logit

    def forward(self, x):
        batch = x.shape[0]
        return torch.tensor(
            [[-self.hit_logit, self.hit_logit]] * batch, dtype=torch.float32
        )


def _make_cxi(tmp_path, n_frames=4, n_hits=2, shape=(500, 500)):
    """Write a minimal CXI-like HDF5 file with data and labels."""
    import h5py

    path = tmp_path / "test.cxi"
    with h5py.File(path, "w") as f:
        data = np.random.default_rng(0).random((n_frames, *shape)).astype(np.float32)
        f.create_dataset("entry_1/data_1/data", data=data)
        labels = np.array(
            [1.0] * n_hits + [0.0] * (n_frames - n_hits), dtype=np.float32
        )
        f.create_dataset("entry_1/labels/hit", data=labels)
    return path


class TestRunPatchAgg:
    def test_returns_required_keys(self, tmp_path):
        path = _make_cxi(tmp_path)
        model = _ConstantModel()
        result = run_patch_agg(
            model,
            session_map={"s0": path},
            session_ids=["s0"],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        for key in ("ap", "auc_roc", "f1", "threshold"):
            assert key in result, f"Missing key: {key}"

    def test_all_metrics_are_finite(self, tmp_path):
        path = _make_cxi(tmp_path, n_frames=8, n_hits=4)
        model = _ConstantModel(hit_logit=2.0)
        result = run_patch_agg(
            model,
            session_map={"s0": path},
            session_ids=["s0"],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        for key in ("ap", "auc_roc", "f1", "threshold"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_empty_session_ids_returns_nan(self, tmp_path):
        model = _ConstantModel()
        result = run_patch_agg(
            model,
            session_map={},
            session_ids=[],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        assert np.isnan(result["ap"])

    def test_threshold_in_unit_interval(self, tmp_path):
        path = _make_cxi(tmp_path, n_frames=6, n_hits=3)
        model = _ConstantModel(hit_logit=2.0)
        result = run_patch_agg(
            model,
            session_map={"s0": path},
            session_ids=["s0"],
            label_key="entry_1/labels/hit",
            patch_stride=224,
            min_hit_patches=3,
            device="cpu",
        )
        assert 0.0 <= result["threshold"] <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_patch_eval.py::TestRunPatchAgg -v
```

Expected: `ImportError` — `run_patch_agg` not yet in `benchmark.py`.

- [ ] **Step 3: Implement `run_patch_agg` in `benchmark.py`**

Append after `run_on_loader` (before `run_fold`) in `src/evaluation/benchmark.py`:

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
    """Evaluate a model on full frames using patch-grid aggregation.

    For each frame: assemble to native resolution → tile into complete
    patch_size × patch_size patches → GCN → LCN each patch → run all patches
    through the model in mini-batches → reduce to a single frame-level score.

    Aggregation rule:
      frame_score = max(softmax[:, 1] over all patches)
      Used for AP, AUC-ROC, and the sklearn-optimised F1 threshold.

    This function replaces run_on_loader for all evaluation steps (validation
    during training, in-domain test, and cross-detector test). The gradient-
    update training loop (train_one_epoch) is unaffected.

    Args:
        model: Trained classifier with 2-class head.
        session_map: Mapping from session_id to CXI file path.
        session_ids: Subset of sessions to evaluate.
        label_key: HDF5 key for per-frame labels.
        patch_size: Patch side length in pixels (default 224).
        patch_stride: Step between patches in pixels. Use patch_size for
            non-overlapping grid; smaller values increase coverage.
        min_hit_patches: Unused in metric computation; reserved for external
            vote-based thresholding. Default 3.
        device: Torch device string ('cpu' or 'cuda').
        inference_batch_size: Patches per forward pass.

    Returns:
        dict with keys: ap, auc_roc, f1, threshold.
    """
    from src.preprocessing.geometry import get_assembler, get_geometry
    from src.preprocessing.io import (
        count_frames,
        read_detector_description,
        read_embedded_labels,
        read_frame,
    )
    from src.preprocessing.pipeline import _to_2d, assemble_only, preprocess_eval_patches

    model.to(device)
    model.eval()

    all_scores: list[float] = []
    all_labels: list[int] = []

    for sid in session_ids:
        path = Path(session_map[sid])
        labels_arr = read_embedded_labels(path, label_key)

        try:
            desc = read_detector_description(path)
            pads = get_geometry(desc)
            assembler = get_assembler(desc)
            use_geometry = True
        except (ValueError, KeyError, OSError):
            use_geometry = False

        n_frames = count_frames(path)
        for frame_idx in range(n_frames):
            frame = read_frame(path, frame_idx)

            if use_geometry:
                try:
                    assembled = assemble_only(frame, pads, desc, assembler)
                except (ValueError, KeyError, OSError):
                    assembled = _to_2d(frame)
            else:
                assembled = _to_2d(frame)

            patches_np = preprocess_eval_patches(
                assembled, patch_size=patch_size, stride=patch_stride
            )
            # (N, 224, 224) → add channel dim → (N, 1, 224, 224)
            patch_tensors = torch.from_numpy(patches_np).unsqueeze(1).to(device)

            patch_scores_list: list[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, len(patch_tensors), inference_batch_size):
                    batch = patch_tensors[i : i + inference_batch_size]
                    logits = model(batch)
                    if logits.ndim == 2 and logits.shape[1] == 2:
                        s = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    else:
                        s = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                    patch_scores_list.append(s)

            patch_scores = np.concatenate(patch_scores_list)
            all_scores.append(float(patch_scores.max()))
            all_labels.append(int(round(float(labels_arr[frame_idx]))))

    if not all_scores:
        nan = float("nan")
        return {"ap": nan, "auc_roc": nan, "f1": nan, "threshold": nan}

    y_score = np.array(all_scores)
    y_true = np.array(all_labels)
    best_f1, threshold = f1_at_optimal_threshold(y_true, y_score)
    return {
        "ap": average_precision(y_true, y_score),
        "auc_roc": auc_roc(y_true, y_score),
        "f1": best_f1,
        "threshold": threshold,
    }
```

Also update `__all__` in `benchmark.py` — add `"run_patch_agg"` to the list.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/test_patch_eval.py::TestRunPatchAgg -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full test suite**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/benchmark.py tests/test_patch_eval.py
git commit -m "feat: add run_patch_agg — full-frame patch-grid aggregation eval"
```

---

## Task 4: Wire `run_patch_agg` into `train_lodo.py` for all eval paths

This replaces every non-training evaluation call (validation per epoch AND final tests).
The `val_dl`, `in_domain_dl`, and `cross_dl` DataLoaders are removed — they existed only
to pass to `run_on_loader` / `evaluate`. `train_dl` (augmented) is kept.

**Files:**
- Modify: `scripts/train_lodo.py`

- [ ] **Step 1: Add `run_patch_agg` to the import block**

Change lines 40–50:

```python
from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
    format_results_table,
    run_on_loader,
    run_patch_agg,
    save_split_artifact,
)
```

- [ ] **Step 2: Build session-ID lists and remove eval DataLoaders**

In `_train_fold`, after the `label_key` / `augment_cfg` block, replace the
`val_dl` / `in_domain_dl` / `cross_dl` DataLoader construction with
session-ID lists:

```python
# Session ID lists for patch-aggregation eval (no DataLoader needed)
val_ids = [sid for sid, s in split_artifact["splits"].items() if s == SPLIT_VAL]
in_domain_ids = [
    sid for sid, s in split_artifact["splits"].items() if s == SPLIT_IN_DOMAIN_TEST
]
cross_ids = [
    sid for sid, s in split_artifact["splits"].items() if s == SPLIT_CROSS_DETECTOR
]
eval_cfg = cfg.get("evaluation", {})
patch_stride = eval_cfg.get("patch_stride", 224)
min_hit_patches = eval_cfg.get("min_hit_patches", 3)
```

Keep `train_dl` exactly as it is (augmented DataLoader).
Remove the three `_make_loader` calls for val / in-domain / cross and the
`n_val`, `n_indomain`, `n_cross` lines that read `.dataset` length from them.

- [ ] **Step 3: Replace per-epoch validation call**

In the training loop, find:

```python
val_m = evaluate(model, val_dl, criterion, device)
```

Replace with:

```python
val_m = run_patch_agg(
    model,
    session_map,
    val_ids,
    label_key=label_key,
    patch_stride=patch_stride,
    min_hit_patches=min_hit_patches,
    device=device,
)
```

`val_m` now has keys `ap, auc_roc, f1, threshold` — no `loss` key.
Find any reference to `val_m["loss"]` or `train_m["loss"]` in logging /
wandb calls and remove the `val_m["loss"]` ones (keep `train_m["loss"]`
if it exists — `train_one_epoch` still returns a loss).

- [ ] **Step 4: Replace final test evaluation calls**

Find:

```python
in_domain_m = run_on_loader(model, in_domain_dl, device)
cross_m = run_on_loader(model, cross_dl, device)
```

Replace with:

```python
in_domain_m = run_patch_agg(
    model,
    session_map,
    in_domain_ids,
    label_key=label_key,
    patch_stride=patch_stride,
    min_hit_patches=min_hit_patches,
    device=device,
)
cross_m = run_patch_agg(
    model,
    session_map,
    cross_ids,
    label_key=label_key,
    patch_stride=patch_stride,
    min_hit_patches=min_hit_patches,
    device=device,
)
```

- [ ] **Step 5: Run the full test suite**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v
```

Expected: all tests pass. (The training smoke tests use synthetic loaders and
don't call `_train_fold`, so they are unaffected.)

- [ ] **Step 6: Commit**

```bash
git add scripts/train_lodo.py
git commit -m "refactor: replace all eval DataLoaders with run_patch_agg in train_lodo"
```

---

## Task 5: Config update and format

**Files:**
- Modify: `configs/supervised/resnet18_lodo.yaml`

- [ ] **Step 1: Add `evaluation` section to `resnet18_lodo.yaml`**

Append after the `augmentation` block:

```yaml
evaluation:
  patch_stride: 224       # non-overlapping grid; reduce for more coverage
  min_hit_patches: 3      # vote threshold: frame=HIT if ≥3 patches score >0.5
```

- [ ] **Step 2: Run black**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/black src/ tests/ scripts/
```

Expected: reformats any touched files; `All done!`

- [ ] **Step 3: Final test run**

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python -m pytest tests/ -v
```

Expected: 225+ passed, 8 skipped.

- [ ] **Step 4: Commit**

```bash
git add configs/supervised/resnet18_lodo.yaml
git commit -m "config: add evaluation.patch_stride and evaluation.min_hit_patches"
```

---

## Verification (smoke test)

```bash
/home/gketawal/.conda/envs/sfx-hitfinder/bin/python - <<'EOF'
import numpy as np
from src.preprocessing.augment import patch_grid
from src.preprocessing.pipeline import preprocess_eval_patches

img = np.random.default_rng(0).random((1273, 1273)).astype(np.float32)
patches = patch_grid(img, 224, 224)
assert len(patches) == 25, f"Expected 25, got {len(patches)}"
print(f"AGIPD patches: {len(patches)} ✓")

out = preprocess_eval_patches(img)
assert out.shape == (25, 224, 224) and out.dtype == np.float32
print(f"preprocess_eval_patches shape: {out.shape} ✓")
EOF
```

Expected:
```
AGIPD patches: 25 ✓
preprocess_eval_patches shape: (25, 224, 224) ✓
```
