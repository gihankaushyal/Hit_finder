# Spec: Patch-Aggregation Evaluation for SFX Hitfinder

**Date:** 2026-06-30
**Branch:** phase-04-augmentation
**Status:** Approved — implementation in progress

---

## Problem

The current evaluation path applies a deterministic centre crop (224×224) to the fully
assembled detector frame before passing it to ResNet18. For a 1273×1273 AGIPD frame, this
evaluates only ~3% of the image area. Bragg peaks landing outside the central patch are
invisible to the classifier, causing artificially low cross-detector AP scores.

---

## Solution

Replace centre-crop eval with a patch-grid aggregation strategy:

1. Tile the full assembled native-resolution frame into a grid of non-overlapping
   (configurable: possibly slightly overlapping) 224×224 patches.
2. Normalise each patch independently: GCN → LCN.
3. Run all patches through the trained ResNet18 in mini-batches (64 patches/pass).
4. Aggregate to a single frame-level score:
   - **frame_score = max(softmax[:,1])** over all patches (primary metric — AP/AUC/F1)
   - Vote rule (secondary): frame = HIT if ≥ `min_hit_patches` patches score > 0.5
5. Compute AP, AUC-ROC, F1 at frame level.

---

## Scope

**In scope:**
- Validation loop during training (used for early stopping on val F1)
- In-domain test evaluation (final, after best checkpoint reload)
- Cross-detector test evaluation (final, LODO headline metric)

**Out of scope:**
- Training forward pass — `train_one_epoch` with augmented `train_dl` is unchanged
- `preprocess_assembled` / `preprocess_with_geometry` / `preprocess_eval` — unchanged
  (kept for backward compatibility with other scripts)

---

## Interface Contracts

### `patch_grid(image, patch_size=224, stride=None) → list[np.ndarray]`
- Input: 2D float32 array (H, W)
- Output: list of (patch_size × patch_size) float32 patches in row-major order
- Edge patches discarded (no padding)
- `stride` defaults to `patch_size` (non-overlapping); stride < patch_size → overlap
- Returns `[]` if image < patch_size in either dimension

### `preprocess_eval_patches(assembled, patch_size=224, stride=None, lcn_window=9) → np.ndarray`
- Input: float32 (H, W) native-resolution assembled image
- Output: float32 (N, 224, 224) — one GCN→LCN-normalised patch per grid position
- Raises `ValueError` if no complete patch fits

### `run_patch_agg(model, session_map, session_ids, ...) → dict`
- Keys: `ap`, `auc_roc`, `f1`, `threshold` (all float)
- Returns `{...: nan}` on empty session_ids
- Geometry routing: tries `assemble_only` (AGIPD/ePix10k/Eiger4M); falls back to `_to_2d`
  for Jungfrau 4M (pre-assembled canvas)
- No DataLoader — iterates CXI files directly via `read_frame` / `read_embedded_labels`

---

## Configuration

In `configs/supervised/resnet18_lodo.yaml`:

```yaml
evaluation:
  patch_stride: 224       # non-overlapping; reduce for more coverage
  min_hit_patches: 3      # vote-rule threshold (not used in AP/AUC computation)
```

---

## Patch Counts (stride=224, non-overlapping)

| Detector | Assembly size | floor(H/224) | Patches |
|----------|-------------|-------------|---------|
| AGIPD | ~1273×1273 | 5 | 25 |
| ePix10k | ~1667×1667 | 7 | 49 |
| Eiger4M | ~1687×1687 | 7 | 49 |
| Jungfrau 4M | 2164×2068 | 9 | 81 |

---

## Files Changed

| File | Change |
|------|--------|
| `src/preprocessing/augment.py` | Add `patch_grid` |
| `src/preprocessing/pipeline.py` | Add `preprocess_eval_patches` |
| `src/evaluation/benchmark.py` | Add `run_patch_agg`, update `__all__` |
| `scripts/train_lodo.py` | Replace `val_dl`/`in_domain_dl`/`cross_dl` + `run_on_loader`/`evaluate` calls with `run_patch_agg` |
| `configs/supervised/resnet18_lodo.yaml` | Add `evaluation` section |
| `tests/test_patch_eval.py` | New — 18+ tests covering all new functions |
