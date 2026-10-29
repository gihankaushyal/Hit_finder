# Data Flow Walkthrough

A frame's journey from raw CXI bytes to a hit/no-hit score, stage by stage.

![Pipeline Overview](../assets/pipeline_architecture.png)

> **Cacheable prefix.** Stages 1–4 (read → assemble → hitfinder → GCN + gap fill) are deterministic per frame. `AsymmetricCXIDataset` and `SSLPretrainCXIDataset` obtain their result through `_load_gcn_frame()` in `src/data/dataset.py`, which serves it from the on-disk `FrameCache` when one is configured and otherwise calls `_compute_gcn_frame()`. See [Frame cache](#frame-cache-optional) below.

---

## Stage 1 — I/O: Reading a Raw Frame

**Where:** `src/preprocessing/io.py`
**Function:** `read_frame(path, frame_idx=0) → np.ndarray`
**Output shape:** detector-native, as stored in the CXI file — `(16, 512, 128)` for AGIPD, `(2164, 2068)` for the Jungfrau 4M canvas, `(5632, 384)` stacked panels for ePix10k and Eiger4M (Resonet production files, measured 2026-10-22)

```python
from src.preprocessing.io import read_frame, read_detector_description
frame = read_frame("compressed0.cxi", frame_idx=42)       # raw ADU values
desc  = read_detector_description("compressed0.cxi")      # e.g. "AGIPD 1M", "EIGER 4M"
```

The data key is found by trying `entry/data/data`, `…/detector_corrected/data`, `entry_1/data_1/data`, `entry_0000/instrument/Simulator/data` in order. Embedded hit labels are read with `read_embedded_labels(path, label_key="entry_1/labels/hit")`. Detector type always comes from the metadata description, never from the pixels.

**Modify here if you want to:** support a new file format (CBF, NeXus, HDF5 variant) or add a new metadata key.

---

## Stage 2 — Geometry Assembly

**Where:** `src/preprocessing/geometry.py`, `src/preprocessing/pipeline.py`
**Functions:** `get_geometry(detector_desc)` → cached `PADGeometryList`; `get_assembler(detector_desc)` → cached `PADAssembler`; `assemble_only(frame, pads, detector_desc, assembler=None)` → 2D canvas
**Output shape:** `(H, W)` float32 — 2D detector canvas at native resolution

```python
from src.preprocessing.geometry import get_geometry, get_assembler
from src.preprocessing.pipeline import assemble_only
pads      = get_geometry(desc)                              # desc from the CXI description
assembler = get_assembler(desc)
canvas    = assemble_only(frame, pads, desc, assembler=assembler)   # (H, W) float32
```

**Detector → assembly method mapping** (all go through Reborn's `PADAssembler`; see `DETECTOR_LOADERS` / `_DESC_LOADERS` in `geometry.py`):

| Detector (CXI description) | Geometry loader | Stored → assembled |
|----------|--------|--------------------|
| AGIPD (`AGIPD 1M`) | `reborn.detector.agipd_pad_geometry_list` | `(16, 512, 128)` → `(1273, 1273)` |
| JUNGFRAU_4M (`Jungfrau 4M`) | `jungfrau_8_pad_geometry_list` (Reborn bundled; 8 panels of 514×1030) | `(2164, 2068)` canvas → `(2173, 2173)` |
| ePix10k (`ePix10k 2.2M`) | `reborn.detector.epix10k_pad_geometry_list` (64 panels) | `(5632, 384)` → `(1667, 1667)` |
| Eiger4M (`EIGER 4M`) | `eiger4m_64_pad_geometry_list` (64 panels of 176×192) | `(5632, 384)` → `(1687, 1687)` |
| EigerRESoNeT | `eiger_resonet_pad_geometry_list` (CrystFEL `eiger_resonet.geom`, same layout) | `(5632, 384)` → `(1687, 1687)` |

Canvas-style inputs are first sliced into panels with `extract_panels_from_canvas()` before `PADAssembler.assemble_data()`. If the description cannot be read, the dataset falls back to a raw 2D passthrough for that file instead of crashing.

**Modify here if you want to:** add a new detector geometry. See [Extension Guide → New Detector](extension-guide.md#new-detector).

---

## Stage 3 — Hitfinding (training; metadata-gated)

**Where:** `src/hitfinders/`
**Function:** `hitfinder.find_peaks(assembled) → np.ndarray`
**Output shape:** `(N_peaks, 2)` — `[x, y]` pixel centroids; `(0, 2)` if no peaks

```python
from src.hitfinders import get_hitfinder
from src.utils.config import load_config
cfg       = load_config("configs/supervised/resnet18_asymmetric.yaml")
hf        = get_hitfinder(cfg)           # reads cfg["hitfinder"]: backend, pf8_threshold, pf8_min_snr, …
centroids = hf.find_peaks(canvas)        # (N, 2) float32
```

The CXI geometry keys (`distance`, `wavelength`, `x_pixel_size`) are read lazily per file and passed to `hitfinder.set_geometry()` when the file changes.

> **Why before GCN?** PF8 threshold is calibrated in raw ADU units. Normalizing first would invalidate the threshold.

**Metadata gating:** `AsymmetricCXIDataset` only passes a hitfinder to the prefix when the frame's embedded label is HIT; for NON-HIT frames the hitfinder is skipped and the centroid array is empty. `SSLPretrainCXIDataset` runs no hitfinder unless peak-aware masking is configured, and `_load_gcn_frame` discards cached centroids when `hitfinder is None`.

**At inference time:** no hitfinder is called. A patch grid replaces centroid-guided crops.

---

## Stage 4 — Normalization (GCN → Gap Fill), then Pad / Grid

**Where:** `src/preprocessing/{normalize,pipeline,augment}.py`

### Training path

```
canvas → gcn() → fill_gaps_after_gcn() → pad_border(112px) → centroid crop (224×224)     [Path A, label 1]
                                                           → miss crop (≥50px clearance) [Path B, label 0]
```

### Inference path

```
canvas → gcn() → fill_gaps_after_gcn() → patch_grid(stride=224) → list of (224×224) patches → masked LCN per patch
```

**Key functions:**

| Function | File | What it does |
|----------|------|-------------|
| `gcn(frame)` | `normalize.py` | `(I − μ) / (σ + ε)` over the full assembled frame |
| `get_valid_mask_for_frame(desc, shape)` | `pipeline.py` | Geometry-derived valid-pixel mask (gap/padding/edge pixels excluded; eroded by `EDGE_EROSION_PX = 2`) |
| `fill_gaps_after_gcn(frame, desc, mask)` | `pipeline.py` | Set invalid pixels to 0 (= global mean in GCN units) |
| `pad_border(frame, 112)` | `augment.py` | Add a 112 px border so edge-peak crops don't go out of bounds |
| `patch_grid(frame, 224, 224)` | `augment.py` | Non-overlapping 224×224 tiles for inference |
| `preprocess_eval_patches_from_gcn(gcn_frame, mask, patch_size, stride, lcn_window)` | `pipeline.py` | Evaluation tiling + per-patch masked LCN from an already-GCN'd frame |

> **Why gap fill before LCN?** LCN uses a 9×9 sliding window. Un-zeroed gap pixels would bleed into adjacent valid pixels through the window; the valid mask also excludes them from the local mean/variance.

### Frame cache (optional)

**Where:** `src/data/frame_cache.py`, built by `scripts/build_frame_cache.py`, enabled by the `cache:` block of a run config (`enabled: false` in `configs/base.yaml`, `true` in the production LODO configs; `--no-cache` / `--cache-root` / `--cache-nvme` on `train_asymmetric`).

The cache stores, per CXI file, `frames.npy` (`(N, H, W)` **float16**, memory-mapped), `centroids.npz` and a per-detector `valid_mask.npy`, under a manifest `cache_manifest.json` that records the preprocessing parameters (hitfinder keys, GCN/LCN constants, geometry digests). `verify_cache_or_raise()` raises `CacheStaleError` when the manifest no longer matches the config; a missing entry raises `CacheMissError`, which `_load_gcn_frame` treats as a normal miss and recomputes live. A two-tier layout lets `scripts/stage_frame_cache.sh` copy a fold's entries to node-local NVMe.

Cached frames are cast back to float32 on read and match the live path up to float16 storage rounding; masks and centroids match exactly (`tests/test_frame_cache.py::TestBitExactnessAgainstRealData`). A change to *how* a detector's assembler is chosen is not covered by the manifest and needs a full cache rebuild. Geometry bundled inside Reborn (the Eiger4M, Jungfrau, AGIPD and ePix10k loaders) is likewise invisible to the geometry digest, which hashes only the `.geom` and `.json` files under `src/preprocessing/data/`.

---

## Stage 5 — Dataset / Augmentation (training only)

**Where:** `src/data/dataset.py` → `AsymmetricCXIDataset.__getitem__`

Per-item pipeline (one item = one frame; it yields `crops_per_frame` crops):

1. Read the embedded label (`entry_1/labels/hit`); a hitfinder is used only when it is HIT
2. `_load_gcn_frame()` → `(gcn_frame, valid_mask, centroids)`: read → assemble → [hitfinder on the raw frame] → `gcn()` → `fill_gaps_after_gcn()`, from the cache or computed live
3. `pad_border(112)` → padded canvas; centroids and mask are shifted to match
4. Pick the crop for each of the `crops_per_frame` draws:
   - **HIT frame with centroids:** 50/50 coin toss (`hit_frac`, default 0.5). **Path A** — 224×224 centred on a random centroid → `label = 1`. **Path B** — random crop ≥ 50 px from all centroids (up to `hard_neg_max_attempts` = 50 tries; falls back to Path A if none is found) → `label = 0`
   - **HIT frame, no centroids found, or NON-HIT frame:** random crop → `label = 0`
5. `random_rot90()`, `random_flip()` — geometric augmentation
6. `lcn(crop, mask)` — 9×9 masked local normalization (`ε = 1e-2`, variance form)
7. `random_cutout(3 holes, 24×24, exact 0, 8 px margin from every centroid)` — after LCN; a hole is skipped after 20 failed draws
8. Return `(crops_per_frame, 1, 224, 224)` float32 and `(crops_per_frame,)` int64 labels (or `None` if the crops could not be drawn within `crops_per_frame × 4` attempts)

Crop positions are re-seeded per epoch: `set_epoch(epoch)` folds the epoch into the per-item RNG seed, so each epoch draws different crops from the same cached frame. `asymmetric_loader()` flattens the per-frame stacks into ordinary batches via `_flatten_collate`.

**Output tensor contract:** `(B, 1, 224, 224)` float32, label `(B,)` int64, where `B = batch_size × crops_per_frame` for `asymmetric_loader()` (`batch_size` counts frames)

---

## Stage 6 — Model + Vote Aggregation + Evaluation

**Training:** `src/training/lodo.py` → `_train_fold()`, which calls `src/training/train_supervised.py` → `train_one_epoch(model, loader, optimizer, device)` each epoch.

**Inference:** `src/evaluation/benchmark.py` → `run_patch_agg(model, session_map, session_ids, label_key, patch_size=224, patch_stride=224, aggregation="vote", …)`

Vote aggregation per frame:

```
patch_grid → masked LCN → model(patch) → softmax → scores[:,1]
frame_score = count(scores > 0.5) / total_patches
```

LODO metrics per fold (`run_patch_agg` returns; `f1` / `threshold` are the optimum on the evaluated frames):

```python
{"ap": float, "auc_roc": float, "f1": float, "threshold": float}
```

Results are saved to `checkpoints/<run-prefix>-fold<N>-seed<S>/results.json` (`cross` and `in_domain` blocks, validation-set `inference_threshold`, and a `training` block) and logged to wandb. `--inference-only` re-evaluates a finished run and writes `results.json` if absent, else `results.inference.json`. See [eval_protocol.md](../eval_protocol.md).

---

## Shape Summary

| Stage | Shape | dtype |
|-------|-------|-------|
| Raw frame (AGIPD) | `(16, 512, 128)` | as stored (float32) |
| Raw frame (ePix10k / Eiger4M) | `(5632, 384)` | as stored |
| Assembled canvas | `(H, W)` — `(1273, 1273)` AGIPD, `(2173, 2173)` Jungfrau, `(1667, 1667)` ePix10k, `(1687, 1687)` Eiger4M | float32 |
| Peak centroids | `(N, 2)` | float32 |
| GCN canvas (live) | `(H, W)` | float64 |
| GCN canvas (cached) | `(H, W)`, stored as float16, returned float32 | float16 → float32 |
| Valid mask | `(H, W)` | bool |
| Training crops (per item) | `(crops_per_frame, 1, 224, 224)` | float32 |
| Inference grid | `patch_grid` returns a list of `(224, 224)` patches | float64 |
| Model input batch | `(B, 1, 224, 224)` | float32 |
| Model output | `(B, 2)` logits | float32 |
| Frame score (vote) | scalar ∈ [0, 1] | float |

> **Note:** `gcn()` returns float64 (checked 2026-10-22); the batched torch LCN used during evaluation (`lcn_torch`) produces the float32 model input.

---

## Track 2 — SSL Pretraining Data Flow

The SSL pretraining path uses unlabeled diffraction frames. **No labels are used**, and no hitfinder runs unless peak-aware masking is configured. Steps 1–4 are the same `_load_gcn_frame` prefix as Track 1, and the crop step is followed by the same rot90 / flip / masked LCN; there is no cutout (MAE masking replaces it).

### Stage 1 — I/O and Assembly

**Where:** `src/data/dataset.py` → `SSLPretrainCXIDataset.__getitem__` → `_load_gcn_frame()`

```python
from src.data.dataloader import ssl_crop_loader
loader = ssl_crop_loader(session_map, session_ids, batch_size=256, num_workers=32,
                         shuffle=True, seed=42, crops_per_frame=4, hitfinder=None,
                         min_valid_frac=0.5, frame_cache=frame_cache)
```

`session_ids` contains only the training detectors' sessions — the fold's held-out detector is excluded by the caller (strict LODO; `train_ssl_pretrain.py --fold N`). Assembly uses the same Reborn path as Track 1 (Stage 2 above).

### Stage 2 — GCN + Gap Fill

`gcn()` then `fill_gaps_after_gcn()`, identical to Track 1, run once per frame (or read from the frame cache); all `crops_per_frame` crops share that result.

### Stage 3 — Random valid-region crops (no hit labels)

For each crop, up to `SSL_CROP_MAX_TRIES` random 224×224 positions are drawn and the first with at least `min_valid_frac` (0.5) valid pixels is used (the best candidate otherwise). Crop positions vary per epoch (`dl.dataset.set_epoch(epoch)` is called every epoch). Each crop is then rotated/flipped and masked-LCN normalized. Each item is a tuple `(crops, peaks, vmasks)` of shapes `(crops_per_frame, 1, 224, 224)`, `(crops_per_frame, 196)` (bool, ViT patches containing a hitfinder centroid; all False without a hitfinder), `(crops_per_frame, 1, 224, 224)`.

> The older whole-image `UnlabeledDataset` / `ssl_pretrain_loader()` path (`.img` / CXI files, no crops) still exists but is no longer used by the training scripts.

### Stage 4 — MAE Pretraining

**Where:** `src/training/train_ssl_pretrain.py`, `src/models/ssl.py` (`MAEViT`)

```
crops → random masking (ssl.mask_ratio = 0.6) → ViT-S/16 encoder (visible patches only)
      → MAE decoder → reconstructed pixel values → MSE loss against masked patches
```

The encoder sees only the unmasked 40% of patches (`masking: random`; `peak_aware` is available via `ssl.masking`). The decoder reconstructs all masked patches. Loss is pixel MSE on masked regions only (`norm_pix_loss: false`, because LCN already normalizes locally).

### Stage 5 — Fine-tuning or linear probe

**Where:** `src/training/train_ssl_finetune.py`, `src/models/ssl.py` → `build_ssl_classifier(cfg, mae_checkpoint, freeze_encoder)`

The pretrained ViT-S/16 encoder is loaded from the fold's `last.pt`, a 2-class head is attached (`ViTClassifier`), and the model is trained on labeled CXI data using the same `AsymmetricCXIDataset` + `asymmetric_loader()` and `_train_fold()` as Track 1. With `--linear-probe` the encoder is frozen and only the head trains. Evaluation is the same frame-level `run_patch_agg` vote protocol.

### Track 2 Shape Summary

| Stage | Shape | dtype |
|-------|-------|-------|
| GCN canvas | `(H, W)` | float64 live / float16 cached |
| Pretraining crop batch | `(batch_size, 1, 224, 224)` after flatten; for `ssl_crop_loader()` `batch_size` counts crops and must be divisible by `crops_per_frame` | float32 |
| Peak-patch flags | `(…, 196)` | bool |
| MAE encoder input (visible patches) | `(N_visible, patch_dim)` — 196 patches of 16×16 per crop, 60% masked | float32 |
| MAE decoder output (reconstructed) | `(N_masked, 16·16)` | float32 |
| Fine-tune model input | `(B, 1, 224, 224)` | float32 |
| Fine-tune model output | `(B, 2)` logits | float32 |
