# Data Flow Walkthrough

A frame's journey from raw CXI bytes to a hit/no-hit score, stage by stage.

![Pipeline Overview](../assets/pipeline_architecture.png)

---

## Stage 1 — I/O: Reading a Raw Frame

**Where:** `src/preprocessing/io.py`
**Function:** `read_frame(cxi_path, frame_idx) → np.ndarray`
**Output shape:** Detector-native (e.g. `(512, 128)` panels for AGIPD, `(2160, 2162)` flat for Eiger4M)

```python
from src.preprocessing.io import read_frame, read_detector_description
frame = read_frame("run_001.cxi", frame_idx=42)          # raw ADU values
desc  = read_detector_description("run_001.cxi")          # e.g. "AGIPD"
```

**Modify here if you want to:** support a new file format (CBF, NeXus, HDF5 variant) or add a new metadata key.

---

## Stage 2 — Geometry Assembly

**Where:** `src/preprocessing/geometry.py`
**Functions:** `get_geometry(desc, cxi_path)` → geometry object; `get_assembler(geometry)` → `PADAssembler`
**Output shape:** `(H, W)` float32 — 2D detector canvas at native resolution

```python
from src.preprocessing.geometry import get_geometry, get_assembler
from src.preprocessing.pipeline import assemble_only
geom      = get_geometry(desc, "run_001.cxi")
assembler = get_assembler(geom)
canvas    = assemble_only(frame, assembler)               # (H, W) float32
```

**Detector → assembly method mapping:**

| Detector | Method |
|----------|--------|
| AGIPD | Reborn std loader + `PADAssembler(frame.ravel())` |
| JUNGFRAU_4M | CrystFEL `.geom` + `PADAssembler` |
| ePix10k | CrystFEL `.geom` + `PADAssembler` |
| Eiger4M | CrystFEL `.geom`, flat panels (pre-assembled) |

**Modify here if you want to:** add a new detector geometry. See [Extension Guide → New Detector](extension-guide.md#new-detector).

---

## Stage 3 — Hitfinding (training only)

**Where:** `src/hitfinders/`
**Function:** `hitfinder.find_peaks(assembled) → np.ndarray`
**Output shape:** `(N_peaks, 2)` — `[x, y]` pixel centroids; `(0, 2)` if no peaks

```python
from src.hitfinders import get_hitfinder
hf        = get_hitfinder("pf8", threshold=800, min_snr=5.0)
centroids = hf.find_peaks(canvas)                          # (N, 2) float32
```

> **Why before GCN?** PF8 threshold is calibrated in raw ADU units. Normalizing first would invalidate the threshold.

**At inference time:** no hitfinder is called. A patch grid replaces centroid-guided crops.

---

## Stage 4 — Preprocessing (GCN → Gap Fill → Pad / Grid)

**Where:** `src/preprocessing/{normalize,pipeline,augment}.py`

### Training path

```
canvas → gcn() → fill_gaps_after_gcn() → pad_border(112px) → centroid crop (224×224)
                                                             → miss crop (≥50px clearance)
```

### Inference path

```
canvas → gcn() → fill_gaps_after_gcn() → patch_grid(stride=224) → list of (224×224) patches
```

**Key functions:**

| Function | File | What it does |
|----------|------|-------------|
| `gcn(frame)` | `normalize.py` | `(I − μ) / (σ + ε)` — zero-mean, unit-variance |
| `fill_gaps_after_gcn(frame, mask)` | `pipeline.py` | Set invalid pixels (inter-panel gaps) to 0 |
| `pad_border(frame, 112)` | `augment.py` | Add 112px border so edge-peak crops don't go OOB |
| `patch_grid(frame, 224, 224)` | `augment.py` | Non-overlapping 224×224 tiles for inference |

> **Why gap fill before LCN?** LCN uses a 9×9 sliding window. Un-zeroed gap pixels would bleed into adjacent valid pixels through the window.

---

## Stage 5 — Dataset / Augmentation (training only)

**Where:** `src/data/dataset.py` → `AsymmetricCXIDataset.__getitem__`

Per-item pipeline:

1. `read_frame()` → raw frame
2. `assemble_only()` → canvas
3. `find_peaks()` → centroids
4. `gcn()` → normalized canvas
5. `fill_gaps_after_gcn()` → gapped canvas
6. `pad_border(112)` → padded canvas
7. **If centroids exist:** crop 224×224 centered on random centroid → `label = 1`
   **Else:** random crop ≥50px from all centroids (up to 50 tries) → `label = 0`
8. `lcn(patch, mask)` — 9×9 local normalization with valid-pixel mask
9. `random_rot90()`, `random_flip()`, `random_cutout(3 holes, 24×24, 8px margin)` — augmentation
10. Return `(1, 224, 224)` float32 tensor + label ∈ {0, 1}

**Output tensor contract:** `(B, 1, 224, 224)` float32, label `(B,)` int64

---

## Stage 6 — Model + Vote Aggregation + Evaluation

**Training:** `src/training/train_supervised.py` → `train_one_epoch(model, loader, optimizer, device)`

**Inference:** `src/evaluation/benchmark.py` → `run_patch_agg(model, sessions, cxi_paths, device)`

Vote aggregation per frame:

```
patch_grid → model(patch) → softmax → scores[:,1]
frame_score = count(scores > 0.5) / total_patches
```

LODO metrics per fold:

```python
{"ap": float, "auc_roc": float, "f1": float, "threshold": float}
```

Results saved to `checkpoints/<run-name>/results.json` and logged to wandb.

---

## Shape Summary

| Stage | Shape | dtype |
|-------|-------|-------|
| Raw frame (AGIPD) | `(512, 128)` panels | float32 |
| Assembled canvas | `(H, W)` — e.g. `(1024, 1024)` | float32 |
| Peak centroids | `(N, 2)` | float32 |
| GCN canvas | `(H, W)` | float64 |
| Training patch | `(1, 224, 224)` | float32 |
| Inference grid | `(N_patches, 224, 224)` | float64 |
| Model input batch | `(B, 1, 224, 224)` | float32 |
| Model output | `(B, 2)` logits | float32 |
| Frame score (vote) | scalar ∈ [0, 1] | float32 |

---

## Track 2 — SSL Pretraining Data Flow

The SSL pretraining path uses unlabeled diffraction frames. **No hitfinder runs.** Stages 3–4 are bit-for-bit identical to the Track 1 inference (eval) path.

### Stage 1 — I/O: Reading Unlabeled Frames

**Where:** `src/data/dataset.py` → `UnlabeledDataset.__getitem__`

```python
from src.data.dataset import UnlabeledDataset
dataset = UnlabeledDataset(file_list="data/splits/ssl_train.txt")
frame = dataset[42]    # raw assembled frame or panel stack
```

`.img` files (ADSC/MAR format) are already assembled — geometry step is skipped. CXI/HDF5 files go through Stage 2.

### Stage 2 — Assembly (CXI/HDF5 only)

**Where:** `src/preprocessing/pipeline.py` → `assemble_only(frame, assembler)`

Same Reborn PADAssembler path as Track 1. `.img` files skip this stage entirely.

### Stage 3 — GCN + Gap Fill

**Where:** `src/preprocessing/normalize.py`, `src/preprocessing/pipeline.py`

```
canvas → gcn() → fill_gaps_after_gcn(canvas, valid_pixel_mask)
```

Identical to Track 1. GCN zero-means and unit-normalizes the full assembled frame. Gap pixels are set to 0.

### Stage 4 — Patch Grid (no hitfinder, no labels)

**Where:** `src/preprocessing/augment.py` → `patch_grid(stride=224)`

```
gcn_canvas → patch_grid(stride=224) → list of (224×224) patches, no labels
```

This is the same patch grid used in Track 1 evaluation. No hitfinder is called. All patches are unlabeled.

### Stage 5 — MAE Pretraining

**Where:** `src/training/train_ssl_pretrain.py`

```
patches → random masking (75%) → ViT-S/16 encoder (visible patches only)
         → MAE decoder → reconstructed pixel values → MSE loss against masked patches
```

The encoder sees only the unmasked 25% of patches. The decoder reconstructs all masked patches. Loss is pixel MSE on masked regions only.

### Stage 6 — Fine-tuning

**Where:** `src/training/train_ssl_finetune.py`

The pretrained ViT-S/16 encoder is loaded, a linear classification head (`nn.Linear(encoder_dim, 2)`) is attached, and the model is fine-tuned on labeled CXI data using the same `AsymmetricCXIDataset` + `asymmetric_loader()` as Track 1.

### Track 2 Shape Summary

| Stage | Shape | dtype |
|-------|-------|-------|
| Unlabeled frame (assembled) | `(H, W)` | float32 |
| GCN canvas | `(H, W)` | float64 |
| Patch grid | `(N_patches, 224, 224)` | float32 |
| MAE encoder input (visible patches) | `(N_visible, patch_dim)` | float32 |
| MAE decoder output (reconstructed) | `(N_masked, patch_size²)` | float32 |
| Fine-tune model input | `(B, 1, 224, 224)` | float32 |
| Fine-tune model output | `(B, 2)` logits | float32 |
