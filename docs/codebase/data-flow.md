# Data Flow Walkthrough

A frame's journey from raw CXI bytes to a hit/no-hit score, stage by stage.

![Pipeline Overview](../assets/diagram.png)

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
9. `random_rot90()`, `random_flip()`, `random_cutout(3 holes, 32px, 8px margin)` — augmentation
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
