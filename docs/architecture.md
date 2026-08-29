# Architecture

## Overview

Hit_finder is a deep learning system for per-frame hit classification in Serial Femtosecond Crystallography (SFX). Given raw detector frames from a CXI file, it outputs a binary hit/no-hit decision for each frame — whether a frame contains Bragg diffraction spots indicating a crystalline sample in the beam.

The system is designed around two core challenges: (1) **detector diversity** — models must generalize across four distinct detector geometries from different facilities without retraining, and (2) **label quality** — frame-level labels from a classical peak-finder are noisy, so patch-level supervision guided by Bragg peak locations produces more reliable training signal.

Two pipeline tracks exist. **Track 1 (primary, production)** is an asymmetric supervised pipeline using a ResNet18 classifier trained on hitfinder-guided 224×224 patches. **Track 2 (planned)** is a self-supervised (MAE) pretraining stage followed by supervised finetuning, intended to exploit the large corpus of unlabeled CXI frames.

---

## System Diagram

```
CXI file (HDF5)
     │
     ▼
Geometry Assembly ─────────────────────────────────────────►  (H, W) float32
     │                                         [detector-native canvas]
     ▼
Hitfinder (PF8 / GPU / NumPy) ─────────────────────────────►  (N, 2) peak centroids
     │
     ▼
Global Contrast Normalization (GCN) ────────────────────────►  (H, W) mean=0, std≈1
     │
     ├── TRAINING ──────────────────────────────────────────────────────────────
     │   Hit crop  (centroid ±112px)   → LCN → peak-aware cutout → (1,224,224) label=1
     │   Miss crop (≥50px clearance)   → LCN → peak-aware cutout → (1,224,224) label=0
     │
     └── INFERENCE ─────────────────────────────────────────────────────────────
         Non-overlapping 224×224 grid  → LCN → Model → softmax[:,1]
                                                        │
                                                        ▼
                                             frame_score = hit_patches / N
```

---

## Data: Detector Diversity and LODO

The system is trained and evaluated across four SFX detectors from different facilities:

| Detector     | Facility      | Native Shape    | Assembly Method                              |
|--------------|---------------|-----------------|----------------------------------------------|
| AGIPD        | EuXFEL        | ~(1024, 1024)   | Reborn std loader + PADAssembler             |
| JUNGFRAU_4M  | LCLS-CXI      | ~(2048, 2048)   | CrystFEL `.geom` + PADAssembler              |
| ePix10k      | LCLS          | ~(1480, 1552)   | CrystFEL `.geom` + PADAssembler              |
| Eiger4M      | Synchrotron   | (2160, 2162)    | CrystFEL `.geom`, flat panel (pre-assembled) |

**LODO evaluation** holds out one detector at test time and trains on the other three. This tests whether learned features are genuinely detector-agnostic rather than encoding detector-specific artifacts.

Sessions (beamtime runs) are the unit of splitting. An 80/10/10 session-stratified split (seed=42) is applied within the training detectors to produce train / val / in-domain-test subsets. The held-out detector forms a separate cross-detector test split that the model never sees during training or hyperparameter selection.

---

## Preprocessing Pipeline

All preprocessing runs on CPU in DataLoader workers (GPU hitfinder is the exception: it requires `num_workers=0` due to OpenCL context forking constraints).

| Stage | Input | Output | Key Parameters |
|-------|-------|--------|----------------|
| **Assembly** | Raw detector frame | `(H, W)` float32 canvas | Geometry from CXI metadata or `.geom` file; see detector table above |
| **Hitfinder** | `(H, W)` raw assembled (pre-GCN) | `(N, 2)` [x, y] centroids | `threshold=800 ADU`, `min_snr=5.0`, `min_pix_count=2`, `max_pix_count=200`, `local_bg_radius=3` |
| **GCN** | `(H, W)` frame | `(H, W)` float64, mean=0, std≈1 | `(I − μ) / (σ + ε)` global |
| **Gap Fill** | GCN'd frame + invalid-pixel mask | GCN'd frame with invalid pixels zeroed | Prevents inter-module gaps from corrupting LCN windows |
| **Pad** *(train only)* | `(H, W)` frame | `(H+224, W+224)` | 112px border; enables centered crops at panel edges |
| **LCN** | `(224, 224)` patch + mask | `(224, 224)` float64 | `(I − μ_W) / sqrt(σ²_W + ε)`, 9×9 sliding window |
| **Augmentation** *(train only)* | `(1, 224, 224)` patch | Augmented patch | rot90 (uniform), H/V-flip (50% each), peak-aware cutout |

> **Why hitfinder before GCN?** PF8 threshold parameters are calibrated in raw ADU units, so it must run on the unmodified assembled frame. GCN runs after peak extraction.

> **Why gap fill before LCN?** Invalid pixels (inter-panel gaps) must be zeroed after GCN so that the 9×9 LCN window does not average across them and propagate artificial signal into valid regions.

---

## Track 1 — Asymmetric Supervised Pipeline (Primary)

### The Core Idea

Classical SFX hit-finding assigns one binary label per frame. This is noisy as supervision: a frame labeled "hit" contains mostly background pixels, and a random 224×224 crop is unlikely to intersect a Bragg centroid. The asymmetric pipeline re-frames the problem at patch level:

- **Hit patch (label = 1):** 224×224 crop centered on a randomly chosen Bragg centroid from the hitfinder.
- **Miss patch (label = 0, hard negative):** 224×224 crop sampled at least 50 px from all centroids (rejection-sampled, up to 50 attempts). This forces the model to distinguish true Bragg signal from background texture rather than memorizing frame-level intensity statistics.

`hit_frac=0.5` controls the training balance between hit and miss patches, decoupled from the dataset's natural hit rate.

### Model

- **Architecture:** ResNet18 via `timm`, `in_chans=1` (ImageNet weights adapted to single-channel via weight averaging), `num_classes=2`
- **Input:** `(B, 1, 224, 224)` float32 (GCN + LCN normalized patch)
- **Output:** `(B, 2)` logits → softmax → `score = softmax[:, 1]` (hit probability)
- **Optimizer:** AdamW, lr=1e-4, weight_decay=1e-4
- **Training:** 100 epochs, early stopping on val F1 (patience=10)

### Peak-Aware Augmentation

Cutout augmentation (3 holes, 32×32 px) respects a peak-protection zone of 8 px around each Bragg centroid. This prevents cutout from masking the very signal that defines a hit patch, while still disrupting background structure to improve robustness.

### Key Files

| Role | File |
|------|------|
| Dataset | `src/data/dataset.py` → `AsymmetricCXIDataset` |
| DataLoader | `src/data/dataloader.py` → `asymmetric_loader()` |
| Model | `src/models/supervised.py` → `build_supervised_model()` |
| Training entry point | `scripts/train_asymmetric.py` |
| Config | `configs/supervised/resnet18_asymmetric.yaml` |

---

## Track 2 — SSL Pipeline (Planned)

Track 2 adds a self-supervised pretraining stage to initialize representations from the large corpus of unlabeled CXI frames before supervised finetuning on asymmetric patches.

**Stage 1 — MAE Pretraining**

- **Architecture:** ViT-Base (`vit_base_patch16_224`), 16×16 patch tokens → 196 patches per 224×224 image
- **Objective:** Reconstruct 75% of randomly masked patches from the visible 25%
- **Training:** 200 epochs, batch size 64, lr=1.5e-4
- **Data:** `UnlabeledDataset` over all CXI files (no hitfinder or labels needed)
- **Config:** `configs/ssl/mae_pretrain.yaml`

**Stage 2 — Supervised Finetuning**

- Replace the MAE decoder with a 2-class classification head
- Fine-tune using the same asymmetric patch pipeline as Track 1
- **Training:** 30 epochs, batch size 32, lr=1e-5 (lower to preserve pretrained representations)
- **Config:** `configs/ssl/mae_finetune.yaml`

> **Status:** Code is implemented (`src/models/ssl.py`, `src/training/train_ssl_pretrain.py`, `src/training/train_ssl_finetune.py`). No training runs submitted yet. Track 2 is blocked on establishing a solid Track 1 baseline (Phase 4 → Phase 5).

---

## Hitfinder Backends

The hitfinder is called during training to label each crop. It is abstracted behind a protocol interface so backends are swappable without changing training code:

```python
class Hitfinder(Protocol):
    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """(H, W) float32 raw frame → (N_peaks, 2) [x, y] centroids"""
```

Three implementations:

| Backend | Implementation | Speed | Multiprocessing | Use case |
|---------|---------------|-------|-----------------|----------|
| **PF8Hitfinder** | C ctypes bridge (`_pf8_wrap.so`, CrystFEL libcrystfel) | Fast (CPU) | Fork-safe ✓ | Production training |
| **GPUHitfinder** | pyFAI `OCL_PeakFinder` (OpenCL) | Fast (GPU) | Requires `num_workers=0` | GPU-constrained runs |
| **NumpyPF8 / PF8Python** | Pure NumPy / pure Python | Slow | Fork-safe ✓ | Unit tests |

The PF8 backend is the production default. Its parameters (`threshold`, `min_snr`, `min_pix_count`, `max_pix_count`, `local_bg_radius`) are set in the YAML config.

---

## Inference: Vote Aggregation

At inference time no hitfinder is needed. The full assembled, GCN-normalized frame is tiled into a non-overlapping 224×224 grid (stride=224), each patch is independently classified, and the results are aggregated to a frame-level score:

```
frame_score = count(patches where softmax[:, 1] > 0.5) / total_patches
```

A frame is a hit if `frame_score > threshold`, where the optimal threshold is determined on the validation set. Patches are processed in mini-batches of 64 by `run_patch_agg()` in `src/evaluation/benchmark.py`.

> **Why vote over max?** Max aggregation is sensitive to the single most confident patch and produces unstable frame scores. Vote aggregation treats the score as the fraction of patches exhibiting Bragg-like features, which correlates better with actual diffraction quality and allows calibrated threshold selection across detectors.

---

## Evaluation: LODO Protocol

| Fold | Held-out detector | Train detectors |
|------|------------------|----------------|
| 1 | AGIPD | JUNGFRAU_4M, ePix10k, Eiger4M |
| 2 | JUNGFRAU_4M | AGIPD, ePix10k, Eiger4M |
| 3 | ePix10k | AGIPD, JUNGFRAU_4M, Eiger4M |
| 4 | Eiger4M | AGIPD, JUNGFRAU_4M, ePix10k |

Metrics are computed at the frame level after vote aggregation:

| Metric | Role |
|--------|------|
| **Average Precision (AP)** | Primary — area under precision-recall curve |
| **AUC-ROC** | Ranking quality across full threshold range |
| **F1 at optimal threshold** | Single operating-point summary |

AP uses sklearn-compatible stable sort with pessimistic ordering (tied positives are not promoted). See `src/evaluation/metrics.py` for implementations and `src/evaluation/benchmark.py` for the LODO orchestration.
