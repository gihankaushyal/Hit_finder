# Architecture

## Overview

Hit_finder is a deep learning system for per-frame hit classification in Serial Femtosecond Crystallography (SFX). Given raw detector frames from a CXI file, it outputs a binary hit/no-hit decision for each frame — whether a frame contains Bragg diffraction spots indicating a crystalline sample in the beam.

The system is designed around two core challenges: (1) **detector diversity** — models must generalize across four distinct detector geometries from different facilities without retraining, and (2) **label quality** — frame-level labels from a classical peak-finder are noisy, so patch-level supervision guided by Bragg peak locations produces more reliable training signal.

Two pipeline tracks exist. **Track 1 (primary, production)** is an asymmetric supervised pipeline using a ResNet18 classifier trained on hitfinder-guided 224×224 patches. **Track 2 (implemented; full-LODO runs complete for the September 2026 generation, rerun in progress)** is a self-supervised (MAE, ViT-S/16) pretraining stage followed by supervised fine-tuning or a linear probe, exploiting the large corpus of unlabeled CXI frames. Both tracks share one preprocessing pipeline (the same code path up to the crop step; Track 2 uses MAE masking instead of cutout).

A self-contained local web dashboard for the repository lives in `dashboard/` (see [Dashboard](#dashboard)); it is not imported by `src/`.

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
     │   Hit crop  (centroid ±112px)   → rot90/flip → LCN → peak-aware cutout → (1,224,224) label=1
     │   Miss crop (≥50px clearance)   → rot90/flip → LCN → peak-aware cutout → (1,224,224) label=0
     │
     └── INFERENCE ─────────────────────────────────────────────────────────────
         Non-overlapping 224×224 grid  → LCN → Model → softmax[:,1]
                                                        │
                                                        ▼
                                             frame_score = hit_patches / N
```

**Frame cache.** Everything from the CXI read through GCN is deterministic per frame, so it can be cached (`src/data/frame_cache.py`, enabled by the `cache:` block in a run config; off by default in `configs/base.yaml`, on in the production LODO configs). The cache is two-tier — a shared on-disk root plus an optional per-node NVMe copy staged by `scripts/stage_frame_cache.sh` — and guarded by a manifest (`verify_cache_or_raise`) that raises `CacheStaleError` / `CacheMissError` when the geometry, hitfinder settings or files have changed. On a miss the dataset falls back to computing the prefix directly (`_compute_gcn_frame`). Frames are stored as float16 (`frames.npy`, memory-mapped) and cast back to float32 on read, so cached frames match the live path up to float16 rounding; valid-pixel masks and centroids match exactly. `TestBitExactnessAgainstRealData` in `tests/test_frame_cache.py` checks this against real data and is skipped wherever no real cache has been built. Crops are still drawn fresh every epoch (epoch-varying crops), with `asymmetric.crops_per_frame` independent crops per cached frame.

**Metadata gating.** The hitfinder only runs on frames whose embedded label (`entry_1/labels/hit`) is HIT. For a HIT frame with centroids, a 50/50 coin toss picks Path A (centred on a random peak, label 1) or Path B (50 px clearance, label 0; falls back to Path A if no clear position is found). A HIT frame with no centroids, or any NON-HIT frame (hitfinder skipped, centroids empty), yields a random crop with label 0.

---

## Data: Detector Diversity and LODO

The system is trained and evaluated across four SFX detectors from different facilities:

| Detector     | Facility      | Stored in CXI          | Assembled     | Geometry loader (`src/preprocessing/geometry.py`)            |
|--------------|---------------|------------------------|---------------|--------------------------------------------------------------|
| AGIPD        | EuXFEL        | (16, 512, 128)         | (1273, 1273)  | `reborn.detector.agipd_pad_geometry_list` + PADAssembler     |
| JUNGFRAU_4M  | LCLS-CXI      | (2164, 2068) canvas    | (2173, 2173)  | `jungfrau_8_pad_geometry_list` (8 panels of 514×1030)        |
| ePix10k      | LCLS          | (5632, 384) stacked    | (1667, 1667)  | `reborn.detector.epix10k_pad_geometry_list` (64 panels)      |
| Eiger4M      | Synchrotron   | (5632, 384) stacked    | (1687, 1687)  | `eiger4m_64_pad_geometry_list` (64 panels of 176×192)        |

All four go through Reborn's `PADAssembler`; stacked/canvas frames are sliced into panels first (`extract_panels_from_canvas`). A fifth loader, `EigerRESoNeT`, builds the same Eiger layout from the CrystFEL file `src/preprocessing/data/eiger_resonet.geom`. Detector type always comes from CXI metadata; `get_geometry(detector_desc)` accepts the descriptions `AGIPD 1M`, `ePix10k 2.2M`, `EIGER 4M` and `Jungfrau 4M`. Shapes were measured on the production files under `Resonet/production/<detector>_20k` and by assembling a zero frame through each loader.

**LODO evaluation** holds out one detector at test time and trains on the other three. This tests whether learned features are genuinely detector-agnostic rather than encoding detector-specific artifacts.

Sessions (CXI files, one per beamtime run chunk) are the unit of splitting: `build_sessions()` in `src/training/lodo.py` globs `compressed*.cxi` under each `lodo.detector_dirs` entry and names a session `<detector>_<file stem>`. A deterministic 80/10/10 session-stratified split (greedy by frame count; the `seed` argument has no effect) is applied within the training detectors to produce train / val / in-domain-test subsets. The held-out detector forms a separate cross-detector test split that the model never sees during training or hyperparameter selection.

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
| **Augmentation, geometric** *(train only, before LCN)* | `(224, 224)` crop | Rotated/flipped crop | `random_rot90` (uniform), `random_flip` (50% each axis) |
| **Augmentation, cutout** *(train only, after LCN)* | `(224, 224)` LCN'd crop | Crop with holes | `random_cutout`: 3 holes of 24×24, exact 0 in LCN space, 8 px margin from every centroid, hole skipped after 20 failed draws |

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

Cutout augmentation (3 holes, 24×24 px — `CUTOUT_HOLE_SIZE_DEFAULT` in `src/preprocessing/augment.py`) is applied after LCN and respects a peak-protection zone of 8 px around each Bragg centroid. This prevents cutout from masking the very signal that defines a hit patch, while still disrupting background structure to improve robustness. (The `n_cutout_holes` / `cutout_hole_size` keys in some `configs/supervised/*.yaml` files are not read by any code; the defaults above apply.)

### Key Files

| Role | File |
|------|------|
| Dataset | `src/data/dataset.py` → `AsymmetricCXIDataset` |
| Frame cache | `src/data/frame_cache.py` → `FrameCache`, `frame_cache_from_cfg`, `verify_cache_or_raise`; built by `scripts/build_frame_cache.py` |
| DataLoader | `src/data/dataloader.py` → `asymmetric_loader()` |
| Model | `src/models/supervised.py` → `build_supervised_model()` |
| Training entry point | `src/training/train_asymmetric.py` |
| LODO fold loop | `src/training/lodo.py` → `build_sessions()`, `_train_fold()` (shared with Track 2) |
| Run naming / resume gate | `src/training/run_naming.py` |
| Override once per SLURM job | `src/training/override_marker.py` |
| W&B run identity | `src/training/wandb_identity.py` |
| Result files / inference-only | `src/training/inference_results.py` |
| Result aggregation | `scripts/aggregate_lodo_results.py` |
| Config | `configs/supervised/resnet18_asymmetric.yaml` |

---

## Track 2 — SSL Pipeline (MAE)

Track 2 adds a self-supervised pretraining stage to initialize representations from the large corpus of unlabeled CXI frames before supervised fine-tuning (or a linear probe) on asymmetric patches. The backbone is a ViT, not a ResNet — the architecture mismatch with Track 1 is intentional. The preprocessing path is the shared pipeline.

**Stage 1 — MAE Pretraining** (`src/training/train_ssl_pretrain.py`, `configs/ssl/mae_pretrain.yaml`)

- **Architecture:** ViT-S/16 (`mae_vit_small_patch16`: embed dim 384, depth 12, 6 heads) with a 4-block, 256-dim decoder; 16×16 patch tokens → 196 patches per 224×224 crop. Built by `build_mae_model(cfg)` in `src/models/ssl.py`.
- **Objective:** reconstruct randomly masked patches (`ssl.mask_ratio: 0.6`, lower than the MAE-standard 0.75; `norm_pix_loss: false` because LCN already locally normalizes). A `peak_aware` masking mode exists (`ssl.masking`) but `random` is the configured default.
- **Training:** 400 epochs (20 warm-up), batch 256, lr 1.5e-4, weight decay 0.05, grad clip 1.0; `ssl.crops_per_frame: 4`; checkpoints (with reconstruction sample images) every 20 epochs.
- **Data:** `SSLPretrainCXIDataset` / `ssl_crop_loader` (`src/data/dataset.py`, `dataloader.py`) draw several random valid-region crops per cached frame (`min_valid_frac: 0.5`); no labels are used. No cutout — MAE masking replaces it. Pretraining is run per LODO fold on the training detectors only, so a held-out detector is never seen.
- **Entry point:** `python -u -m src.training.train_ssl_pretrain --config configs/ssl/mae_pretrain.yaml --fold N --run-name-prefix <prefix>`

**Stage 2 — Supervised Fine-tuning or Linear Probe** (`src/training/train_ssl_finetune.py`, `configs/ssl/mae_finetune.yaml`)

- Replace the MAE decoder with a 2-class classification head on the encoder (`pretrain_checkpoint` points at the Stage-1 `last.pt`).
- **Fine-tune:** all weights trainable; 100 epochs, lr 5e-5, early-stopping patience 10, same asymmetric patch pipeline and vote-aggregated frame-level evaluation as Track 1.
- **Linear probe** (`--linear-probe`): encoder frozen, only the head trains. Result directories are named `<prefix>-probe-…` instead of `<prefix>-finetune-…` (`expand_finetune_prefix` in `run_naming.py`).
- `--inference-only` re-evaluates a finished run from its `best.pt` without training (see [Evaluation: LODO Protocol](#evaluation-lodo-protocol)).

**Orchestration:** `scripts/submit_ssl_pretrain_all.sh` and `scripts/submit_ssl_finetune_all.sh` stage the CXI files on local NVMe, train each fold, then clean up (`stage_ssl_data.sh`, `cleanup_ssl_stage.sh`).

> **Status (2026-10-22):** The September 2026 generation completed fine-tune on 4/4 folds (mean cross AP 0.836) and linear probe on 3/4 folds (0.829; the fold-2 probe was cancelled). A rerun under the current run-naming standard (`mae-vits16-v2`) is in progress. Results and caveats → [detailed_notes.md](detailed_notes.md).

---

## Hitfinder Backends

The hitfinder is called during training to label each crop. It is abstracted behind a protocol interface so backends are swappable without changing training code:

```python
class Hitfinder(Protocol):
    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """(H, W) float32 raw frame → (N_peaks, 2) [x, y] centroids"""
```

Backends are selected by `hitfinder.backend` in the config through `get_hitfinder(cfg)` (`src/hitfinders/__init__.py`):

| `backend` | Module | Implementation | Multiprocessing | Use case |
|-----------|--------|----------------|-----------------|----------|
| `pf8` | `pf8.py` | C ctypes bridge (`_pf8_wrap.so`, built by the `Makefile` in `src/hitfinders/`) | Fork-safe ✓ | Production training |
| `pf8_numpy` | `numpy_pf8.py` | Pure NumPy PF8 | Fork-safe ✓ | Tests, no C build |
| `pf8_python` | `pf8_python.py` | ssc Cython PF8 bridge | Fork-safe ✓ | Alternative CPU path |
| `gpu` | `gpu.py` (+ `gpu_pf8.py`, pyFAI `OCL_PeakFinder`) | Delegates to a user GPU script (`gpu_script_path`, `gpu_device`) | CUDA context is not fork-safe → `num_workers=0` enforced | GPU-constrained runs |
| `mock` | `base.py` → `MockHitfinder` | Returns fixed peaks | Fork-safe ✓ | Unit tests |

The PF8 backend is the production default. Its parameters (`pf8_threshold: 800`, `pf8_min_snr: 5.0`, `pf8_min_pix_count: 2`, `pf8_max_pix_count: 200`, `pf8_local_bg_radius: 3`, …) are set in the YAML `hitfinder:` block.

---

## Inference: Vote Aggregation

At inference time no hitfinder is needed. The full assembled, GCN-normalized frame is tiled into a non-overlapping 224×224 grid (stride=224), each patch is independently classified, and the results are aggregated to a frame-level score:

```
frame_score = count(patches where softmax[:, 1] > 0.5) / total_patches
```

A frame is a hit if `frame_score > threshold`, where the optimal threshold is determined on the validation set. Patches are processed in mini-batches of 64 by `run_patch_agg()` in `src/evaluation/benchmark.py`. `aggregation` accepts `"vote"` (default) or `"max"` (kept for backward compatibility); `min_hit_patches` is accepted but reserved and does not change the score.

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

### Run lifecycle

- **Naming:** every run takes a required, versioned `--run-name-prefix`; the checkpoint directory is `checkpoints/<prefix>-fold<N>-seed<S>/`. An existing `best.pt` under the resolved name stops the run unless `--resume-training` or `--override-training` is given (`src/training/run_naming.py`).
- **Override once per SLURM job:** after an override deletes the old checkpoint, `override_applied.json` records the job id and restart count (`override_marker.py`), so a requeued job resumes instead of overriding again and destroying the new attempt's checkpoint.
- **W&B identity:** each override attempt logs to a new run id (`<run>`, `<run>-o2`, …) stored in `<run_dir>/wandb_id.txt`; the superseded run is tagged `overridden` (`wandb_identity.py`).
- **Results:** each fold writes `results.json` with `fold_id`, `test_detector`, `inference_threshold` and `cross` / `in_domain` blocks (`ap`, `auc_roc`, `f1`, `threshold`); newer runs also carry a `training` block (e.g. `configured_epochs`, `last_epoch`, `best_epoch`, `early_stopped`).
- **Inference-only:** `--inference-only` (Track 1 and SSL fine-tune/probe) re-evaluates `best.pt` without training. It writes `results.json` if none exists, otherwise `results.inference.json`, and logs `inference/*` summaries to W&B (`inference_results.py`).
- **Aggregation:** `python scripts/aggregate_lodo_results.py` collects finished folds into per-variant tables.

---

## Dashboard

`dashboard/` is a local, self-contained repository dashboard (TypeScript: a Hono API plus a React/Vite front end). It shows the latest pull request, open issues, the Phase-5 kanban (synced with GitHub issues labelled `task`), CI and local test status, and a runs-and-checkpoints table (its Epochs column reads the `training` block of `results.json`). It also hosts worker agents with approval cards, a chief-of-staff chat and a terminal. It runs on Sol, binds to `127.0.0.1` only, and is reached through an SSH tunnel. Nothing in `src/` imports it and it is not needed for training or evaluation; see [dashboard/README.md](../dashboard/README.md).
