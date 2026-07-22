# SFX Hitfinder — Progress Notes (Track 1: Supervised Baseline)

*Last updated: 2026-06-30. These notes cover Phase 1–4 of the project roadmap.*

---

## 1. Objective

We aim to train a **detector-agnostic hit classifier** for Serial Femtosecond X-ray Crystallography (SFX) diffraction images. The classifier must distinguish hit frames (frames containing a crystal diffraction pattern with Bragg spots) from non-hit frames (blank or background-only frames), and must generalize across four detector types without per-detector retraining:

| Detector | Facility | Raw dimensions |
|---|---|---|
| AGIPD | EuXFEL | 16 × 512 × 128 px (16 modules) |
| JUNGFRAU 4M | LCLS CXI | 2164 × 2068 px (pre-assembled) |
| ePix10k | LCLS | 5632 × 384 px (panels stacked) |
| Eiger4M | Synchrotron/SSX | 5632 × 384 px (panels stacked) |

The project follows two parallel tracks:
- **Track 1 (current):** Supervised fine-tuning of ResNet18/50 on labeled hit/non-hit frames
- **Track 2 (planned):** Self-supervised MAE-style pretraining on unlabeled frames, followed by supervised fine-tuning

The comparison between Track 1 and Track 2 is itself a scientific contribution — quantifying how much labeled data is worth relative to large-scale unlabeled pretraining in the XFEL domain.

---

## 2. Preprocessing Pipeline (Phases 1–4)

All frames pass through an identical pipeline regardless of track, ensuring fair comparison:

```
1. Read CXI/HDF5 frame     →  raw detector array
2. Geometry assembly        →  PADAssembler → native-resolution 2D spatial image
3. Hitfinder (on-the-fly)  →  Bragg spot centroids; hit/non-hit label derived per crop
4. Crop to 224 × 224       →  centre crop (eval) or hitfinder-guided random crop (training)
5. Augmentation (train)     →  random rot90 → random flip → random cutout
6. GCN                     →  I_gcn = (I − μ) / (σ + ε)   [global contrast normalisation]
7. LCN                     →  I_lcn(x,y) = (I(x,y) − μ_W(x,y)) / (σ_W(x,y) + ε)   [local]
```

**LCN window selection (Phase 3 ablation):** Window sizes 3, 9, 15, and 31 were evaluated across all four detector types. Window 31 caused visible panel-edge ringing artifacts. Windows 3, 9, and 15 produced equivalent outputs on non-hit frames. Window **9** was selected as the smallest safe choice that avoids artifacts — this is now the fixed default (`LCN_WINDOW_DEFAULT = 9`).

**Critical ordering constraint (Phase 4 update):** GCN → LCN are applied after the 224×224 crop and all augmentations. There is no resize step — 224×224 is achieved via crop only. Cutout augmentation happens before normalization so GCN/LCN statistics reflect the masked patch.

---

## 3. Model Architecture (Track 1)

- **Backbone:** ResNet18 (primary), ResNet50 (secondary comparison planned)
- **Weights:** ImageNet pretrained, loaded via `timm` / Hugging Face Hub
- **Head:** 2-class linear classifier (`num_classes=2`) replacing the ImageNet head
- **Loss:** CrossEntropyLoss
- **Optimizer:** AdamW
- **Input:** Single-channel 224 × 224 float32 tensor (grayscale diffraction image)
- **Output:** Softmax probability over {non-hit, hit}; positive class is index 1

All hyperparameters are version-controlled in YAML configs under `configs/supervised/`. No hardcoded values in training scripts.

---

## 4. Data

### Source
Resonet production dataset: 80,000 labeled frames across 4 detector types, stored as multi-frame CXI files with embedded hit/non-hit labels (`entry_1/labels/hit`).

```
AGIPD:       5 files × 4,000 frames = 20,000 frames   (EuXFEL)
JUNGFRAU 4M: 10 files × 2,000 frames = 20,000 frames  (LCLS CXI)
ePix10k:     5 files × 4,000 frames = 20,000 frames   (LCLS)
Eiger4M:     5 files × 4,000 frames = 20,000 frames   (Synchrotron/SSX)
Total:       25 sessions, 80,000 frames
```

### Session granularity
One CXI file = one session. Sessions are the atomic unit for train/val/test splitting. This ensures all frames from the same beamtime session stay on the same side of every split — preventing temporal or environmental leakage between train and test sets.

---

## 5. Evaluation Protocol — Leave-One-Detector-Out (LODO)

### Design
LODO is the primary cross-detector generalization benchmark. The experiment runs 4 folds; in each fold one detector type is completely held out from training:

```
Fold 1: train on JUNGFRAU_4M + ePix10k + Eiger4M  →  test on AGIPD
Fold 2: train on AGIPD + ePix10k + Eiger4M         →  test on JUNGFRAU_4M
Fold 3: train on AGIPD + JUNGFRAU_4M + Eiger4M     →  test on ePix10k
Fold 4: train on AGIPD + JUNGFRAU_4M + ePix10k     →  test on Eiger4M
```

Within each fold, sessions are split into four buckets using stratified sampling:

| Split | Source | Purpose |
|---|---|---|
| train | 3 training detectors | Gradient updates |
| val | 3 training detectors (held-out slice) | Early stopping on F1 |
| in_domain_test | 3 training detectors (held-out slice) | Sanity check — should be high |
| cross_detector | 100% of held-out detector | True generalization score |

Early stopping patience = 10 epochs (no val F1 improvement).

### Metrics
- **Average Precision (AP):** Primary metric — area under the precision-recall curve
- **AUC-ROC:** Secondary metric
- **F1 at optimal threshold:** Operational metric for deployment decisions
- **Cross-detector AP** is the headline result; in-domain AP is a sanity check

### Result storage
Each fold saves `checkpoints/<run_name>/results.json` on completion. The aggregation script `scripts/aggregate_lodo_results.py` computes mean ± std AP across all completed folds.

---

## 6. Current Results — All 4 Folds Complete

*As of 2026-06-27. All folds trained and evaluated. Production data: `/data/bioxfel/user/gihan/Resonet/production/` (per-detector subdirs, 5 files × 4,000 frames each for AGIPD/ePix10k/Eiger4M; 10 files × 2,000 frames for JUNGFRAU_4M). Geometry assembly fix (section 8b) was applied before these runs.*

| Fold | Held-out detector | Cross AP | Cross AUC | Cross F1 | In-domain AP |
|---|---|---|---|---|---|
| 1 | AGIPD | 0.5649 | 0.5904 | 0.6661 | 1.0000 |
| 2 | JUNGFRAU_4M | 0.8683 | 0.8156 | 0.7816 | 0.9999 |
| 3 | ePix10k | 0.8825 | 0.8886 | 0.8092 | 1.0000 |
| 4 | Eiger4M | 0.9310 | 0.9138 | 0.8189 | 1.0000 |
| **Mean** | | **0.8117 ± 0.167** | | | |

**Observations:**
- In-domain AP ≈ 1.0 across all folds confirms the model fits the training-detector distribution perfectly — the capacity and training protocol are not limiting factors.
- Cross-detector AP ranges from 0.565 (AGIPD) to 0.931 (Eiger4M). JUNGFRAU_4M and ePix10k sit in the 0.87–0.88 band.
- **AGIPD is the clear outlier.** A cross AP of 0.565 is barely above random (0.5), suggesting the model trained on the other three detectors fails to generalise to AGIPD. The other three detectors generalise to each other reasonably well.
- The mean AP of 0.81 with a std of 0.17 is dominated by the AGIPD gap — without fold 1 the mean would be ~0.89.

---

## 7. Key Finding — Geometry Assembly Issue

A smoke test reading frame 0 from each detector's CXI file revealed the following raw shapes entering the preprocessing pipeline:

| Detector | Raw shape | Assembly status |
|---|---|---|
| AGIPD | (16, 512, 128) | Unassembled — 16 panels row-stacked by `_to_2d()` |
| JUNGFRAU_4M | (2164, 2068) | Properly assembled 2D image |
| ePix10k | (5632, 384) | Panels pre-stacked in CXI (no geometry) |
| Eiger4M | (5632, 384) | Panels pre-stacked in CXI (no geometry) |

Only JUNGFRAU_4M has spatially correct geometry. The other three detectors produce images where panels are naively concatenated — panel boundaries appear as hard horizontal edges at fixed pixel positions in every frame regardless of hit/non-hit status.

**Scientific concern:** A ResNet18 trained on these images may learn to identify detector type from panel-edge artifacts rather than from Bragg spot features. Under LODO, these detector-specific edge signatures are absent in the held-out detector — leaving the model without the spurious cues it relied on. This is a plausible explanation for the low fold 1 cross-detector AP of 0.5649.

**Fix implemented (PR #16, merged 2026-06-30):** See section 8b. Geometry-aware assembly is now active for all 4 detectors. The full LODO results in section 6 were obtained with geometry-corrected inputs.

---

## 8. Infrastructure Notes

- **Compute:** ASU Sol HPC — 8× NVIDIA A100 (80 GB), SLURM scheduler
- **Experiment tracking:** Weights & Biases, project `sfx-hitfinder`
- **Per-fold wall time:** ~10 hours observed for fold 1
- **Parallel submission:** Each fold submitted as a separate SLURM job via `scripts/submit_lodo_fold.sh` with 14-hour time limit
- **Checkpoint resume:** If a fold crashes after training completes, resubmission detects the existing `best.pt` and skips directly to evaluation

---

## 8b. Geometry Assembly Fix (2026-06-26)

**Problem confirmed by visual inspection:** Running `scripts/visualize_assembled.py --all` showed that AGIPD, ePix10k, and Eiger4M were all producing vertically stacked panel strips rather than spatially correct 2D images. Only Jungfrau 4M (pre-assembled) looked correct.

**Root cause (two bugs):**
1. `assemble_image()` condition `canvas_size != n_pixels` was False for ePix10k and Eiger4M with their CrystFEL geom files → fell through to `concat_data + reshape` (stacking) every time; `PADAssembler` never called.
2. For AGIPD, `PADAssembler` was called but received a list of 2D panels — internal `concat_data` only extracted one module's worth (65k/1M pixels) due to `parent_data_slice` being per-module, not full-detector.

**Fix implemented:**
- **AGIPD + ePix10k** → Reborn standard loaders (`detector.agipd_pad_geometry_list()`, `detector.epix10k_pad_geometry_list()`) + `PADAssembler(frame.ravel())`. Raw array ravel order matches PADAssembler's `flat_indices` exactly.
- **Eiger4M** → CrystFEL geom file (`eiger4m.geom`) + `PADAssembler(concat panel ravels)` via `extract_panels_from_canvas`.
- **Jungfrau 4M** → unchanged (`preprocess_assembled` on the pre-assembled 2164×2068 canvas).

**Verified assembled shapes:**
| Detector | Assembled shape | Visual |
|---|---|---|
| AGIPD | (1273, 1273) | Correct ring/octagonal layout ✅ |
| ePix10k | (1667, 1667) | Correct panel grid around beam centre ✅ |
| Eiger4M | (1687, 1687) | Correct Bragg ring layout ✅ |
| Jungfrau 4M | (2164, 2068) | Unchanged — correct ✅ |

**Files changed (PR #16, merged 2026-06-30):** `src/preprocessing/geometry.py` (`get_geometry()` dispatch, `get_assembler()` module-level cache), `src/preprocessing/pipeline.py` (`preprocess_with_geometry()` uses `PADAssembler` with flat data), `src/data/dataset.py` (`_use_geometry` flag replaces `is` identity check; Reborn objects fetched lazily from cache instead of stored as instance attrs), `scripts/visualize_assembled.py`, `tests/test_geometry_assembly.py` (18 tests, all passing). Additionally 8 code-review findings fixed: `label_key` forwarding in `cxi_session_loader`, `OSError` added to exception handler, pickle safety under spawn workers, fold key validation at startup, backbone/num_classes checkpoint validation, stable `id=run_name` in `wandb.init`, aggregate script bare-open fix.

**Outcome:** All 4 LODO folds completed with geometry-corrected inputs (results in section 6). AGIPD cross AP = 0.565 despite correct assembly — the gap is not explained by panel-edge artifacts alone and warrants further investigation.

---

## 9. What Comes Next

**Immediate (Phase 4 — remaining):**
1. **Investigate AGIPD generalisation gap** — cross AP = 0.565 persists even after geometry-correct assembly. Candidate causes: (a) EuXFEL vs. LCLS domain shift (different photon energies, sample environments); (b) AGIPD 16-module sparse layout produces a fundamentally different image structure than the other three detectors; (c) hit rate or label distribution differs in the AGIPD production data. Next step: inspect assembled AGIPD frames visually and compare hit/non-hit distributions.
2. **Phase 5 starts only when user confirms Phase 4 testing complete.**

**Track 2 (Phase 5):**
- MAE-style self-supervised pretraining on pooled unlabeled XFEL frames
- Attach classification head, fine-tune on labeled data
- LODO evaluation using identical protocol to Track 1
- Compare Track 1 vs Track 2 cross-detector AP as scientific contribution

---

## 10. Asymmetric Pipeline — Design and Implementation (2026-07-02)

### 10.1 Motivation

The LODO baseline (section 6) uses frame-level hit/non-hit labels inherited by all training crops. This is weak supervision: a random 224×224 crop from a hit frame may contain only background diffuse scatter and no Bragg spots. The model receives label=1 for a patch that looks identical to a non-hit background patch — degrading the signal-to-noise of the training signal.

The asymmetric pipeline solves this by routing each assembled frame through a peak-finding algorithm (PeakFinder8 or a GPU hitfinder) before cropping. The crop label is then assigned based on **crop content** rather than frame label:

```
label = 1  if crop contains ≥1 peak centroid
label = 0  if crop is from a non-hit frame (any crop)
label = 0  if crop is from a hit frame but no centroid within 50 px of any crop edge
              (hard negative — background crop from a hit frame)
```

### 10.2 Full Training Pipeline

```
Raw CXI frame
  → Reborn geometry assembly → assembled (H, W) float32
  → Hitfinder → peak centroid map [(x₁,y₁), (x₂,y₂), ...]
  → Sample 224×224 crop (class-balanced, see §10.3)
  → random_rot90 → random_flip → GCN(patch) → LCN(patch) → random_cutout
  → ResNet18
```

Validation and test use the existing blind grid approach:
```
Assembled frame → non-overlapping 224×224 grid → GCN+LCN per patch → ResNet18
  → hit_tile = softmax[:,1] > 0.5
  → frame_score = hit_tile_count / n_tiles  (vote aggregation)
  → frame_label = 1 if frame_score ≥ 3/n_tiles
  → compare with ground-truth label → AP, AUC, F1
```

### 10.3 Class-Balance Sampling

`hit_frac = 0.5` (configurable):

| Branch | Condition | Sampling | Label |
|---|---|---|---|
| Hit crop | `frame_label=1`, `rand < hit_frac` | Random crop until ≥1 centroid inside; fallback to label=0 after `hard_neg_max_attempts=50` | 1 |
| Hard negative | `frame_label=1`, `rand ≥ hit_frac` | Random crop until no centroid within 50 px of any edge | 0 |
| True miss | `frame_label=0` | Any random crop | 0 |

The 50 px margin in the hard-negative branch (`_crop_within_margin`) prevents including Bragg peak halos — the diffuse scatter ring surrounding a peak centroid — in label-0 training patches.

**Practical note on small frames:** On a 512×512 frame with a 224×224 crop, the 50 px margin leaves ≤0 px of unoccupied space if a peak sits near frame centre. The fallback (random crop, label=0) handles this without crashing.

### 10.4 Hitfinder Abstraction

Two pluggable backends, same `find_peaks(assembled) → (N,2) float32` interface:

| Backend | Class | Status | Worker safety |
|---|---|---|---|
| `pf8` | `PF8Hitfinder` | **Stub** — `NotImplementedError`; C++ wrapper interface TBD | Fork-safe, any `num_workers` |
| `gpu` | `GPUHitfinder` | **Stub** — `NotImplementedError`; user script TBD | Requires `num_workers=0` |
| `mock` | `MockHitfinder` | Fully implemented — returns fixed peaks; use for tests/smoke | Fork-safe |

Config selects backend:
```yaml
hitfinder:
  backend: pf8     # "pf8" | "gpu" | "mock"
  pf8_threshold_snr: 5.0
  pf8_min_peaks: 1
```

`train_asymmetric.py` automatically forces `num_workers=0` and prints a warning when `backend: gpu` with `num_workers > 0`.

### 10.5 Vote Aggregation in Validation

Added `aggregation` parameter to `run_patch_agg` in `src/evaluation/benchmark.py`:

- `aggregation="max"` (default, backward-compatible): `frame_score = max(softmax[:,1])` across patches
- `aggregation="vote"`: `frame_score = hit_tile_count / n_tiles` where `hit_tile = softmax[:,1] > 0.5`

Vote aggregation matches the training objective better: the model is trained on individual crops that are each labeled hit or miss, so its scores should be interpreted as per-crop decisions rather than frame-wide confidence values.

All new code (`train_asymmetric.py`, `resnet18_asymmetric.yaml`) defaults to `aggregation="vote"`. Old code (`train_lodo.py`) is unaffected (uses default `"max"`).

### 10.6 Files Added / Modified

| File | Change |
|---|---|
| `src/hitfinders/__init__.py` | New — exports Protocol, stubs, `get_hitfinder` factory |
| `src/hitfinders/base.py` | New — `Hitfinder` Protocol, `MockHitfinder` |
| `src/hitfinders/pf8.py` | New — `PF8Hitfinder` stub |
| `src/hitfinders/gpu.py` | New — `GPUHitfinder` stub |
| `src/data/dataset.py` | Added `AsymmetricCXIDataset`, `_crop_contains_centroid`, `_crop_within_margin(margin=50)`; deprecated `MultiFrameCXIDataset` |
| `src/data/dataloader.py` | Added `asymmetric_loader`; deprecated `cxi_session_loader` |
| `src/evaluation/benchmark.py` | Added `aggregation` param to `run_patch_agg`; validated against `("max","vote")` |
| `configs/supervised/resnet18_asymmetric.yaml` | New — full asymmetric pipeline config |
| `scripts/train_asymmetric.py` | New — training entrypoint (mirrors `train_lodo.py`) |
| `tests/test_hitfinders.py` | New — 9 tests |
| `tests/test_asymmetric_dataset.py` | New — 9 tests |
| `tests/test_vote_aggregation.py` | New — 4 tests |
| `tests/test_train_supervised.py` | Fixed stale `evaluate` import (function was removed) |
| `src/training/train_supervised.py` | Stripped to `_set_seeds` + `train_one_epoch` only |

**Deprecated and deleted scripts** (used old `evaluate()` approach):
- `scripts/train_synthetic_full.py`, `train_synthetic_4epochs.py`, `train_resonet_cxi.py`
- `scripts/evaluate_supervised.py`, `evaluate_resonet_cxi.py`
- `scripts/smoke_test_synthetic.py`, `smoke_test_synthetic_reborn.py`
- `scripts/submit_resonet_train.sh`, `submit_supervised.sh`

### 10.7 Test Status

```
263 passed, 8 skipped  (as of 2026-07-02, branch phase-04-augmentation)
```

All new tests pass. Deprecated symbols (`MultiFrameCXIDataset`, `cxi_session_loader`) still importable — no backward-compat breakage.

### 10.8 What Remains Before Running

1. **PF8 C++ wrapper:** `PF8Hitfinder.find_peaks` is a stub. User needs to provide the call signature for the existing PF8 wrapper on Sol (subprocess? ctypes? Python binding?). Then implement in `src/hitfinders/pf8.py`.
2. **GPU hitfinder:** `GPUHitfinder.find_peaks` is a stub. User to provide GPU hitfinder script for integration.
3. **Smoke test with mock backend:** `python scripts/train_asymmetric.py --config configs/supervised/resnet18_asymmetric.yaml --device cpu --folds 1` after setting `hitfinder.backend: mock` in YAML — confirms the full pipeline runs end-to-end before real hitfinder is wired up.
4. **Full training run:** Once PF8 is wired, run all 4 LODO folds and compare asymmetric pipeline AP vs LODO baseline AP (section 6).
