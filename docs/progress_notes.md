# SFX Hitfinder — Progress Notes (Track 1: Supervised Baseline)

*Last updated: 2026-06-26. These notes cover Phase 1–4 of the project roadmap.*

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

## 2. Preprocessing Pipeline (Phases 1–3)

All frames pass through an identical pipeline regardless of track, ensuring fair comparison:

```
1. Read CXI/HDF5 frame  →  raw detector array
2. Flatten to 2D        →  (_to_2d: row-stack if multi-panel, reshape if 1D)
3. GCN                  →  I_gcn = (I − μ) / (σ + ε)   [global contrast normalisation]
4. LCN                  →  I_lcn(x,y) = (I(x,y) − μ_W(x,y)) / (σ_W(x,y) + ε)   [local]
5. Resize to 224 × 224  →  anti-aliased, preserve_range=True
```

**LCN window selection (Phase 3 ablation):** Window sizes 3, 9, 15, and 31 were evaluated across all four detector types. Window 31 caused visible panel-edge ringing artifacts. Windows 3, 9, and 15 produced equivalent outputs on non-hit frames. Window **9** was selected as the smallest safe choice that avoids artifacts — this is now the fixed default (`LCN_WINDOW_DEFAULT = 9`).

**Critical ordering constraint:** Normalization (GCN → LCN) always precedes resize. Resize is solely for model input compatibility, not detector correction.

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

## 6. Current Results (Fold 1 Only — Folds 2–4 In Progress)

*As of 2026-06-26. Folds 2–4 are currently training.*

| Fold | Held-out detector | Cross AP | Cross AUC | Cross F1 | In-domain AP |
|---|---|---|---|---|---|
| 1 | AGIPD | 0.5649 | 0.5904 | 0.6661 | 1.0000 |
| 2 | JUNGFRAU_4M | pending | — | — | — |
| 3 | ePix10k | pending | — | — | — |
| 4 | Eiger4M | pending | — | — | — |

**Observation:** In-domain AP = 1.0000 confirms the model fits the training detectors perfectly. Cross-detector AP = 0.5649 for AGIPD (barely above random chance of 0.5) suggests the model is not generalizing to the held-out detector.

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

**Planned fix:** Replace `preprocess_assembled` in `MultiFrameCXIDataset` with geometry-aware assembly using Reborn for all 4 detectors. This requires geometry files for AGIPD, ePix10k, and Eiger4M (JUNGFRAU geometry already exists at `src/preprocessing/data/jungfrau4m_jf4m_103mm.json`). The user is preparing a Reborn geometry reference document.

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

**Files changed:** `src/preprocessing/geometry.py` (`get_geometry()` dispatch, `get_assembler()` cache), `src/preprocessing/pipeline.py` (`preprocess_with_geometry()` uses `PADAssembler` with flat data), `src/data/dataset.py` (pre-builds assembler cache), `scripts/visualize_assembled.py` (updated assembly paths), `tests/test_geometry_assembly.py` (18 tests, all passing).

**Next step:** Re-run all 4 LODO folds with geometry-corrected inputs and compare cross-detector AP against the 0.5649 fold-1 baseline.

---

## 9. What Comes Next

**Immediate (before manuscript):**
1. Wait for folds 2–4 to complete → get full LODO mean ± std AP
2. Implement proper Reborn geometry assembly for all 4 detectors
3. Re-run LODO with geometry-corrected inputs → compare cross-detector AP before/after
4. Determine whether geometry correction closes the generalization gap

**Track 2 (Phase 5):**
- MAE-style self-supervised pretraining on pooled unlabeled XFEL frames
- Attach classification head, fine-tune on labeled data
- LODO evaluation using identical protocol to Track 1
- Compare Track 1 vs Track 2 cross-detector AP as scientific contribution
