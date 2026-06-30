# PLANNING.md — Detector-Agnostic SFX Hitfinder

## Current Phase Checklist

- [x] Finalize NSF-style proposal draft
- [x] Formalize cross-detector benchmark protocol (leave-one-detector-out)
- [x] Define synthetic data generation strategy
- [x] Verify Reborn handles all four target detector types
- [x] Confirm Sol HPC CUDA version and validate environment

~~Move to Phase 2 only when all five are checked.~~ ✅ All complete — Phase 2 active.

> **Note:** Synthetic data strategy resolved — real `.img` diffraction images available as unlabeled SSL pretraining data. nanoBragg/augmentation deferred to Phase 4.

---

## Phase 2 Checklist

- [x] Unified image reader (`src/preprocessing/io.py`) — `.img` via fabio, `.h5`/`.cxi` via h5py
- [x] `UnlabeledDataset` for SSL pretraining (`.img` files, no labels)
- [x] `SFXDataset` scaffold for supervised training (label loading deferred)
- [x] DataLoader factories (`ssl_pretrain_loader`, `supervised_loader`)
- [x] GitHub Actions CI — formatting check + full pytest suite on every push/PR
- [x] Resolve label format: JSON sidecar (`labels.json`, absolute path → 0/1)

~~Move to Phase 3 only when all six are checked.~~ ✅ All complete — Phase 3 next.

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Proposal & methodology finalization | COMPLETE |
| 2 | Data infrastructure (real + synthetic) | COMPLETE |
| 3 | Preprocessing implementation | COMPLETE |
| **4** | **Supervised baseline (ResNet18 → ResNet50)** | **CURRENT** |
| 5 | SSL model (MAE pretraining → fine-tune) | Pending |
| 6 | Ablations & cross-detector benchmarking | Pending |
| 7 | Deployment preparation | Future |
| 8 | Thesis writing | Future |

> **2026-06-30:** PR #16 merged — geometry-aware assembly for all 4 detectors, full LODO training pipeline, 4-fold LODO evaluation complete, 8 code-review findings fixed. Two Phase 4 items remain: Resonet CXI evaluation and real-detector baseline.

---

## Phase 3 Checklist ✅ (complete, merged 2026-05-23)

- [x] `normalize.py` — GCN and LCN implementations with ε-guarded denominators
- [x] `pipeline.py` — full pipeline: assemble → `_to_2d` → GCN → LCN → resize 224×224
- [x] `tests/test_normalize.py` — unit tests for GCN and LCN (including order enforcement)
- [x] `tests/test_pipeline.py` — end-to-end tests for all four detectors
- [x] Ablate LCN window size — window=9 confirmed (window=31 shows panel-edge artifacts; 3/9/15 equivalent on non-hit frames; 9 is smallest safe choice)
- [x] Confirm HDF5 key `entry/data/data` against real detector files — see `docs/data_spec.md`

## Phase 4 Checklist 🔄 (current — in progress)

- [x] `src/data/dataset.py` — complete `SFXDataset` (labeled HDF5, lazy-load per `__getitem__`)
- [x] `configs/supervised/resnet18.yaml` — learning rate, batch size, weight decay, seed
- [x] `src/utils/config.py` — `load_config()` YAML deep-merge utility
- [x] `src/models/supervised.py` — ResNet18 fine-tune with `timm` pretrained weights
- [x] `src/training/train_supervised.py` — training loop with wandb logging
- [x] `tests/test_dataset.py` — unit tests for `SFXDataset` (lazy-load, label lookup, splits)
- [x] `tests/test_config.py` — unit tests for config loader
- [x] `tests/test_models.py` — unit tests for supervised model forward pass
- [x] `tests/test_train_supervised.py` — smoke tests for training loop
- [x] Synthetic baseline run: `resnet18-10k-full-seed42` — 10k frames, early stop epoch 22, best val F1=1.0000 (2026-06-05)
- [x] `scripts/evaluate_supervised.py` — evaluate checkpoint on held-out test set; reports AP/AUC/F1/confusion matrix
- [x] Held-out evaluation on `hitfinder_val` (2000 frames): AP=0.9998, AUC=0.9998, F1=0.9995, Precision=1.0, Recall=0.999 (2026-06-05)
- [x] `src/data/dataset.py` — `MultiFrameCXIDataset` for multi-frame CXI files with embedded labels (`entry_1/labels/hit`)
- [x] `src/preprocessing/geometry.py` — `eiger_resonet_pad_geometry_list()`, `extract_panels_from_canvas()`; EigerRESoNeT added to `DETECTOR_LOADERS`
- [x] `src/preprocessing/pipeline.py` — `preprocess_assembled()` geometry-bypass path for already-assembled detectors (Eiger/Resonet)
- [x] `src/preprocessing/io.py` — `count_frames()`, `read_frame()`, `read_embedded_labels()` for CXI multi-frame files
- [x] `scripts/train_resonet_cxi.py` — training script for Resonet CXI data with 70/20/10 split and test evaluation
- [x] `scripts/evaluate_resonet_cxi.py` — inference + metrics script for Resonet CXI files
- [x] `scripts/submit_resonet_train.sh` — SLURM job script for Resonet CXI training
- [x] `configs/supervised/resnet18_resonet.yaml` — config for Resonet CXI training (early_stopping_patience=10, test_fraction=0.1)
- [x] Resonet CXI domain investigation: confirmed `cxi_merged_25k.cxi` and `cxi_1k/` are in same intensity regime; `cxi_100/` is anomalous low-intensity distribution (2026-06-12)
- [x] Geometry-aware assembly for all 4 detectors: AGIPD/ePix10k via Reborn std pads + PADAssembler, Eiger4M via CrystFEL `.geom` throughout — PR #16 merged 2026-06-30
- [x] CrystFEL `.geom` files added: `src/preprocessing/data/agipd.geom`, `eiger4m.geom`, `epix10k.geom`
- [x] `scripts/train_lodo.py` — LODO training script with session-level split, wandb logging, per-fold checkpoint + results.json
- [x] `scripts/submit_lodo.sh`, `scripts/submit_lodo_fold.sh` — SLURM submission (fold-level log names fixed)
- [x] `configs/supervised/resnet18_lodo.yaml` — LODO config
- [x] `scripts/aggregate_lodo_results.py` — aggregate per-fold results.json into summary table
- [x] LODO 4-fold evaluation complete (2026-06-27): mean cross AP=0.812 ± 0.167; AGIPD held-out AP=0.565 is outlier; checkpoints on disk under `checkpoints/`
- [x] 8 code-review findings fixed: label_key forwarding, geometry routing identity check, OSError handling, pickle safety, fold key validation, checkpoint config validation, wandb stable run ID, aggregate script bare open()
- [ ] Resonet CXI training run: evaluate `resnet18-resonet-seed42` checkpoint (SLURM job 55337893 completed 2026-06-12) on held-out test set
- [ ] Real-detector LODO baseline: investigate AGIPD generalisation gap (cross AP=0.565 vs 0.868–0.931 for other detectors)

> **Note:** Geometry assembly and LODO pipeline merged (PR #16, 2026-06-30). Remaining Phase 4 work: Resonet CXI evaluation and AGIPD gap investigation. Phase 5 (SSL) does not begin until user confirms Phase 4 testing complete.

---

## Phase 5 Checklist

- [ ] ViT variant decision: ViT-Base vs. ViT-Small (decide before writing any model code)
- [ ] `src/models/ssl.py` — MAE encoder + classification head
- [ ] `src/training/train_ssl_pretrain.py` — masked image pretraining loop
- [ ] `src/training/train_ssl_finetune.py` — fine-tune SSL encoder with classification head
- [ ] `configs/ssl/mae_pretrain.yaml` — pretraining hyperparameters
- [ ] `configs/ssl/mae_finetune.yaml` — fine-tuning hyperparameters
- [ ] `tests/test_ssl_model.py` — unit tests for MAE encoder and classification head
- [ ] `tests/test_train_ssl.py` — smoke tests for pretraining and fine-tuning loops
- [ ] Pretraining run on Sol HPC (unlabeled `.img` files)
- [ ] Fine-tuning run on Sol HPC (labeled HDF5 splits)

Move to Phase 6 only when all ten are checked.

---

## Open Decisions

- [x] Synthetic data generation tool: real `.img` images used for SSL pretraining; nanoBragg deferred to Phase 4
- [x] Formal definition of "detector-agnostic" as a measurable evaluation criterion
- [x] **Label format and storage convention for HDF5/CXI files** — JSON sidecar (`labels.json`)
- [ ] ViT variant for SSL track: ViT-Base vs. ViT-Small (latency vs. capacity tradeoff)
- [x] LCN window size parameter — window=9 confirmed (Phase 3 ablation)

---

## Known Risks

| ID | Risk | When It Bites |
|----|------|--------------|
| B3 | MAE reconstruction may fail on sparse diffraction images | Phase 5 |
| B1 | 224×224 resize destroys sub-pixel Bragg peaks | Phase 3–4 |
| A4 | Synthetic data too clean to reflect real facility data | Phase 4 |
| C1 | Reborn coverage gaps for specific detector configurations | Phase 3 |
| D1 | Multi-detector real data access not guaranteed | Phase 2 — partially mitigated (.img files available) |

Full risk register: `PhD_Project_Gaps_and_Pitfalls.md`

---

## Literature Knowledge Base

Located in `/mnt/project/` (read-only). Key files:

| File | Purpose |
|------|---------|
| `00_literature_master_summary.md` | Top 10 papers, readiness by theme, gaps to fill |
| `01_literature_inventory.csv` | Full 34-paper inventory with tags |
| `02_thematic_synthesis.md` | 8-theme synthesis of the literature |
| `03_gap_analysis.md` | Gaps by theme; what's covered vs. missing |
| `04_literature_review_draft.md` | NSF-style literature review draft (~1700 words) |

**Priority acquisitions not yet in collection:** Reborn library paper, EfficientNet (Tan & Le 2019), DINOv2 (Oquab et al. 2023).
