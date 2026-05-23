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
| 3 | Preprocessing implementation | CURRENT |

## Phase 3 Checklist

- [x] `normalize.py` — GCN and LCN implementations with ε-guarded denominators
- [x] `pipeline.py` — full pipeline: assemble → `_to_2d` → GCN → LCN → resize 224×224
- [x] `tests/test_normalize.py` — unit tests for GCN and LCN (including order enforcement)
- [x] `tests/test_pipeline.py` — end-to-end tests for all four detectors
- [x] Ablate LCN window size — window=9 confirmed (window=31 shows panel-edge artifacts; 3/9/15 equivalent on non-hit frames; 9 is smallest safe choice)
- [x] Confirm HDF5 key `entry/data/data` against real detector files — see `docs/data_spec.md`

Move to Phase 4 only when all six are checked.
| 4 | Supervised baseline (ResNet18 → ResNet50) | Pending |
| 5 | SSL model (MAE pretraining → fine-tune) | Pending |
| 6 | Ablations & cross-detector benchmarking | Pending |
| 7 | Deployment preparation | Future |
| 8 | Thesis writing | Future |

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
