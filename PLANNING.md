# PLANNING.md — Detector-Agnostic SFX Hitfinder

## Current Phase Checklist

- [x] Finalize NSF-style proposal draft
- [x] Formalize cross-detector benchmark protocol (leave-one-detector-out)
- [ ] Define synthetic data generation strategy
- [x] Verify Reborn handles all four target detector types
- [ ] Confirm Sol HPC CUDA version and validate environment

Move to Phase 2 only when all five are checked.

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Proposal & methodology finalization | CURRENT |
| 2 | Data infrastructure (real + synthetic) | Pending |
| 3 | Preprocessing implementation | Pending |
| 4 | Supervised baseline (ResNet18 → ResNet50) | Pending |
| 5 | SSL model (MAE pretraining → fine-tune) | Pending |
| 6 | Ablations & cross-detector benchmarking | Pending |
| 7 | Deployment preparation | Future |
| 8 | Thesis writing | Future |

---

## Open Decisions (resolve before Phase 2)

- [ ] Synthetic data generation tool: nanoBragg vs. simulated augmentation vs. other
- [x] Formal definition of "detector-agnostic" as a measurable evaluation criterion
- [ ] ViT variant for SSL track: ViT-Base vs. ViT-Small (latency vs. capacity tradeoff)
- [ ] Label format and storage convention for HDF5/CXI files
- [ ] LCN window size parameter (needs ablation in Phase 3)

---

## Known Risks

| ID | Risk | When It Bites |
|----|------|--------------|
| B3 | MAE reconstruction may fail on sparse diffraction images | Phase 5 |
| B1 | 224×224 resize destroys sub-pixel Bragg peaks | Phase 3–4 |
| A4 | Synthetic data too clean to reflect real facility data | Phase 4 |
| C1 | Reborn coverage gaps for specific detector configurations | Phase 3 |
| D1 | Multi-detector real data access not guaranteed | Phase 2 |

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
