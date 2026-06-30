# Project Memory — SFX Hitfinder

> Updated: 2026-06-30 | Read this at session start before anything else.

---

## Current Status

| Item | Value |
|------|-------|
| Active branch | `main` |
| Current phase | **Phase 4 — supervised baseline (2 items remaining)** |
| Last merge | PR #16 — geometry assembly + LODO pipeline (2026-06-30) |
| Test suite | 198 passed, 8 skipped (fabio absent + bitshuffle) |

---

## Phase History

### Phase 4 extended — Geometry Assembly & LODO (2026-06-30)

PR #16 merged. Key additions:

- **Geometry-aware assembly**: AGIPD/ePix10k use Reborn std pad loaders + `PADAssembler(frame.ravel())`; Eiger4M uses CrystFEL `.geom` throughout (`DETECTOR_LOADERS` unified, `parent_data_slice` always defined). CrystFEL geom files at `src/preprocessing/data/{agipd,eiger4m,epix10k}.geom`.
- **LODO pipeline**: `scripts/train_lodo.py`, `submit_lodo.sh`, `submit_lodo_fold.sh`, `configs/supervised/resnet18_lodo.yaml`, `scripts/aggregate_lodo_results.py`. Checkpoints on disk at `checkpoints/resnet18-lodo-fold{1-4}-seed42/`.
- **8 code-review fixes**: label_key forwarding in `cxi_session_loader`, geometry routing uses `_use_geometry` flag not `is` identity, `OSError` added to exception handler, Reborn objects removed as instance attrs (module-level cache instead), fold key validation at startup, backbone+num_classes saved/validated in checkpoint, `id=run_name` in `wandb.init`, aggregate script uses already-loaded data for missing-fold check.

**LODO 4-fold results (ResNet18, synthetic/Resonet data, 2026-06-27):**

| Fold | Held-out | Cross AP | Cross AUC | Cross F1 |
|------|----------|----------|-----------|----------|
| 1 | AGIPD | 0.565 | 0.590 | 0.666 |
| 2 | JUNGFRAU_4M | 0.868 | 0.816 | 0.782 |
| 3 | ePix10k | 0.883 | 0.889 | 0.809 |
| 4 | Eiger4M | 0.931 | 0.914 | 0.819 |
| **Mean** | | **0.812 ± 0.167** | | |

In-domain AP = 1.0 for all folds. AGIPD is the clear outlier — likely domain shift from panel structure or assembly differences.

### Phase 4 — Resonet CXI Integration (2026-06-11)

`MultiFrameCXIDataset`, `preprocess_assembled()` geometry-bypass, `eiger_resonet_pad_geometry_list()`, `extract_panels_from_canvas()`. Training script `scripts/train_resonet_cxi.py`, evaluation `scripts/evaluate_resonet_cxi.py`. Production data at `/data/bioxfel/user/gihan/Resonet/production/` (per-detector subdirs, used by LODO config).

### Phase 4 — Supervised Baseline (complete, 2026-05-28)

Full ResNet18/50 supervised track: `SFXDataset`, `load_config()`, `build_supervised_model()` via timm, training loop with AdamW + CrossEntropyLoss, wandb logging.

**Synthetic baseline (2026-06-05):** `resnet18-10k-full-seed42`, 10k frames, early stop epoch 22. Held-out eval (2000 frames): AP=0.9998, AUC=0.9998, F1=0.9995, Precision=1.0, Recall=0.999.

Label encoding: `labels[:, -1]` is `bg_only`; `1.0` → non-hit (class 0), `0.0` → hit (class 1). Images pre-assembled 512×512 uint16; Reborn geometry step skipped.

### Phase 3 — Preprocessing (complete, 2026-05-23)

Full pipeline: Reborn geometry → GCN → LCN (window=9) → resize 224×224. LCN window=9 confirmed via ablation. JUNGFRAU_4M uses CrystFEL `.geom`, not Reborn built-in. HDF5 keys confirmed for all detectors (see `docs/data_spec.md`).

---

## Known Gotchas

| # | Gotcha | Impact |
|---|--------|--------|
| 1 | AGIPD CXI file replaced mid-Phase 3 — old key `entry_1/data_1/data` (N×5632×384) no longer valid | `io.py` fallback list updated; old files silently match wrong shape |
| 2 | JUNGFRAU_4M Reborn built-in loader expects raw panels; actual HDF5 is pre-assembled canvas with gap pixels | Always use `jungfrau4m_crystfel_pad_geometry_list()` |
| 3 | Eiger4M real-data tests raise `OSError: Can't find plugin` (bitshuffle HDF5 filter) | Needs `hdf5plugin` import before `h5py.File()`. CI skips (file absent). |
| 4 | GitHub ruleset `protect_main` required check name is `"test"` not `"CI / test"` | Fixed 2026-05-23. Check repo Settings → Rules if CI gating breaks. |
| 5 | `notebooks/lcn_ablation_executed.ipynb` is untracked — intentional executed artifact | Do not commit. |
| 6 | `anaconda_projects/` directory is untracked | Status unresolved — determine if active or legacy before Phase 5. |
| 7 | Resonet geometry file is named `Eigar.geom` (typo) | Use `src/preprocessing/data/eiger_resonet.geom` — not the original. |
| 8 | Resonet production data root | `/data/bioxfel/user/gihan/Resonet/production/` — per-detector subdirs (agipd_20k, jungfrau_20k, epix10k_20k, eiger4m_20k) |
| 9 | `geometry_file_to_pad_geometry_list()` is in `reborn.external.crystfel` | Always import from `reborn.external.crystfel`, not `reborn.detector`. |
| 10 | `gh pr edit` is broken by GitHub Projects Classic deprecation warning (exit 1) | Use `gh api repos/<owner>/<repo>/pulls/<N> -X PATCH -f body="..."` instead. |

---

## Immediate Next Steps

1. **Investigate AGIPD gap** — cross AP=0.565 is 30+ points below other detectors; likely causes: panel structure (16×512×128 raw), assembly path differences, or data distribution shift
2. **Real-detector LODO baseline** — evaluate model generalisation on actual facility data
3. **Phase 5 starts only when user confirms Phase 4 testing complete**

---

## Open Decisions

| Decision | Status | Notes |
|----------|--------|-------|
| ViT variant (Base vs. Small) | Open | Decide at Phase 5 start |
| nanoBragg synthetic data | Deferred | Revisit if augmentation needed in Phase 6 |
| AGIPD generalisation gap | Under investigation | Cross AP=0.565 — root cause unknown |
