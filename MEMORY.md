# Project Memory — SFX Hitfinder

> Updated: 2026-08-04 | Read this at session start before anything else.

---

## Current Status

| Item | Value |
|------|-------|
| Active branch | `phase-04-augmentation` |
| Current phase | **Phase 4 — augmentation (asymmetric pipeline implemented; inference = patch-grid aggregation)** |
| Last commit | GCN refactor — full-frame GCN before crop/tile in both training and inference paths |
| Test suite | 44 passed (normalize + patch_eval + asymmetric_dataset, 2026-08-04) |

---

## Phase History

### Phase 4 — Augmentation / Asymmetric Pipeline (2026-07-21 → 2026-08-04)

Branch `phase-04-augmentation`. Key deliverables:

- **Hitfinder-guided crop** (`AsymmetricCXIDataset.__getitem__`): Path A (peaks found → crop centred on random Bragg peak, label=1); Path B (no peaks → random crop with 50 px clearance, label=0). Replaces all prior centre-crop and random-crop logic.
- **Full-frame GCN before crop/tile (both paths)** (2026-08-04): `gcn(assembled)` is applied to the full assembled frame immediately after assembly, before padding/crop (training) or before `patch_grid` (inference). LCN is then applied per crop/patch. `gcn_apply` removed from `normalize.py` — no longer needed. Ensures both paths share the same global contrast scale.
- **Patch-grid inference** (`run_patch_agg` in `src/evaluation/benchmark.py`): full assembled frame tiled into 224×224 patches (stride=224), GCN applied to full frame first then LCN per patch, scores aggregated per frame with `vote` (default) or `max`. Centre crop is no longer used at eval time.
- **Inference threshold** saved in checkpoint (`val_threshold` key, 0.5 fallback).
- **CLAUDE.md corrected** (2026-08-04): pipeline step 4 now correctly documents eval as patch-grid tiling, not centre crop.
- Removed: LODO scripts/configs, `random_crop`, `center_crop`, deprecated loaders (all superseded).

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
| 11 | `gcn_apply` was removed from `normalize.py` (2026-08-04) | Do not import or use it — `gcn()` on the full assembled frame before crop/tile replaced it everywhere |
| 12 | LCN changed to variance form `(I−μ_W)/sqrt(σ²_W+ε)` with `LCN_EPSILON=1e-2` (2026-08-17) | Old std-form ε=1e-6 amplified background readout noise to unit variance (salt-and-pepper on JUNGFRAU non-hits). Checkpoints trained before 2026-08-17 (LODO folds, synthetic baseline) used the old form — retrain before comparing against new runs. Ablation in `notebooks/pipeline_debug.ipynb`. |
| 13 | Masked LCN + gap fill + 2 px edge erosion (2026-08-17) | `lcn(..., mask=)` excludes gap/padding/edge pixels from local stats (fixes panel-boundary halos); `valid_pixel_mask`/`get_valid_mask_for_frame`/`fill_gaps_after_gcn` in `pipeline.py`. Mask is geometry-derived (assemble-ones; JUNGFRAU via `parent_data_slice` — its `PADAssembler` is broken: flat_indices/n_pixels disagree). Never derive masks from `pixel == 0` (legit photon-count zeros). JF panel-edge pixels are physically ~33% brighter → mask eroded `EDGE_EROSION_PX=2`. In `AsymmetricCXIDataset`/debug runner, image+mask are stacked `(H,W,2)` through crop/rot90/flip/cutout so both get identical transforms. |

---

## Immediate Next Steps

1. **Run asymmetric training** — submit `scripts/train_asymmetric.py` on Sol A100 nodes; log to W&B with tag `phase-04-augmentation`
2. **Evaluate with patch-grid aggregation** — run `run_patch_agg(..., aggregation="vote")` on held-out Resonet data; compare against LODO baseline AP/AUC/F1
3. **Phase-closing PR** — `phase-04-augmentation` → `main`; run `/code-review ultra` before merge; update README.md phase badge
4. **Phase 5 starts only when user confirms Phase 4 training complete**

---

## Open Decisions

| Decision | Status | Notes |
|----------|--------|-------|
| ViT variant (Base vs. Small) | Open | Decide at Phase 5 start |
| nanoBragg synthetic data | Deferred | Revisit if augmentation needed in Phase 6 |
| AGIPD generalisation gap | Under investigation | Cross AP=0.565 — root cause unknown |
