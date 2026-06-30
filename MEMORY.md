# Project Memory — SFX Hitfinder

> Updated: 2026-06-05 | Read this at session start before anything else.

---

## Current Status

| Item | Value |
|------|-------|
| Active branch | `main` (Phase 4 merged 2026-05-28) |
| Current phase | **Phase 4 extended — supervised baseline testing** |
| Last merge | PR #15 — evaluation script + synthetic baseline results (2026-06-05) |
| Test suite | 135 passed, 2 skipped (fabio absent), 1 pre-existing Eiger4M bitshuffle fail |

---

## Phase History

### Phase 4 — Supervised Baseline (complete, 2026-05-28)
Full ResNet18/50 supervised track implemented: `SFXDataset` (lazy HDF5 load), `load_config()` YAML deep-merge, `build_supervised_model()` via timm (in_chans=1), training loop with AdamW + CrossEntropyLoss, wandb logging (loss/AP/AUC/F1 per epoch), checkpoint saved on best val F1.

**Synthetic baseline results (2026-06-05):** `resnet18-10k-full-seed42` trained on 10k synthetic frames (`hitfinder_10k_merged.h5`). Early stopped epoch 22/200. Evaluated on 2000 held-out frames (`hitfinder_val`): AP=0.9998, AUC=0.9998, F1=0.9995, Precision=1.0, Recall=0.999 (1 miss, 0 false alarms). Checkpoint at `checkpoints/resnet18-10k-full-seed42/best.pt`. Evaluation script: `scripts/evaluate_supervised.py`.

Label encoding (both train and val): `labels[:, -1]` is `bg_only`; value `1.0` → non-hit (class 0), value `0.0` → hit (class 1). Images pre-assembled 512×512 uint16; Reborn geometry step skipped.

Real-detector LODO HPC run still pending — carries forward as parallel task during Phase 5.

### Docs update (2026-05-26)
Added `MEMORY.md` (session-start context), updated `CLAUDE.md` with session-start pointer, fixed directory tree, corrected JUNGFRAU 4M dimensions, fixed HDF5 example to use `[0]` not `[()]`, added confirmed `lcn_window=9`. Fixed `PLANNING.md` roadmap table (Phase 3 → COMPLETE, Phase 4 → CURRENT) and added Phase 4 checklist.

### Phase 3 — Preprocessing (complete, 2026-05-23)
Built the full shared preprocessing pipeline: Reborn geometry assembly for all four detectors → GCN → LCN → resize 224×224. Key outcomes:
- **LCN window = 9** confirmed via ablation (window=31 caused panel-edge ringing artifacts)
- **JUNGFRAU_4M geometry** replaced Reborn built-in with custom JSON derived from CrystFEL `.geom` file (`src/preprocessing/data/jungfrau4m_jf4m_103mm.json`). The HDF5 frame is a 2164×2068 canvas with gap pixels between 8 modules; panels must be extracted via `parent_data_slice` before passing to `PADAssembler`.
- **HDF5 keys confirmed** for all four detectors (see `docs/data_spec.md`). AGIPD key is `entry_1/instrument_1/detector_1/detector_corrected/data` (file was silently replaced mid-phase — old file had key `entry_1/data_1/data` and shape N×5632×384).
- **Label format** decided: JSON sidecar (`labels.json`), keys are absolute paths, values 0/1.

### Phase 2 — Data Infrastructure (complete)
Synthetic data pipeline (`src/data/synthetic.py`), `UnlabeledDataset` skeleton, real `.img` files used for SSL pretraining.

### Phase 1 — Proposal (complete)
Architecture fixed: Track 1 ResNet supervised, Track 2 MAE ViT SSL. Shared preprocessing pipeline. Detector type always from metadata.

---

## Known Gotchas

| # | Gotcha | Impact |
|---|--------|--------|
| 1 | AGIPD CXI file replaced mid-Phase 3 — old key `entry_1/data_1/data` (N×5632×384) no longer valid | `io.py` fallback list updated; old AGIPD files will silently match `entry_1/data_1/data` and return wrong shape |
| 2 | JUNGFRAU_4M Reborn built-in loader expects raw panels; actual HDF5 is pre-assembled canvas with gap pixels | Always use `jungfrau4m_crystfel_pad_geometry_list()` not `detector.jungfrau4m_pad_geometry_list` |
| 3 | Eiger4M real-data tests raise `OSError: Can't find plugin` (bitshuffle HDF5 filter) | Pre-existing; needs `hdf5plugin` import before `h5py.File()`. CI is safe (file absent → skip). |
| 4 | GitHub ruleset `protect_main` had wrong required check name `"CI / test"` — actual name is `"test"` | Fixed 2026-05-23. If CI gating breaks again, check ruleset at repo Settings → Rules. |
| 5 | `notebooks/lcn_ablation_executed.ipynb` is untracked — intentional executed artifact | Do not commit. The source notebook `lcn_ablation.ipynb` is the committed version. |
| 6 | `anaconda_projects/` directory is untracked in working directory | Status unresolved through Phase 4 — determine if active setup or legacy artifact before Phase 5 branch. |
| 7 | Resonet geometry file is named `Eigar.geom` (typo for "Eiger") | Copied to `src/preprocessing/data/eiger_resonet.geom` — use that path, not the original. |
| 8 | Resonet 25k file is `cxi_merged_25k.cxi` not `cxi_25k.cxi` | Full path: `/data/bioxfel/user/gihan/Resonet/cxi_merged_25k.cxi` |
| 9 | `geometry_file_to_pad_geometry_list()` is in `reborn.external.crystfel`, not `reborn.detector` | Always import from `reborn.external.crystfel` |

---

## Resonet CXI Integration (2026-06-11)

New multi-frame CXI format (Resonet-generated) now fully supported. Key additions:

- `src/preprocessing/io.py` — `count_frames()`, `read_frame(idx)`, `read_embedded_labels()`
- `src/preprocessing/pipeline.py` — `preprocess_assembled()` (geometry-bypass)
- `src/preprocessing/geometry.py` — `eiger_resonet_pad_geometry_list()`, `extract_panels_from_canvas()`, `"EigerRESoNeT"` in `DETECTOR_LOADERS`; geometry file at `src/preprocessing/data/eiger_resonet.geom`
- `src/data/dataset.py` — `MultiFrameCXIDataset` (multi-frame CXI + embedded labels)
- `scripts/evaluate_resonet_cxi.py`, `scripts/train_resonet_cxi.py`, `scripts/submit_resonet_train.sh`
- `configs/supervised/resnet18_resonet.yaml`

**Key finding:** Reborn geometry assembly is NOT needed for Resonet Eiger normalization — `preprocess_assembled()` is identical in output (canvas_size == n_pixels → assembly is identity). Reborn geometry only matters for AGIPD and JUNGFRAU_4M.

**Training job:** SLURM job 55223044 submitted — ResNet18 on `cxi_merged_25k.cxi` (20k train / 5k val, 100 epochs). Checkpoint → `checkpoints/resnet18-resonet-seed42/best.pt`.

**Inference baseline (resnet18-10k-full-seed42 on Resonet CXI):** AUC=0.59, AP=0.61, F1=0.68 — domain shift from synthetic hitfinder_10k data.

## Immediate Next Steps

> **Phase 5 on hold (2026-06-05):** Continuing to test and validate the supervised baseline before starting SSL work. Do not begin Phase 5 until explicitly instructed.

1. **Wait for job 55223044** — evaluate `resnet18-resonet-seed42` on `cxi_100/` once complete
2. Real-detector LODO HPC run still pending (train on 3 detectors, eval on held-out 4th)
3. `scripts/submit_supervised.sh` — needed before first real-detector HPC run
4. **Phase 5 starts only when user says testing is complete**

---

## Open Decisions

| Decision | Status | Notes |
|----------|--------|-------|
| ViT variant (Base vs. Small) | Open | Latency vs. capacity; decide at Phase 5 start |
| nanoBragg synthetic data | Deferred past Phase 4 | Real `.img` files used for Phase 3; not addressed in Phase 4 either — revisit if augmentation needed in Phase 6 |
