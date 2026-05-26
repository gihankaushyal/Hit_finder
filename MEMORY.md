# Project Memory — SFX Hitfinder

> Updated: 2026-05-23 | Read this at session start before anything else.

---

## Current Status

| Item | Value |
|------|-------|
| Active branch | `main` (Phase 3 merged; Phase 4 branch not yet created) |
| Current phase | **Phase 4 — Supervised Baseline** (starting) |
| Last merge | PR #5 `phase-03 → main`, 2026-05-23 |
| Test suite | 135 passed, 2 skipped (fabio absent), 1 pre-existing Eiger4M bitshuffle fail |

---

## Phase History

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
| 6 | `anaconda_projects/` directory is untracked in working directory | Investigate before Phase 4 branch; determine if it's active setup or legacy artifact. |

---

## Immediate Next Steps (Phase 4)

1. **Create `phase-04` branch** from main: `git checkout -b phase-04 main`
2. **Run CLAUDE.md audit** per constraint #9 (at phase start): `/claude-md-management:claude-md-improver`
3. **First feature**: `src/data/dataset.py` — `SFXDataset` (labeled HDF5, lazy-load) and update `UnlabeledDataset`
4. **Config**: `configs/supervised/resnet18.yaml` — learning rate, batch size, weight decay
5. **Training script**: `src/training/train_supervised.py` — ResNet18 fine-tune loop with wandb logging
6. **ViT variant decision** (Base vs. Small) needed before Phase 5 SSL track — defer until Phase 5 start

---

## Open Decisions

| Decision | Status | Notes |
|----------|--------|-------|
| ViT variant (Base vs. Small) | Open | Latency vs. capacity; decide at Phase 5 start |
| nanoBragg synthetic data | Deferred to Phase 4 | Real `.img` files used for Phase 3 |
