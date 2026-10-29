# Module Reference

> ⚡ = god node (high blast radius — run full test suite if you touch these)

Dependency layers flow bottom to top. No circular imports exist.

```
Layer 6 — Entry points   src/training/{train_asymmetric,train_ssl_pretrain,train_ssl_finetune}.py  scripts/*.py
Layer 5 — Training glue  src/training/{lodo,run_naming,wandb_identity,override_marker,inference_results}.py
Layer 4 — DataLoader     src/data/dataloader.py
Layer 3 — Dataset        src/data/dataset.py          src/evaluation/benchmark.py
Layer 2 — Cache + Pipeline  src/data/frame_cache.py   src/preprocessing/pipeline.py
Layer 1 — I/O + Hitfinders  src/preprocessing/{io,geometry}.py   src/hitfinders/*
Layer 0 — Leaves         src/preprocessing/{normalize,augment}.py   src/evaluation/metrics.py
                         src/models/{supervised,ssl}.py   src/training/train_supervised.py   src/utils/config.py
```

(Layering is approximate: `dataset.py` imports `frame_cache.py`, and `lodo.py` imports the dataloader, benchmark, model, hitfinder base and the helper modules above it. No circular imports exist.)

---

## `src/preprocessing/`

Stateless functions. No class state. All functions are safe to call from multiple DataLoader workers.

| File | Purpose | Key exports |
|------|---------|-------------|
| `io.py` | Read CXI/HDF5/CBF frames and metadata | `read_frame()` ⚡, `read_embedded_labels()` ⚡, `read_detector_description()`, `count_frames()`, `read_image()` |
| `geometry.py` | Detector panel geometry and assembly. `DETECTOR_LOADERS` maps short names (`AGIPD`, `JUNGFRAU_4M`, `ePix10k`, `Eiger4M`, `EigerRESoNeT`) to Reborn geometry-list functions; `_DESC_LOADERS` does the same for CXI description strings | `get_geometry(detector_desc)` ⚡, `get_assembler(detector_desc)`, `assemble_image()`, `load_pad_geometry(detector_type, detector_distance)`, `extract_panels_from_canvas()`, `jungfrau_8_pad_geometry_list()`, `eiger4m_64_pad_geometry_list()`, `eiger_resonet_pad_geometry_list()` |
| `normalize.py` | Global (GCN) and local (LCN) contrast normalization | `gcn()` ⚡, `lcn()` ⚡, `lcn_torch()` (batched torch LCN used by `run_patch_agg`; must match the masked NumPy `lcn` within float32 tolerance) |
| `augment.py` | Spatial augmentation and patch utilities | `pad_border()`, `random_rot90()`, `random_flip()`, `patch_grid()`, `random_cutout()` (defaults: 3 holes, `CUTOUT_HOLE_SIZE_DEFAULT=24`, `CUTOUT_AVOID_MARGIN_DEFAULT=8`, `CUTOUT_MAX_TRIES=20`) |
| `pipeline.py` | Assembled multi-step preprocessing | `assemble_only(frame, pads, detector_desc, assembler)`, `valid_pixel_mask()`, `get_valid_mask_for_frame()`, `fill_gaps_after_gcn()`, `preprocess_eval_patches()`, `preprocess_eval_patches_from_gcn()` |

**Imports from:** `reborn` (geometry assembly), `h5py`/`fabio` (file I/O), `scipy.ndimage` (LCN convolution)
**Imported by:** `src/data/dataset.py`, `src/data/frame_cache.py` (the GCN/LCN constants and `EDGE_EROSION_PX`), `src/evaluation/benchmark.py`, `scripts/`

---

## `src/hitfinders/`

Pluggable backend pattern. Swap backends by changing the config key `hitfinder.backend`.

| File | Purpose | Key exports |
|------|---------|-------------|
| `base.py` | `Hitfinder` Protocol and `MockHitfinder` test fixture | `Hitfinder`, `MockHitfinder` |
| `pf8.py` | C ctypes bridge to CrystFEL libcrystfel | `PF8Hitfinder` |
| `numpy_pf8.py` | Pure NumPy PF8 reimplementation | `NumpyPF8Hitfinder` |
| `pf8_python.py` | Cython wrapper via ssc package | `PF8HitfinderPythonWrapper` |
| `gpu_pf8.py` | User-script module built on pyFAI `OCL_PeakFinder` (OpenCL); module-level functions, no class. Loaded by path via `hitfinder.gpu_script_path` | `find_peaks(frame)`, `set_geometry(...)` |
| `gpu.py` | Wrapper that imports the user GPU script and exposes it as a `Hitfinder` | `GPUHitfinder` |
| `__init__.py` | Factory — picks the backend from `cfg["hitfinder"]["backend"]` (`pf8`, `pf8_numpy`, `pf8_python`, `gpu`, `mock`; ValueError otherwise) | `get_hitfinder(cfg)` |

**Interface all backends implement:**
```python
class Hitfinder(Protocol):
    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """(H, W) float32 raw frame → (N_peaks, 2) [x, y] centroids"""
```

**Multiprocessing note:** PF8 and NumPy backends are fork-safe. GPU backend requires `num_workers=0` (CUDA/OpenCL context is not fork-safe). The guard is an `isinstance(hitfinder, GPUHitfinder)` check in `asymmetric_loader()` and `ssl_crop_loader()`, so a new hitfinder that holds a GPU context is not protected unless it handles this itself.

**Imports from:** `reborn`, `pyFAI`, `pyopencl` (gpu backends only)
**Imported by:** `src/data/dataset.py` and `src/data/dataloader.py` (the `Hitfinder` type from `hitfinders.base`), `src/training/lodo.py` (`Hitfinder` type), `get_hitfinder()` callers `src/training/train_asymmetric.py`, `train_ssl_pretrain.py`, `train_ssl_finetune.py`, `scripts/build_frame_cache.py` and `scripts/diagnose_crop_shortcut.py`

---

## `src/data/`

PyTorch Dataset and DataLoader wrappers, plus the on-disk frame cache. `AsymmetricCXIDataset` is the production Track 1 dataset; `SSLPretrainCXIDataset` is the Track 2 pretraining dataset.

| File | Purpose | Key exports |
|------|---------|-------------|
| `dataset.py` | Dataset classes | `AsymmetricCXIDataset` ⚡ (Track 1; `set_epoch()` re-seeds crops each epoch), `SSLPretrainCXIDataset` (Track 2; several random valid-region crops per frame, no labels), `UnlabeledDataset` (whole-image, no labels; only used by `ssl_pretrain_loader()` and tests), `MultiFrameCXIDataset` (deprecated) |
| `dataloader.py` | DataLoader factory functions | `asymmetric_loader()`, `ssl_crop_loader()`, `ssl_pretrain_loader()` (legacy), `none_collate_fn()`; internal `_flatten_collate` / `_ssl_flatten_collate` flatten the per-frame crop stacks |
| `frame_cache.py` | Two-tier on-disk cache of the read → assemble → hitfinder → GCN prefix (Phase 5) | `FrameCache`, `frame_cache_from_cfg()`, `verify_cache_or_raise()`, `build_manifest()`, `write_manifest()`, `manifest_diff()`, `verify_manifest()`, `CacheStaleError`, `CacheMissError` |
| `synthetic.py` | Synthetic data helpers | (unused — reserved for future tests) |

**`AsymmetricCXIDataset.__getitem__` contract:**
- Returns `(crops, labels)` where `crops` is `(crops_per_frame, 1, 224, 224)` float32 and `labels` is `(crops_per_frame,)` long ∈ {0, 1}; `asymmetric_loader()` flattens these into ordinary batches
- Returns `None` if `crops_per_frame` crops could not be drawn within `crops_per_frame × CROP_RETRY_FACTOR` (4) attempts; on a frame with centroids a Path-B draw that finds no clear position (`hard_neg_max_attempts`, default 50) falls back to Path A; an attempt fails (None) only when the frame has no centroids and no clear crop is found
- Path A / Path B coin toss is controlled by `hit_frac` (default 0.5); the hitfinder only runs when the embedded label is HIT (metadata gating)
- The GCN'd frame comes from `_load_gcn_frame()`: the `FrameCache` if configured, falling back to `_compute_gcn_frame()` on `CacheMissError`

**Imports from:** `src/preprocessing/*`, `src/hitfinders/*`; `dataset.py` and `dataloader.py` also import `frame_cache.py`
**Imported by:** `asymmetric_loader` ← `src/training/lodo.py`; `ssl_crop_loader` ← `src/training/train_ssl_pretrain.py`; `frame_cache.py` ← `dataset.py`, `dataloader.py`, `src/evaluation/benchmark.py`, `lodo.py`, the three training entry points, `scripts/build_frame_cache.py`
**Track separation:** `ssl_crop_loader` / `SSLPretrainCXIDataset` feed MAE pretraining (`train_ssl_pretrain.py`); `asymmetric_loader` / `AsymmetricCXIDataset` feed Track 1 and SSL fine-tuning (both go through `lodo.py`). Pretraining and supervised paths share the same `_load_gcn_frame` prefix.

---

## `src/models/`

Model factory functions. The classifiers return `(B, 2)` logit tensors from `(B, 1, 224, 224)` float32 input.

| File | Purpose | Key exports |
|------|---------|-------------|
| `supervised.py` | ResNet18/50 builder via timm | `build_supervised_model(backbone, pretrained, num_classes)` |
| `ssl.py` | MAE ViT-S/16 autoencoder, masking, and the fine-tune classifier (Track 2) | `build_mae_model(cfg)` → `MAEViT`, `build_ssl_classifier(cfg, mae_checkpoint, freeze_encoder)` → `ViTClassifier`, `random_masking()`, `peak_aware_masking()`, `get_2d_sincos_pos_embed()` |

**`build_supervised_model` signature:**
```python
def build_supervised_model(backbone: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module
```

**`build_mae_model` / `build_ssl_classifier` signatures** (both read `cfg["ssl"]` for `embed_dim`, `depth`, `num_heads`, decoder sizes, `mask_ratio`, …):
```python
def build_mae_model(cfg: dict) -> MAEViT
def build_ssl_classifier(cfg: dict, mae_checkpoint: str | None = None, freeze_encoder: bool = False) -> ViTClassifier
```
`build_mae_model` returns the full MAE (encoder + decoder) used for pretraining. `build_ssl_classifier` loads the pretrained encoder from `mae_checkpoint`, attaches a 2-class head, and with `freeze_encoder=True` gives the linear-probe variant (`--linear-probe`). Output: `(B, 2)` logits — same contract as `build_supervised_model`.

**Imports from:** `timm` (supervised only)
**Imported by:** `supervised.py` ← `src/training/lodo.py`; `ssl.py` ← `src/training/train_ssl_pretrain.py` (`build_mae_model`), `src/training/train_ssl_finetune.py` (`build_ssl_classifier`), `scripts/visualize_mae_recon.py`, `scripts/diagnose_crop_shortcut.py`

---

## `src/training/`

Training loop functions and the primary entry-point scripts. Invoked as `python -m src.training.<name>`.

| File | Purpose | Key exports |
|------|---------|-------------|
| `train_asymmetric.py` | **Primary Track 1 entry point** — LODO driver, hitfinder-guided crop pipeline, wandb logging. Flags: `--config`, `--run-name-prefix` (required), `--folds`, `--device`, `--intra`, `--tags`, `--resume-training` / `--override-training`, `--inference-only`, `--cache-root`, `--cache-nvme`, `--no-cache` | `main()` |
| `lodo.py` | Shared LODO fold loop (train, evaluate, write `results.json`) and session builder, used by Track 1 and the SSL fine-tune | `build_sessions(lodo_cfg)`; private `_train_fold()`, `_build_intra_split()` |
| `train_supervised.py` | One-epoch supervised training loop (utility, called by `lodo.py`) | `train_one_epoch(model, loader, optimizer, device)` |
| `train_ssl_pretrain.py` | MAE pretraining loop — mask patches (`ssl.mask_ratio`, default 0.6), reconstruct, log pixel MSE and sample reconstructions. Requires `--fold` (the fold's test detector is excluded) and `--run-name-prefix` | `main()` |
| `train_ssl_finetune.py` | SSL fine-tuning or linear probe (`--linear-probe`) — attach classification head to the pretrained encoder (`--pretrain-checkpoint`), train on labeled CXI; also supports `--inference-only` | `main()` |
| `run_naming.py` | Versioned run-name prefix validation and the resume/override checkpoint gate shared by all entry points | `validate_run_name_prefix()`, `fold_run_name()`, `expand_finetune_prefix()`, `check_checkpoint_collisions()`, `effective_resume()` |
| `override_marker.py` | `override_applied.json` in the run dir: a requeued SLURM job (same job id, higher restart count) resumes instead of re-running `--override-training` | `write_override_marker()`, `override_already_applied()`, `current_job_id()`, `current_restart_count()` |
| `wandb_identity.py` | New W&B run id per override attempt (`<run>`, `<run>-o2`, …) kept in `<run_dir>/wandb_id.txt`; previous run tagged `overridden` | `resolve_wandb_id()`, `rotate_wandb_id()`, `next_wandb_id()`, `ensure_fresh_wandb_id()`, `wandb_id_for_training()`, `tag_overridden()`, `OverrideHook`, `override_hook_from_cfg()` |
| `inference_results.py` | Result-file naming (`results.json` vs `results.inference.json`), the `training` / `inference` blocks, W&B summary updates, and the training-sanity check for `--inference-only` | `inference_result_path()`, `training_block()`, `inference_block()`, `summary_updates()`, `assess_training()`, `fetch_wandb_history()`, `loss_flatness()`, `wandb_enabled()` |

**Imports from:** `torch`, `wandb`, `src/data/`, `src/models/`, `src/evaluation/`; `lodo.py` also imports `inference_results.py` and `wandb_identity.py`
**Imported by:** `train_asymmetric.py` and `train_ssl_finetune.py` import `lodo` (`_train_fold`, `build_sessions`), `run_naming`, `wandb_identity` and `inference_results`; `train_ssl_pretrain.py` imports `lodo.build_sessions`, `run_naming`, `wandb_identity` and `train_supervised._set_seeds`. Nothing imports the entry points themselves.

---

## `src/evaluation/`

Metric functions and LODO orchestration. `benchmark.py` is the highest-level entry point for evaluation.

| File | Purpose | Key exports |
|------|---------|-------------|
| `metrics.py` | Standalone metric functions | `average_precision()`, `auc_roc()`, `f1_at_optimal_threshold()` |
| `benchmark.py` | LODO fold orchestration + vote aggregation | `run_patch_agg()` ⚡, `build_lodo_folds()`, `build_session_stratified_split()`, `run_fold()`, `run_benchmark()`, `run_on_loader()`, `save_split_artifact()`, `load_split_artifact()`, `format_results_table()` |

**`run_patch_agg` signature:**
```python
def run_patch_agg(model, session_map, session_ids, label_key, patch_size=224, patch_stride=224,
                  min_hit_patches=3, device="cpu", inference_batch_size=64,
                  aggregation="vote", frame_cache=None) -> dict
```
Returns `{"ap": float, "auc_roc": float, "f1": float, "threshold": float}`. `min_hit_patches` is accepted and recorded in the result files but reserved: it does not change the score; the vote score is `hit_count / n_patches` with a patch counted as a hit above 0.5. `f1` / `threshold` are the optimum on the evaluated frames.

**Imports from:** `src/preprocessing/{geometry,io,pipeline}`, `src/evaluation/metrics`, `src/data/frame_cache`
**Imported by:** `src/training/lodo.py`, `src/training/train_asymmetric.py`, `src/training/train_ssl_finetune.py`, `src/training/train_ssl_pretrain.py` (`build_lodo_folds`), and `scripts/{plan_cache_staging,diagnose_crop_shortcut,visualize_mae_recon}.py`. (`scripts/aggregate_lodo_results.py` reads `results.json` files and does not import `benchmark`.)

---

## `src/utils/`

| File | Purpose | Key exports |
|------|---------|-------------|
| `config.py` | Deep-merge `configs/base.yaml` with a model YAML | `load_config(config_path, base_path) -> dict` |

`load_config` takes a path to the model YAML (e.g. `configs/supervised/resnet18_asymmetric.yaml`) and always deep-merges `configs/base.yaml` underneath it; the model YAML's values win. Do not add `defaults: [base]` to model YAMLs.

**Imported by:** `src/training/` entry points and the scripts `build_frame_cache.py`, `diagnose_crop_shortcut.py`, `plan_cache_staging.py`, `run_pipeline_debug.py` and `visualize_mae_recon.py` (the other scripts do not use it).

---

## `scripts/` — Entry Points

All scripts are self-contained and import from `src/`. Run with `python scripts/<name>.py`. Training entry points live in `src/training/` and are run with `python -m src.training.<name> --config <name>`.

Python utilities (files here that import from `src/` carry a `sys.path.insert` hack that `src/training/` files avoid):

| Script | Purpose | Key imports |
|--------|---------|-------------|
| `run_pipeline_debug.py` | Debug and profile the preprocessing pipeline | `preprocessing.*`, `data`, `hitfinders.gpu` |
| `smoke_test_detector_shapes.py` | Verify geometry + assembly for all detectors | `preprocessing.{geometry,io,augment,pipeline}` |
| `visualize_assembled.py` | Save assembled detector images to PNG | `preprocessing.{geometry,io,pipeline}`, `matplotlib` |
| `visualize_mae_recon.py` | Masked input / reconstruction / target renders for an MAE checkpoint (output in `docs/figures/mae_recon/`) | `models.ssl`, `data.dataloader`, `evaluation.benchmark` |
| `probe_hdf5.py` | Print HDF5/CXI file tree: keys, shapes, dtypes | (stdlib + h5py only) |
| `aggregate_lodo_results.py` | Read `checkpoints/<run>-fold*/results.json`, print per-variant tables with sample std; interactive menu, `--select`, `--run-prefix` | (stdlib + numpy; reads `results.json` files only) |
| `build_frame_cache.py` | Build the on-disk frame cache once per frame (float16, memory-mappable) | `data.dataset._compute_gcn_frame`, `data.frame_cache` |
| `plan_cache_staging.py` | Emit one fold's cache entry directories in NVMe staging priority order (train, then val, then test sessions) | `evaluation.benchmark`, `training.lodo`, `utils.config` |
| `plot_hit_frac.py` | Plot realized `train/realized_hit_frac` against the configured value from W&B (`wandb.Api()`) | `wandb`, `matplotlib`, `training.wandb_identity` |
| `diagnose_crop_shortcut.py` | Measure how much a trained classifier relies on crop-construction shortcuts (peak-centred positives vs. clearance negatives) | `data.dataset`, `models.ssl`, `evaluation.benchmark` |
| `verify_eiger4m_geometry_match.py` | One-off check that Reborn's bundled `eiger4m_64_pad_geometry_list()` matches the former local `eiger4m.geom` geometry (that file has since been deleted, so the script can no longer run: it raises `FileNotFoundError` on the missing local Eiger4M `.geom` file); kept as a record, not part of pytest | `preprocessing.geometry`, `reborn` |

SLURM wrappers (start from `scripts/templates/slurm_gpu_job.sh`; the training wrappers request one `h100` on node `scg020` and `source .secrets/wandb.env` before the training command):

| Group | Scripts |
|-------|---------|
| Track 1 LODO (current) | `submit_asymmetric_lodo_all.sh` (orchestrates all folds: NVMe cache staging, training, cleanup), `submit_asymmetric_lodo_fold.sh` (one fold) |
| Frame cache | `stage_frame_cache.sh` (stage one fold's entries to NVMe, `xargs -P 8`), `submit_eiger_cache.sh`, `submit_epix10k_cache.sh`, `submit_jungfrau_cache.sh` (per-detector builds) |
| Track 2 (SSL) | `submit_ssl_pretrain.sh` / `submit_ssl_pretrain_all.sh`, `submit_ssl_finetune.sh` / `submit_ssl_finetune_all.sh` (orchestrators stage → train → cleanup per fold), `stage_ssl_data.sh`, `cleanup_ssl_stage.sh` |
| Diagnostics and smoketests | `submit_epix_cache_smoketest.sh` (+ `_crops4`, `_crops8`), `submit_crop_shortcut_diag.sh`, `env_check.sh` |
| **Deprecated** (exit immediately and name their replacement; pinned by `tests/test_deprecated_scripts.py`) | `submit_all_lodo_folds.sh`, `submit_asymmetric_lodo.sh`, `submit_lodo_parallel.sh`, `submit_agipd_lodo.sh` → `submit_asymmetric_lodo_all.sh`; `submit_epix_smoketest.sh`, `submit_agipd_smoketest.sh`, `submit_resonet_smoketest.sh` → `submit_epix_cache_smoketest.sh` |
