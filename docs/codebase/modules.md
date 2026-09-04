# Module Reference

> ⚡ = god node (high blast radius — run full test suite if you touch these)

Dependency layers flow bottom to top. No circular imports exist.

```
Layer 5 — Scripts        src/training/train_asymmetric.py  scripts/run_pipeline_debug.py  …
Layer 4 — DataLoader     src/data/dataloader.py
Layer 3 — Dataset        src/data/dataset.py          src/evaluation/benchmark.py
Layer 2 — Pipeline       src/preprocessing/pipeline.py
Layer 1 — I/O + Hitfinders  src/preprocessing/{io,geometry}.py   src/hitfinders/*
Layer 0 — Leaves         src/preprocessing/{normalize,augment}.py   src/evaluation/metrics.py
                         src/models/supervised.py   src/training/train_supervised.py   src/utils/config.py
```

---

## `src/preprocessing/`

Stateless functions. No class state. All functions are safe to call from multiple DataLoader workers.

| File | Purpose | Key exports |
|------|---------|-------------|
| `io.py` | Read CXI/HDF5/CBF frames and metadata | `read_frame()` ⚡, `read_embedded_labels()` ⚡, `read_detector_description()`, `count_frames()` |
| `geometry.py` | Detector panel geometry and assembly | `get_geometry()` ⚡, `get_assembler()`, `assemble_image()`, `load_pad_geometry()` |
| `normalize.py` | Global (GCN) and local (LCN) contrast normalization | `gcn()` ⚡, `lcn()` ⚡ |
| `augment.py` | Spatial augmentation and patch utilities | `pad_border()`, `random_rot90()`, `random_flip()`, `patch_grid()`, `random_cutout()` |
| `pipeline.py` | Assembled multi-step preprocessing | `assemble_only()`, `valid_pixel_mask()`, `fill_gaps_after_gcn()`, `preprocess_eval_patches()` |

**Imports from:** `reborn` (geometry assembly), `h5py`/`fabio` (file I/O), `scipy.ndimage` (LCN convolution)
**Imported by:** `src/data/dataset.py`, `src/evaluation/benchmark.py`, `scripts/`

---

## `src/hitfinders/`

Pluggable backend pattern. Swap backends by changing the config key `hitfinder.backend`.

| File | Purpose | Key exports |
|------|---------|-------------|
| `base.py` | `Hitfinder` Protocol and `MockHitfinder` test fixture | `Hitfinder`, `MockHitfinder` |
| `pf8.py` | C ctypes bridge to CrystFEL libcrystfel | `PF8Hitfinder` |
| `numpy_pf8.py` | Pure NumPy PF8 reimplementation | `NumpyPF8Hitfinder` |
| `pf8_python.py` | Cython wrapper via ssc package | `PF8HitfinderPythonWrapper` |
| `gpu_pf8.py` | pyFAI `OCL_PeakFinder` (OpenCL) | `GPUHitfinder` ⚡ |
| `gpu.py` | Thin GPU script wrapper | `GPUHitfinder` |
| `__init__.py` | Factory — returns backend by name | `get_hitfinder(name, **kwargs)` |

**Interface all backends implement:**
```python
class Hitfinder(Protocol):
    def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
        """(H, W) float32 raw frame → (N_peaks, 2) [x, y] centroids"""
```

**Multiprocessing note:** PF8 and NumPy backends are fork-safe. GPU backend requires `num_workers=0`.

**Imports from:** `reborn`, `pyFAI`, `pyopencl` (gpu backends only)
**Imported by:** `src/data/dataset.py`, `src/data/dataloader.py`

---

## `src/data/`

PyTorch Dataset and DataLoader wrappers. `AsymmetricCXIDataset` is the production dataset.

| File | Purpose | Key exports |
|------|---------|-------------|
| `dataset.py` | Dataset classes | `UnlabeledDataset` (Track 2 SSL — reads `.img`/CXI, no labels, used by `ssl_pretrain_loader()`), `MultiFrameCXIDataset` ⚡ (deprecated), `AsymmetricCXIDataset` ⚡ |
| `dataloader.py` | DataLoader factory functions | `ssl_pretrain_loader()`, `asymmetric_loader()`, `none_collate_fn()` |
| `synthetic.py` | Synthetic data helpers | (unused — reserved for future tests) |

**`AsymmetricCXIDataset.__getitem__` contract:**
- Returns `(tensor, label)` where `tensor` is `(1, 224, 224)` float32, `label` ∈ {0, 1}
- Returns `None` if no valid miss crop found after 50 attempts (filtered by `none_collate_fn`)

**Imports from:** `src/preprocessing/*`, `src/hitfinders/*`
**Imported by:** `src/training/train_asymmetric.py`
**Track separation:** `UnlabeledDataset` is imported by `src/training/train_ssl_pretrain.py` only. `AsymmetricCXIDataset` is imported by `src/training/train_asymmetric.py` only. The two tracks have separate data paths from the dataset layer up.

---

## `src/models/`

Model factory functions. Both return `(B, 2)` logit tensors from `(B, 1, 224, 224)` float32 input.

| File | Purpose | Key exports |
|------|---------|-------------|
| `supervised.py` | ResNet18/50 builder via timm | `build_supervised_model(backbone, pretrained, num_classes)` |
| `ssl.py` | MAE ViT-S/16 encoder + classification head (Track 2) | `build_mae_model()` |

**`build_supervised_model` signature:**
```python
def build_supervised_model(backbone: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module
```

**`build_mae_model` signature:**
```python
def build_mae_model(pretrained_checkpoint: str | None = None, num_classes: int = 2) -> nn.Module
```
Returns a ViT-S/16 MAE encoder with a linear classification head. Output: `(B, 2)` logits from `(B, 1, 224, 224)` float32 input — same contract as `build_supervised_model`.

**Imports from:** `timm`
**Imported by:** `src/training/train_asymmetric.py`

---

## `src/training/`

Training loop functions and the primary entry-point scripts. Invoked as `python -m src.training.<name>`.

| File | Purpose | Key exports |
|------|---------|-------------|
| `train_asymmetric.py` | **Primary Track 1 entry point** — LODO training loop, hitfinder-guided crop pipeline, wandb logging | `main()` |
| `lodo.py` | Shared LODO fold loop + session builder used by both Track 1 and Track 2 | `build_sessions()`, `run_lodo()` |
| `train_supervised.py` | One-epoch supervised training loop (utility, called by `train_asymmetric.py`) | `train_one_epoch(model, loader, optimizer, device)` |
| `train_ssl_pretrain.py` | MAE pretraining loop — mask 75% of patches, reconstruct, log pixel MSE | `main()` |
| `train_ssl_finetune.py` | SSL fine-tuning loop — attach classification head, train on labeled CXI | `main()` |

**Imports from:** `torch`, `wandb`, `src/data/`, `src/models/`, `src/evaluation/`
**Imported by:** Nothing (entry points)

---

## `src/evaluation/`

Metric functions and LODO orchestration. `benchmark.py` is the highest-level entry point for evaluation.

| File | Purpose | Key exports |
|------|---------|-------------|
| `metrics.py` | Standalone metric functions | `average_precision()`, `auc_roc()`, `f1_at_optimal_threshold()` |
| `benchmark.py` | LODO fold orchestration + vote aggregation | `run_patch_agg()` ⚡, `build_lodo_folds()`, `build_session_stratified_split()`, `run_fold()`, `run_benchmark()`, `format_results_table()` |

**`run_patch_agg` signature:**
```python
def run_patch_agg(model, sessions, cxi_paths, device, aggregation="vote", batch_size=64) -> dict
```
Returns `{"ap": float, "auc_roc": float, "f1": float, "threshold": float}`.

**Imports from:** `src/preprocessing/{geometry,io,pipeline}`, `src/evaluation/metrics`
**Imported by:** `src/training/train_asymmetric.py`, `scripts/aggregate_lodo_results.py`

---

## `src/utils/`

| File | Purpose | Key exports |
|------|---------|-------------|
| `config.py` | Merge base + named YAML config | `load_config(config_name: str) -> dict` |

Config lookup order: `configs/base.yaml` merged with `configs/<track>/<name>.yaml`. Later keys win.

**Imported by:** All scripts.

---

## `scripts/` — Entry Points

All scripts are self-contained and import from `src/`. Run with `python scripts/<name>.py`. Training entry points live in `src/training/` and are run with `python -m src.training.<name> --config <name>`.

| Script | Purpose | Key imports |
|--------|---------|-------------|
| `run_pipeline_debug.py` | Debug and profile the preprocessing pipeline | `preprocessing.*`, `data`, `hitfinders.gpu` |
| `smoke_test_detector_shapes.py` | Verify geometry + assembly for all detectors | `preprocessing.{geometry,io,augment,pipeline}` |
| `visualize_assembled.py` | Save assembled detector images to PNG | `preprocessing.{geometry,io,pipeline}`, `matplotlib` |
| `probe_hdf5.py` | Print HDF5/CXI file tree: keys, shapes, dtypes | (stdlib + h5py only) |
| `aggregate_lodo_results.py` | Read `results.json` from all checkpoints, compute mean±std | `evaluation.benchmark` |
