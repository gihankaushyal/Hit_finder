# Extension Guide

Step-by-step recipes for the common extension points: a new detector, hitfinder backend, model backbone, metric or SSL config, plus the frame cache and the SLURM submit scripts. The last sections (run naming, re-evaluating a finished run, W&B ids) describe behaviour you must keep intact when extending the training entry points.

---

## New Detector {#new-detector}

Add support for a detector geometry not yet in the system. Detector identity always comes from the CXI description string (`entry_1/instrument_1/detector_1/description`), never from the pixels.

- [ ] **1. Write a geometry loader in `src/preprocessing/geometry.py`**

  It must return a Reborn `PADGeometryList`. Either delegate to a Reborn bundled loader (as `jungfrau_8_pad_geometry_list()` and `eiger4m_64_pad_geometry_list()` do) or parse a CrystFEL file with `geometry_file_to_pad_geometry_list()` (as `eiger_resonet_pad_geometry_list()` does, using `src/preprocessing/data/eiger_resonet.geom`). If the CXI file stores all panels in one 2D canvas, the geometry must define `parent_data_slice` (`pads.defines_slicing()` is True): `assemble_only()` then slices panels out with `extract_panels_from_canvas()` before `PADAssembler`.

  ```python
  def my_detector_pad_geometry_list(detector_distance: float = 0.1) -> detector.PADGeometryList:
      return detector.my_detector_pad_geometry_list(detector_distance=detector_distance)
  ```

- [ ] **2. Register it in three places in `geometry.py`**

  ```python
  _KNOWN_DESCS = {"AGIPD 1M", "ePix10k 2.2M", "EIGER 4M", "Jungfrau 4M", "My Detector 2M"}   # CXI description strings

  _DESC_LOADERS = {          # keyed by the CXI description string, used by get_geometry()
      ...,
      "My Detector 2M": my_detector_pad_geometry_list,
  }

  DETECTOR_LOADERS = {       # keyed by short detector name, used by load_pad_geometry()
      ...,
      "MyDetector": my_detector_pad_geometry_list,
  }
  ```

  `get_geometry(detector_desc)` and `get_assembler(detector_desc)` take only the description string and cache their results; `assemble_only()` raises `ValueError` for a description that is not in `_KNOWN_DESCS`.

- [ ] **3. Check the valid-pixel mask**

  `valid_pixel_mask(detector_desc)` in `src/preprocessing/pipeline.py` builds the gap/padding mask automatically by assembling an all-ones array through the same assembler, then erodes it by `EDGE_EROSION_PX` (2). Call it once for the new description and check the returned shape equals the assembled canvas.

- [ ] **4. Add to the LODO fold definition in `src/evaluation/benchmark.py`** (only if it should be a held-out fold)

  ```python
  DETECTORS: list[str] = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M", "MyDetector"]
  ```

  `build_lodo_folds()` derives one fold per entry. Also add the detector directory to `lodo.detector_dirs` in the run configs (`configs/supervised/*.yaml`, `configs/ssl/*.yaml`); `build_sessions()` globs `compressed*.cxi` under it.

- [ ] **5. Frame cache**

  If you use the frame cache, build entries for the new directory: `python -m scripts.build_frame_cache --config <cfg> --detectors MyDetector`. A new local `.geom`/`.json` file under `src/preprocessing/data/` is picked up by the manifest's geometry digest automatically. Geometry that comes from **Reborn's bundled loaders is invisible to that digest**; if such geometry changes, bump `PIPELINE_VERSION` in `src/data/frame_cache.py` by hand to force a rebuild. See [Build or invalidate the frame cache](#frame-cache).

- [ ] **6. Smoke test and unit test**

  Add the detector to `scripts/smoke_test_detector_shapes.py`, and a test in `tests/test_geometry_assembly.py` using the existing style (`get_geometry(desc)` returns a `PADGeometryList`; `assemble_only(frame, pads, desc)` returns a 2D float32 canvas):

  ```python
  def test_my_detector_assembly():
      pads = get_geometry("My Detector 2M")
      frame = np.zeros(NATIVE_SHAPE, dtype=np.float32)      # shape as stored in the CXI file
      canvas = assemble_only(frame, pads, "My Detector 2M")
      assert canvas.ndim == 2 and canvas.dtype == np.float32
  ```

---

## New Hitfinder Backend {#new-hitfinder}

Add a new peak-finding algorithm that implements the `Hitfinder` protocol.

- [ ] **1. Create `src/hitfinders/my_finder.py`**

  ```python
  import numpy as np

  class MyHitfinder:
      def __init__(self, threshold: float = 100.0):
          self.threshold = threshold

      def find_peaks(self, assembled: np.ndarray) -> np.ndarray:
          """
          Args:
              assembled: (H, W) float32 raw frame (pre-GCN)
          Returns:
              (N_peaks, 2) float32 array of [x, y] centroids
              (0, 2) array if no peaks found
          """
          centroids = []
          return np.array(centroids, dtype=np.float32).reshape(-1, 2)
  ```

  If the backend holds a GPU context it is not fork-safe: the `num_workers=0` guard in the loaders is an `isinstance(..., GPUHitfinder)` check, so extend it for your class.

  Optionally add `set_geometry(dist, wavelength, pixel_size)`; the dataset calls it (when present) each time the CXI file changes.

- [ ] **2. Register in `src/hitfinders/__init__.py`**

  `get_hitfinder(cfg)` takes the whole config dict and dispatches with an `if`/`elif` chain on `cfg["hitfinder"]["backend"]` (currently `pf8`, `pf8_numpy`, `pf8_python`, `gpu`, `mock`; unknown names raise `ValueError`). Add a branch that reads its keys from `cfg["hitfinder"]`, and export the class in `__all__`:

  ```python
  from src.hitfinders.my_finder import MyHitfinder
  ...
      if backend == "my_finder":
          return MyHitfinder(threshold=hf_cfg.get("my_finder_threshold", 100.0))
  ```

- [ ] **3. Add config keys** to the `hitfinder:` block of the config you run (for example `configs/supervised/resnet18_asymmetric.yaml`; shared defaults live in `configs/base.yaml`):

  ```yaml
  hitfinder:
    backend: my_finder
    my_finder_threshold: 100.0
  ```

- [ ] **4. Add tests in `tests/test_hitfinders.py`**, which already contains the patterns to copy (`test_get_hitfinder_mock`, `test_get_hitfinder_unknown_backend_raises`, `test_hitfinder_protocol_isinstance`):

  ```python
  def test_my_hitfinder_returns_correct_shape():
      peaks = MyHitfinder(threshold=100.0).find_peaks(np.random.rand(512, 512).astype(np.float32))
      assert peaks.ndim == 2 and peaks.shape[1] == 2

  def test_get_hitfinder_my_finder():
      cfg = {"hitfinder": {"backend": "my_finder", "my_finder_threshold": 100.0}}
      assert isinstance(get_hitfinder(cfg), MyHitfinder)
  ```

  The frame-cache manifest records the hitfinder config through `_HITFINDER_KEYS` in `src/data/frame_cache.py`. Add your new config keys (and `backend` is already covered) to that tuple, otherwise a changed `my_finder_*` setting will not mark an existing cache stale; then rebuild the cache.

---

## New Model / Backbone {#new-model}

Add a new backbone to the supervised track. `build_supervised_model(backbone, pretrained, num_classes=2)` in `src/models/supervised.py` is already generic: it passes the name straight to `timm.create_model(backbone, pretrained=..., in_chans=1, num_classes=...)` and raises `RuntimeError` if timm cannot create it. Any timm single-channel-capable backbone therefore needs **no code change**.

- [ ] **1. Add a config YAML in `configs/supervised/`** (it is deep-merged over `configs/base.yaml`; do not add `defaults:`)

  ```yaml
  # configs/supervised/efficientnet_b0.yaml
  model:
    backbone: efficientnet_b0
    pretrained: true
    num_classes: 2
  training:
    learning_rate: 5.0e-5
    epochs: 100
  ```

  Copy the `lodo:`, `hitfinder:`, `asymmetric:`, `benchmark:` and `cache:` blocks from `configs/supervised/resnet18_asymmetric.yaml`, and note the cache default (see the frame-cache section).

- [ ] **2. Add a test in `tests/test_models.py`** (existing tests: `test_resnet18_output_shape`, `test_invalid_backbone_raises`, …)

  ```python
  def test_build_efficientnet_b0():
      model = build_supervised_model("efficientnet_b0", pretrained=False, num_classes=2)
      assert model(torch.zeros(2, 1, 224, 224)).shape == (2, 2)
  ```

- [ ] **3. Launch with a versioned run name.** Track 1 prefixes must match `<backbone>-asymmetric-v<N>` (`run_naming.ASYMMETRIC_PREFIX_RE`), so `--run-name-prefix efficientnet-asymmetric-v1`. The prefix is only a label: it allows lowercase letters and digits before `-asymmetric-`, so drop underscores and hyphens from a timm name (`efficientnet_b0` → `efficientnet`); the backbone actually trained is `model.backbone` in the config.

For a new **self-supervised** backbone, `src/models/ssl.py` (`MAEViT`, `ViTClassifier`, `build_mae_model`, `build_ssl_classifier`) is fixed to ViT-S/16; changing the architecture changes the Track 1 / Track 2 comparison and needs design review.

---

## New Evaluation Metric {#new-metric}

Add a metric that is computed alongside AP, AUC-ROC, and F1 after vote aggregation.

- [ ] **1. Add function to `src/evaluation/metrics.py`**

  ```python
  def my_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
      """
      Args:
          y_true:  (N,) int array of ground-truth labels {0, 1}
          y_score: (N,) float array of predicted scores in [0, 1]
      Returns:
          scalar metric value
      """
      if len(np.unique(y_true)) < 2:
          return 0.0
      return float(value)
  ```

- [ ] **2. Wire into the return dicts in `src/evaluation/benchmark.py`**

  Both `run_on_loader()` and `run_patch_agg()` return `{"ap", "auc_roc", "f1", "threshold"}` (and a NaN version for empty input). Add the key to each, including the all-NaN fallback:

  ```python
  return {
      "ap":        average_precision(y_true, y_score),
      "auc_roc":   auc_roc(y_true, y_score),
      "f1":        best_f1,
      "threshold": threshold,
      "my_metric": my_metric(y_true, y_score),      # ← add here
  }
  ```

- [ ] **3. Carry it into the result files.** `src/training/lodo.py` copies `ap`, `auc_roc`, `f1`, `threshold` explicitly into the `cross` and `in_domain` blocks of `results.json`, and `summary_updates()` in `src/training/inference_results.py` builds the `inference/*` W&B keys. Add the new key in both places, and in `scripts/aggregate_lodo_results.py` if it should appear in the tables. Remember that an F1-style metric computed at the evaluated split's own optimum is a best-case figure (see [eval_protocol.md](../eval_protocol.md)).

- [ ] **4. Add test in `tests/test_evaluation.py`**

  ```python
  def test_my_metric_perfect():
      y_true  = np.array([0, 0, 1, 1])
      y_score = np.array([0.1, 0.2, 0.8, 0.9])
      assert my_metric(y_true, y_score) == pytest.approx(1.0, abs=0.01)

  def test_my_metric_no_positives():
      assert my_metric(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) == 0.0
  ```

---

## New SSL Pretraining Config {#new-ssl-config}

Add a hyperparameter variant for MAE pretraining (different mask ratio, crops per frame, or learning rate). The MAE backbone is fixed at ViT-S/16 — this recipe is for config variants only, not new architectures. The pretraining config reads the `ssl:`, `training:` and `lodo:` blocks; there is no `data.file_list` (sessions come from `lodo.detector_dirs`).

- [ ] **1. Create a YAML file under `configs/ssl/`** — copy `configs/ssl/mae_pretrain.yaml` and change what you need:

  ```yaml
  # configs/ssl/mae_pretrain_mask75.yaml
  ssl:
    masking: random          # random | peak_aware
    mask_ratio: 0.75         # default 0.6
    norm_pix_loss: false
    crops_per_frame: 4
    min_valid_frac: 0.5
  training:
    epochs: 400
    warmup_epochs: 20
    learning_rate: 1.5e-4
    batch_size: 256
  ```

  Keep the unchanged `model:`, `lodo:`, `hitfinder:` and `cache:` blocks too. Anything omitted falls back to `configs/base.yaml`.

- [ ] **2. Verify `load_config()` merges correctly**

  ```python
  from src.utils.config import load_config
  cfg = load_config("configs/ssl/mae_pretrain_mask75.yaml")
  assert cfg["ssl"]["mask_ratio"] == 0.75
  assert "training" in cfg
  ```

- [ ] **3. Launch pretraining with a versioned run name**

  ```bash
  source .secrets/wandb.env
  python -u -m src.training.train_ssl_pretrain \
      --config configs/ssl/mae_pretrain_mask75.yaml \
      --fold 1 \
      --run-name-prefix mae-vits16-v3   # required; matches mae-<backbone>-v<N>
  ```

  For all folds use `bash scripts/submit_ssl_pretrain_all.sh --run-name-prefix mae-vits16-v3`. Fine-tune with the matching `--pretrain-checkpoint checkpoints/mae-vits16-v3-fold1-seed42/last.pt`.

- [ ] **4. Add a config-load test in `tests/test_config.py`** (existing tests build a temporary YAML and call `load_config(path)`):

  ```python
  def test_ssl_config_loads_mask_ratio(tmp_path):
      cfg_path = tmp_path / "mae_pretrain_mask75.yaml"
      cfg_path.write_text("ssl:\n  mask_ratio: 0.75\n")
      cfg = load_config(str(cfg_path))
      assert cfg["ssl"]["mask_ratio"] == pytest.approx(0.75)
  ```

---

## Build or Invalidate the Frame Cache {#frame-cache}

The frame cache (`src/data/frame_cache.py`) holds the read → assemble → hitfinder → GCN prefix per frame. It is opt-in per run through the `cache:` block (`enabled: false` in `configs/base.yaml`; the production LODO configs set `true`).

- [ ] **Build:** `python -m scripts.build_frame_cache --config configs/supervised/resnet18_asymmetric.yaml [--cache-root <path>] [--detectors AGIPD ePix10k] [--workers 32] [--overwrite]`. Entries are written per CXI file as float16 arrays; the manifest is written last, so an interrupted build looks unbuilt.
- [ ] **Stage to NVMe for a fold:** `sbatch scripts/stage_frame_cache.sh <fold_id>` is a SLURM job (node `scg020`). It calls `scripts/plan_cache_staging.py` itself for the fold's priority order (train, then val), copies entries with `cp -r` until free space hits a reserve, and reads env `CONFIG` (default `configs/ssl/mae_finetune.yaml`), `CACHE_NFS` and `CACHE_NVME`. Partial staging is expected for fold 1; the rest streams from NFS. `submit_asymmetric_lodo_all.sh` does this per fold.
- [ ] **What invalidates it:** `verify_cache_or_raise()` compares the manifest with the live config and raises `CacheStaleError` naming the first differing key — hitfinder parameters (`_HITFINDER_KEYS`), GCN/LCN constants, `EDGE_EROSION_PX`, the cache dtype, the hash of every `.geom`/`.json` file in `src/preprocessing/data/`, and `PIPELINE_VERSION`. Not covered: Reborn-bundled geometry changes, and changes to *how* a detector's assembler is chosen — bump `PIPELINE_VERSION` and rebuild.
- [ ] **Missing entries are not errors.** A `CacheMissError` makes the dataset compute the frame live; a partially staged NVMe tier is therefore slower, not wrong.
- [ ] **Test:** extend `tests/test_frame_cache.py` (manifest staleness, two-tier resolution, `TestBitExactnessAgainstRealData`, which is skipped when no real cache exists).

---

## New or Changed SLURM Submit Script {#submit-scripts}

Start from `scripts/templates/slurm_gpu_job.sh` (partition `general`, QOS `grp_cxfel`, `gpu:h100:1`, node `scg020`) and `source .secrets/wandb.env` before the training command.

- The SSL orchestrators re-implement two Python rules in bash so a typo fails at submit time: the run-name prefix regexes and the top-level seed lookup. If you change `run_naming.SSL_PRETRAIN_PREFIX_RE`, `SSL_FINETUNE_PREFIX_RE` or the top-level `seed:` key in the configs (the scripts grep it from the config and `configs/base.yaml`; the test compares it with `load_config(...)["seed"]`), update `submit_ssl_pretrain_all.sh` / `submit_ssl_finetune_all.sh`; `tests/test_submit_script_parity.py` fails if the copies drift.
- Retired scripts must exit immediately and name their replacement; add them to the table in `tests/test_deprecated_scripts.py`.
- Queued SLURM jobs read the **live checkout** when they start. Do not leave the main checkout on an older branch while jobs are pending.

---

## Re-evaluate a Finished Run {#inference-only}

Evaluate an existing `best.pt` on its in-domain and cross-detector sets without training — for example to complete a run whose evaluation crashed, or to re-score with a changed `benchmark` setting. `--inference-only` is the third mode next to `--resume-training` and `--override-training` (mutually exclusive) and needs `best.pt` to exist. A plain rerun on a finished fold still exits.

- If the run has no `results.json`, it is written (this completes the run). If one exists it is kept untouched and the new numbers go to `results.inference.json`. Either way the file carries an `inference` block with the aggregation, stride, checkpoint epoch and a `training_check`. `scripts/aggregate_lodo_results.py` reads only `results.json`.
- `training_check` tells you whether the run behind `best.pt` actually finished. It reads the run's W&B history read-only (the run is not resumed) and records the W&B state, the last logged epoch against the configured epochs, and whether the train loss had flattened (improved by less than 1% over the last 5 logged epochs). A run that did not finish prints `[inference] WARNING` lines first: `status: "incomplete"` means the numbers are provisional. If W&B is offline or unreachable the status is `unknown` and only the epoch counts are compared.
- W&B (when enabled) gets run-summary entries under `inference/…` only; training curves are not touched.
- MAE pretraining has no `best.pt` and no evaluation, so it has no inference mode.

```bash
# Track 1
python -m src.training.train_asymmetric --config configs/supervised/resnet18_asymmetric.yaml \
    --run-name-prefix resnet18-asymmetric-v2 --folds 2 --inference-only

# Track 2 fine-tune / probe (no --pretrain-checkpoint needed)
python -u -m src.training.train_ssl_finetune --config configs/ssl/mae_finetune.yaml \
    --fold 1 --run-name-prefix vits16-mae-v2 --inference-only
```

The orchestrators (`submit_asymmetric_lodo_all.sh`, `submit_ssl_finetune_all.sh`) offer it as the `inference` answer when `best.pt` already exists.

## W&B Run Ids After `--override-training` {#override-wandb-ids}

Logging to an existing W&B run id with `step=epoch` after an override loses data (measured on
wandb 0.27.0): a shorter second attempt is dropped entirely, a longer one loses its first
epochs, and a deleted run's id can never be reused. So each override gets a new run id:

| Attempt | W&B id | Display name |
|---|---|---|
| first (and every legacy run) | `<run_name>` | `<run_name>` |
| after one override | `<run_name>-o2` | `<run_name>` |
| after two overrides | `<run_name>-o3` | `<run_name>` |

The live id is stored in `checkpoints/<run_name>/wandb_id.txt` (absent = id equals the run
name). `src/training/wandb_identity.py` owns it: `check_checkpoint_collisions` calls an
`OverrideHook` before deleting anything; the hook rotates the id and tags the previous run
`overridden` (kept, not deleted). Resume, `--inference-only` and the completeness check all
read the file through `resolve_wandb_id`. Several W&B runs can share one display name, so
tools should filter on the `overridden` tag (`scripts/plot_hit_frac.py` does) or read the
file; never group by name alone.

Behaviour worth knowing:

- **Failed delete rolls back.** The hook returns a rollback callable. If deleting the
  checkpoint or its extras raises, `check_checkpoint_collisions` restores `wandb_id.txt`
  (or removes it if it did not exist), best-effort removes the `overridden` tag from the old
  run, and re-raises. The checkpoint is deleted last, so it survives for a rerun.
- **Plots pick the highest attempt per name, tag or not.** `select_current_runs` drops
  `overridden`-tagged runs, then keeps only the highest attempt (`<name>` = 1,
  `<name>-oN` = N) among runs sharing a display name. Offline mode, where the old run cannot be
  tagged, is therefore covered; `scripts/plot_hit_frac.py` uses it.
- **A fresh start never reuses an existing W&B run id.** Before the training `wandb.init`,
  `wandb_id_for_training` (Track 1 `_train_fold`, SSL pretrain) asks W&B whether the current id
  exists; if so it tags that run `overridden` and rotates past it (e.g. after `rm -rf` of the
  checkpoint dir, or a crash before `best.pt`). Genuine resumes and inference keep the current
  id. If W&B cannot be queried, the current id is kept with a warning.
- **Track 1 override is per fold.** `train_asymmetric` runs the gate once up front with
  `dry_run=True` (fail fast, nothing deleted or rotated) and the real gate for each fold just
  before that fold starts, so a crash on fold 1 leaves fold 2's checkpoint and id untouched.
  `_train_fold` resolves the id after that real pass; keep that order.

### Override is applied once per SLURM job {#override-once}

`--override-training` deletes the existing checkpoint, so re-running the same launch after
the new attempt wrote one would destroy it. A requeued SLURM job keeps its `SLURM_JOB_ID` and
gets a higher `SLURM_RESTART_COUNT` (a fresh `sbatch` gets a new job id), so the gate keys on
both:

- After a run is overridden, `check_checkpoint_collisions` writes
  `checkpoints/<run_name>/override_applied.json` (`job_id`, `restart_count`, `run_name`,
  `applied_at`). The order is rotate the W&B id, delete the checkpoint, then write the marker.
  If the marker write fails the gate warns on stderr and carries on (the delete already
  happened, so failing protects nothing); a requeue of that job would then override again.
- A **requeue** (same job id, higher restart count) prints `[override] already applied by
  job N ...`, keeps the checkpoint, does not rotate the W&B id again, and returns the run in
  a `continued` set. This is checked before looking for the checkpoint, so a requeue that
  lands after the delete but before the first save is also continued. Every entry point turns
  the set into a resume with `effective_resume(...)` (Track 1 per fold and intra, SSL
  fine-tune, SSL pretrain), and a resume reuses the current W&B id even with no checkpoint.
- Same job id and same restart count (two launches in one `salloc` / `srun --pty`
  allocation) is a deliberate second override and applies again, as does a different job id.
  With no `SLURM_JOB_ID` (a direct `python -m ...` launch) the flag is honoured every time
  and no marker is written.
- A requeue that happens before the gate ever ran finds no marker and applies the override,
  so a requested override is no longer silently turned into a resume.
- The job scripts no longer rewrite `RESUME_FLAG` on a requeue; they pass it through and
  Python decides.

Known limit (accepted): the marker is keyed on the launch, not on the checkpoint, and is never
cleared. Copying a different checkpoint into the run directory and requeueing the same job
resumes from that file, and a recycled job id (cluster counter reset) could skip a requested
override.

Limit: a manual re-`sbatch` of the same command is a new job id, hence a new, deliberate
override. Code: `src/training/override_marker.py`, `check_checkpoint_collisions` and
`effective_resume` in `src/training/run_naming.py`.
