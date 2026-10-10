# Extension Guide

Step-by-step recipes for the four most common extension points.

---

## New Detector {#new-detector}

Add support for a detector geometry not yet in the system.

- [ ] **1. Add geometry loader to `src/preprocessing/geometry.py`**

  Add a branch in `get_geometry()` for the new detector description string:

  ```python
  elif desc == "MY_DETECTOR":
      return load_pad_geometry(cxi_path, geom_file="src/preprocessing/data/my_detector.geom")
  ```

- [ ] **2. Add `.geom` file (if CrystFEL-style geometry)**

  Place it at `src/preprocessing/data/my_detector.geom`. This directory co-locates geometry files with the preprocessing code that loads them. CrystFEL geometry format:
  `p0/corner_x`, `p0/corner_y`, `p0/fs`, `p0/ss`, etc.

- [ ] **3. Add to `DETECTOR_LOADERS` in `src/preprocessing/geometry.py`**

  ```python
  DETECTOR_LOADERS = {
      "AGIPD":       _load_agipd,
      "JUNGFRAU_4M": _load_crystfel,
      "ePix10k":     _load_crystfel,
      "Eiger4M":     _load_eiger,
      "MY_DETECTOR": _load_crystfel,   # ← add here
  }
  ```

- [ ] **4. Add to LODO fold definition in `src/evaluation/benchmark.py`**

  ```python
  DETECTORS: list[str] = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M", "MY_DETECTOR"]
  ```

  This automatically creates a new fold in `build_lodo_folds()`.

- [ ] **5. Add smoke-test entry in `scripts/smoke_test_detector_shapes.py`**

  Add a dict entry with a sample CXI path and expected assembled shape.

- [ ] **6. Add test in `tests/test_geometry_assembly.py`**

  ```python
  def test_my_detector_assembly():
      frame = np.zeros((512, 512), dtype=np.float32)   # native shape
      geom = get_geometry("MY_DETECTOR", cxi_path=None)
      assembler = get_assembler(geom)
      canvas = assembler.assemble_image(frame)
      assert canvas.ndim == 2
      assert canvas.shape[0] > 0
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
          # your implementation here
          centroids = []
          return np.array(centroids, dtype=np.float32).reshape(-1, 2)
  ```

- [ ] **2. Register in `src/hitfinders/__init__.py`**

  ```python
  from .my_finder import MyHitfinder

  def get_hitfinder(name: str, **kwargs):
      backends = {
          "pf8":       PF8Hitfinder,
          "numpy":     NumpyPF8Hitfinder,
          "gpu":       GPUHitfinder,
          "mock":      MockHitfinder,
          "my_finder": MyHitfinder,   # ← add here
      }
      ...
  ```

- [ ] **3. Add config key in `configs/supervised/resnet18_resonet.yaml`**

  ```yaml
  hitfinder:
    backend: my_finder
    my_finder_threshold: 100.0
  ```

- [ ] **4. Add test in `tests/test_hitfinders.py`**

  ```python
  def test_my_hitfinder_returns_correct_shape():
      hf = MyHitfinder(threshold=100.0)
      frame = np.random.rand(512, 512).astype(np.float32)
      peaks = hf.find_peaks(frame)
      assert peaks.ndim == 2
      assert peaks.shape[1] == 2

  def test_get_hitfinder_my_finder():
      hf = get_hitfinder("my_finder", threshold=100.0)
      assert isinstance(hf, MyHitfinder)
  ```

---

## New Model / Backbone {#new-model}

Add a new backbone or architecture variant to the supervised track.

- [ ] **1. Add a branch in `src/models/supervised.py`**

  ```python
  def build_supervised_model(backbone: str = "resnet18", pretrained: bool = True, num_classes: int = 2):
      if backbone in ("resnet18", "resnet50"):
          model = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=num_classes)
      elif backbone == "efficientnet_b0":
          model = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=1, num_classes=num_classes)
      else:
          raise ValueError(f"Unknown backbone: {backbone}")
      return model
  ```

- [ ] **2. Add a config YAML in `configs/supervised/`**

  ```yaml
  # configs/supervised/efficientnet_b0.yaml
  model:
    backbone: efficientnet_b0
    pretrained: true
    num_classes: 2
  training:
    learning_rate: 5.0e-5
    num_epochs: 100
  ```

- [ ] **3. Add to `scripts/smoke_test_detector_shapes.py` model list**

  Add `"efficientnet_b0"` to the list of backbones the smoke test exercises.

- [ ] **4. Add test in `tests/test_models.py`**

  ```python
  def test_build_efficientnet_b0():
      model = build_supervised_model("efficientnet_b0", pretrained=False, num_classes=2)
      x = torch.zeros(2, 1, 224, 224)
      out = model(x)
      assert out.shape == (2, 2)
  ```

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
      # your implementation
      return float(value)
  ```

- [ ] **2. Wire into `run_patch_agg()` results dict in `src/evaluation/benchmark.py`**

  Find the return dict in `run_patch_agg()` and add:

  ```python
  from src.evaluation.metrics import my_metric
  ...
  return {
      "ap":        average_precision(y_true, y_score),
      "auc_roc":   auc_roc(y_true, y_score),
      "f1":        f1_at_optimal_threshold(y_true, y_score)[0],
      "threshold": f1_at_optimal_threshold(y_true, y_score)[1],
      "my_metric": my_metric(y_true, y_score),      # ← add here
  }
  ```

- [ ] **3. Add test in `tests/test_evaluation.py`**

  ```python
  def test_my_metric_perfect():
      y_true  = np.array([0, 0, 1, 1])
      y_score = np.array([0.1, 0.2, 0.8, 0.9])
      assert my_metric(y_true, y_score) == pytest.approx(1.0, abs=0.01)

  def test_my_metric_no_positives():
      y_true  = np.array([0, 0, 0])
      y_score = np.array([0.1, 0.2, 0.3])
      assert my_metric(y_true, y_score) == 0.0
  ```

---

## New SSL Pretraining Config {#new-ssl-config}

Add a hyperparameter variant for MAE pretraining (different mask ratio, patch size, or learning rate). The MAE backbone is fixed at ViT-S/16 — this recipe is for config variants only, not new architectures.

- [ ] **1. Create a YAML file under `configs/ssl/`**

  ```yaml
  # configs/ssl/mae_pretrain_v2.yaml
  model:
    mask_ratio: 0.80          # fraction of patches masked (default: 0.75)
    patch_size: 16            # ViT patch size in pixels
  training:
    learning_rate: 1.5e-4
    num_epochs: 200
    batch_size: 64
    warmup_epochs: 10
  data:
    file_list: data/splits/ssl_train.txt   # plaintext list of .img or .cxi paths
  ```

  Required keys: `model.mask_ratio`, `model.patch_size`, `training.learning_rate`, `training.num_epochs`, `data.file_list`. All other keys are optional and fall back to `configs/base.yaml` via `load_config()`.

- [ ] **2. Verify `load_config()` merges correctly**

  ```python
  from src.utils.config import load_config
  cfg = load_config("ssl/mae_pretrain_v2")
  assert cfg["model"]["mask_ratio"] == 0.80
  assert "training" in cfg
  ```

- [ ] **3. Launch pretraining with the new config**

  ```bash
  source .secrets/wandb.env
  python -m src.training.train_ssl_pretrain \
      --config configs/ssl/mae_pretrain_v2.yaml \
      --fold 1 \
      --run-name-prefix mae-vits16-v2   # required; mae-<backbone>-v<N>
  ```

- [ ] **4. Add a config-load test in `tests/test_config.py`**

  ```python
  def test_ssl_config_loads_mask_ratio(tmp_path):
      cfg_path = tmp_path / "ssl" / "mae_pretrain_v2.yaml"
      cfg_path.parent.mkdir()
      cfg_path.write_text("model:\n  mask_ratio: 0.80\n  patch_size: 16\n")
      cfg = load_config(str(cfg_path))
      assert cfg["model"]["mask_ratio"] == pytest.approx(0.80)
  ```

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
