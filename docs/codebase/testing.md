# Testing Guide

699 tests across 30 files (counted with `pytest tests/ --collect-only -q` on 2026-10-22). Test counts per file are in the map below.

## Running Tests

```bash
# Full suite (what CI runs)
pytest tests/ -v

# Single file
pytest tests/test_evaluation.py -v

# Single test
pytest tests/test_evaluation.py::test_average_precision_perfect -v

# Single class
pytest tests/test_normalize.py::TestNormalizationOrder -v
```

There is no pytest configuration file and **no custom markers** — the only markers in use are `parametrize` and `skipif`. Do not rely on `-m "not slow"`; no test carries that marker. Tests that need real data or an existing cache skip themselves (for example `tests/test_frame_cache.py::TestBitExactnessAgainstRealData`, which needs the real frame cache on disk).

### CI

`.github/workflows/ci.yml` runs on pushes to `main` and `phase-NN` branches and on pull requests to `main` and `phase-**`, on Python 3.11 with `ubuntu-latest`:

0. `apt-get install libgl1 libglib2.0-0` (system libraries for the Qt/plotting dependencies); a `concurrency` group cancels a superseded run on the same ref
1. `pip install -r requirements-ci.txt` — CPU-only torch/torchvision, h5py, numpy, pyyaml, timm, wandb, fabio, black, pytest, PyQt5, pyqtgraph, scipy, matplotlib, scikit-image, pyFAI, and Reborn from the `gihan` branch of the Kirian Lab GitLab repo
2. `black --check src/ tests/`
3. `pytest tests/ -v`

So **run `black src/ tests/` before pushing** or CI fails before the tests start. A separate workflow, `.github/workflows/dashboard.yml`, runs only for changes under `dashboard/**` (or to the workflow file itself) on pushes to `main` and `phase-NN` and on pull requests to `main` and `phase-**`: Node 24, then `npm ci`, `npm run typecheck`, `npm test` and `npm run build` in `dashboard/`. The dashboard has its own suites (about 1440 vitest tests in 80 files, plus 60 Playwright end-to-end tests run locally with `npm run e2e`); the Python suite does not depend on it.

Local setup without the full environment:

```bash
pip install -r requirements-ci.txt
pytest tests/ -v
```

---

## Test File Map

| Test file | Tests | What it covers | Key fixture / helper |
|-----------|-------|---------------|----------------------|
| `test_augmentation.py` | 20 | `augment.py`: rot90, flip, cutout, pad_border, patch_grid | `np.random.default_rng()` |
| `test_normalize.py` | 25 | `normalize.py`: GCN zero-mean, LCN local stats, masked LCN, `lcn_torch` parity | synthetic arrays |
| `test_io.py` | 29 | `io.py`: read_frame, read_embedded_labels, count_frames | tmp HDF5 files |
| `test_geometry_assembly.py` | 10 | `geometry.py`: panel assembly, detector description reading | mock geometry objects |
| `test_pipeline.py` | 12 | `pipeline.py`: assemble_only, fill_gaps_after_gcn, preprocess_eval_patches | synthetic frames |
| `test_preprocessing.py` | 48 | Integration: full preprocessing from raw frame to patch | synthetic CXI |
| `test_hitfinders.py` | 23 | All hitfinder backends + `get_hitfinder()` factory + MockHitfinder | `MockHitfinder` |
| `test_dataset.py` | 14 | `UnlabeledDataset`, `MultiFrameCXIDataset` (deprecated), loaders | synthetic CXI + `MockHitfinder` |
| `test_asymmetric_dataset.py` | 28 | `AsymmetricCXIDataset`: centroid crops, miss crops, peak-aware cutout, `crops_per_frame`, epoch-varying crops | `MockHitfinder` |
| `test_frame_cache.py` | 45 | `_compute_gcn_frame` / `_load_gcn_frame` split, manifest staleness, two-tier resolution, cache-from-config, LODO wiring, bit-exactness gate (skipped without a real cache) | synthetic CXI fixture |
| `test_models.py` | 4 | `build_supervised_model()` for resnet18/50 | `torch.zeros()` |
| `test_train_supervised.py` | 3 | `train_one_epoch()`: loss decreases, returns metrics dict | small synthetic loader |
| `test_train_asymmetric_resume.py` | 12 | Resume-from-checkpoint logic in `src/training/train_asymmetric.py` | tmp checkpoint dir |
| `test_run_naming.py` | 85 | Run-name prefix patterns, fold run names, checkpoint collision / resume / override gate | tmp checkpoint dir |
| `test_override_marker.py` | 11 | `override_applied.json`: a requeued SLURM job resumes instead of overriding again | `SLURM_JOB_ID` / `SLURM_RESTART_COUNT` env |
| `test_wandb_identity.py` | 33 | W&B run id file, rotation (`-o2`, `-o3`), `overridden` tagging | faked `wandb` |
| `test_wandb_isolation.py` | 3 | No test may reach W&B by accident (`wandb_enabled()` creates a process-wide singleton) | — |
| `test_training_block.py` | 18 | `results.json` `training` block (`configured_epochs`, `last_epoch`, `best_epoch`, `early_stopped`) driven by scripted validation F1 scores | tiny fake fold |
| `test_inference_only.py` | 64 | `--inference-only`: result helpers, the `_train_fold` inference path, CLI wiring | faked `wandb`; the training loader raises if built |
| `test_orchestrator_inference.py` | 30 | Orchestrator scripts' `inference` answer, run against a stub `sbatch` | temp dir with fake `sbatch` on `PATH` |
| `test_ssl_dataset.py` | 13 | `SSLPretrainCXIDataset` — random valid-region crops for MAE pretraining | synthetic CXI + valid-pixel mask |
| `test_ssl_model.py` | 22 | `src/models/ssl.py` — MAE ViT-S/16 builder, masking, pixel MSE loss | `torch.zeros()` |
| `test_train_ssl.py` | 42 | SSL pretraining and fine-tuning loops (CPU smoke test, `MockHitfinder`) | small synthetic loader |
| `test_ssl_main_requeue.py` | 6 | A requeued SSL launch through the real `main()` entry points | tmp run dir |
| `test_submit_script_parity.py` | 10 | The bash copies of the run-name prefix regexes and the seed lookup in the SSL submit scripts match `run_naming.py` and `load_config()` | — |
| `test_deprecated_scripts.py` | 23 | The seven legacy v1 submit scripts refuse to run and name their replacement | temp dir with stub `sbatch` / `python` |
| `test_evaluation.py` | 34 | `metrics.py`: AP, AUC-ROC, F1 edge cases; `benchmark.py` helpers | `np.array` fixtures |
| `test_patch_eval.py` | 22 | Patch-level centroid crop geometry, patch-grid evaluation | synthetic canvas + centroids |
| `test_vote_aggregation.py` | 4 | Vote aggregation: frame score formula, threshold behavior | synthetic patch scores |
| `test_config.py` | 6 | `load_config()`: base merge, model values win, missing file | tmp YAML files |

Some files test behaviour that spans modules or scripts rather than mirroring one `src/` module (the run-lifecycle, orchestrator and submit-script tests).

---

## Key Fixtures

**`conftest.py` (repo root)** only inserts the project root into `sys.path` so tests can `from src... import`.

**`tests/conftest.py`** defines shared fixtures, two of them `autouse`:

- `wandb_disabled` (autouse) — sets `WANDB_MODE=disabled` for every test and calls `wandb.teardown()` afterwards, so W&B's process-wide singleton cannot freeze settings for later tests (CI has no API key); a test that needs another mode sets or deletes `WANDB_MODE` itself
- `fake_jungfrau_assembly_via_dataset` (autouse) — monkeypatches `src.data.dataset.assemble_only` so that Jungfrau frames whose shape is not (2164, 2068) go through a `_to_2d()` passthrough (many tests use undersized synthetic Jungfrau frames that real assembly would reject); every other frame still goes to the real `assemble_only`
- `fake_jungfrau_assembly_via_pipeline` (opt-in) — the same patch applied to `src.preprocessing.pipeline.assemble_only`

Other than these, each test file builds its own synthetic data.

**`MockHitfinder`** (`src/hitfinders/base.py`, exported from `src.hitfinders`) — returns a fixed set of centroids. Use it any time a test needs a hitfinder without running PF8:

```python
from src.hitfinders import MockHitfinder
hf = MockHitfinder(peaks=np.array([[100, 100], [200, 200]], dtype=np.float32))
centroids = hf.find_peaks(frame)   # always returns the fixed peaks (copy)
```

---

## Adding a New Test

1. **Pick the right file.** Most test files mirror one `src/` module. If your code is in `src/preprocessing/normalize.py`, the test goes in `tests/test_normalize.py`. For run lifecycle (naming, override, W&B ids, inference-only) or shell scripts, use the matching file in the map above.

2. **Use synthetic data, not real CXI files.** Create minimal arrays:

   ```python
   def test_my_function():
       frame = np.zeros((512, 512), dtype=np.float32)
       frame[100, 100] = 1000.0    # one bright pixel
       result = my_function(frame)
       assert result.shape == (512, 512)
   ```

3. **Test the failure case first.** Write a test that asserts the behavior you want *before* implementing it.

4. **Use `MockHitfinder` instead of PF8** to avoid needing `_pf8_wrap.so` in CI:

   ```python
   hf = MockHitfinder(peaks=np.array([[112, 112]], dtype=np.float32))
   dataset = AsymmetricCXIDataset(session_ids, session_map, hitfinder=hf, ...)
   ```

5. **Never touch W&B, SLURM, or a real cache from a test.** W&B is disabled by the autouse fixture (fake it if you need to inspect calls); shell scripts are run in a temp directory with stub `sbatch` / `python` first on `PATH`; anything needing a real cache must `pytest.skip` when the cache is absent, as `TestBitExactnessAgainstRealData` does.

6. **There is no `slow` marker.** If a test is expensive, make it skip itself when its prerequisites (real data, GPU) are missing rather than inventing a marker.

7. **Run `black src/ tests/` before committing** — CI runs `black --check` first.
