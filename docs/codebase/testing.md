# Testing Guide

## Running Tests

```bash
# Full suite
pytest

# Single file
pytest tests/test_evaluation.py -v

# Single test
pytest tests/test_evaluation.py::test_average_precision_perfect -v

# CI subset (no GPU, no heavy I/O)
pytest -m "not slow" --tb=short
```

CI uses `requirements-ci.txt`. Install with:
```bash
pip install -r requirements-ci.txt
pytest
```

---

## Test File Map

| Test file | What it covers | Key fixture / helper |
|-----------|---------------|----------------------|
| `test_augmentation.py` | `augment.py`: rot90, flip, cutout, pad_border, patch_grid | `np.random.default_rng()` |
| `test_normalize.py` | `normalize.py`: GCN zero-mean, LCN local stats | synthetic arrays |
| `test_io.py` | `io.py`: read_frame, read_embedded_labels, count_frames | tmp HDF5 files |
| `test_geometry_assembly.py` | `geometry.py`: panel assembly for each detector | mock geometry objects |
| `test_pipeline.py` | `pipeline.py`: assemble_only, fill_gaps_after_gcn, preprocess_eval_patches | synthetic frames |
| `test_preprocessing.py` | Integration: full preprocessing from raw frame to patch | synthetic CXI |
| `test_hitfinders.py` | All hitfinder backends + `get_hitfinder()` factory + MockHitfinder | `MockHitfinder` |
| `test_dataset.py` | `UnlabeledDataset`, `MultiFrameCXIDataset` | synthetic CXI + `MockHitfinder` |
| `test_asymmetric_dataset.py` | `AsymmetricCXIDataset`: centroid crops, miss crops, peak-aware cutout | `MockHitfinder` |
| `test_models.py` | `build_supervised_model()` for resnet18/50 | `torch.zeros()` |
| `test_train_supervised.py` | `train_one_epoch()`: loss decreases, returns metrics dict | small synthetic loader |
| `test_train_asymmetric_resume.py` | Resume-from-checkpoint logic in `src/training/train_asymmetric.py` | tmp checkpoint dir |
| `test_evaluation.py` | `metrics.py`: AP, AUC-ROC, F1 edge cases | `np.array` fixtures |
| `test_patch_eval.py` | Patch-level centroid crop geometry | synthetic canvas + centroids |
| `test_vote_aggregation.py` | Vote aggregation: frame score formula, threshold behavior | synthetic patch scores |
| `test_config.py` | `load_config()`: base merge, unknown key error | tmp YAML files |

---

## Key Fixtures

**`conftest.py`** (repo root) adds `src/` to `sys.path` — no other fixtures defined. Each test file creates its own synthetic data.

**`MockHitfinder`** (`src/hitfinders/base.py`) — returns a fixed set of centroids. Use it any time a test needs a hitfinder without running PF8:

```python
from src.hitfinders import MockHitfinder
hf = MockHitfinder(peaks=np.array([[100, 100], [200, 200]], dtype=np.float32))
centroids = hf.find_peaks(frame)   # always returns the fixed peaks
```

---

## Adding a New Test

1. **Pick the right file.** Each test file mirrors one `src/` module. If your code is in `src/preprocessing/normalize.py`, the test goes in `tests/test_normalize.py`.

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
   dataset = AsymmetricCXIDataset(sessions, cxi_paths, hitfinder=hf, ...)
   ```

5. **Mark slow tests** (anything involving real HDF5 I/O or model forward pass) with `@pytest.mark.slow` so they can be excluded from fast CI runs.
