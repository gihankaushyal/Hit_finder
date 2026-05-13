# Cross-Detector Benchmark Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/evaluation/metrics.py` and `src/evaluation/benchmark.py` with full test coverage, add benchmark config to `base.yaml`, and write the formal `docs/eval_protocol.md` document.

**Architecture:** Pure-numpy metrics module then fold/split logic in benchmark.py then DataLoader-based fold runner that accepts a factory callable (decoupled from data loading so it can be tested with synthetic data now and wired to real HDF5 datasets in Phase 3).

**Tech Stack:** Python 3.11, NumPy, PyTorch (for DataLoader interface only), PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-05-13-benchmark-protocol-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/evaluation/metrics.py` | Modify (currently stub) | `average_precision`, `auc_roc`, `f1_at_optimal_threshold` |
| `src/evaluation/benchmark.py` | Modify (currently stub) | Fold defs, split builder, artifact I/O, fold runner, aggregator, baseline formatter |
| `tests/test_evaluation.py` | Create | All evaluation tests |
| `configs/base.yaml` | Modify | Add `benchmark:` section |
| `data/splits/.gitkeep` | Create | Track directory in git |
| `docs/eval_protocol.md` | Modify (currently stub) | Human-readable protocol reference |

---

## Task 1: Metrics Module

**Files:**
- Modify: `src/evaluation/metrics.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluation.py`:

```python
import numpy as np
import pytest
from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold


def test_average_precision_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert abs(average_precision(y_true, y_score) - 1.0) < 1e-6


def test_average_precision_known_value():
    # Sorted: (0.9,1), (0.8,0), (0.7,0), (0.6,1)
    # precision: [1.0, 0.5, 0.333, 0.5], recall: [0.5, 0.5, 0.5, 1.0]
    # recall_diff: [0.5, 0, 0, 0.5] -> AP = 1.0*0.5 + 0.5*0.5 = 0.75
    y_true = np.array([1, 0, 0, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    assert abs(average_precision(y_true, y_score) - 0.75) < 1e-6


def test_average_precision_no_positives():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.9, 0.5, 0.1])
    assert average_precision(y_true, y_score) == 0.0


def test_auc_roc_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert abs(auc_roc(y_true, y_score) - 1.0) < 1e-6


def test_auc_roc_inverted():
    # Worst possible predictor: every positive gets lower score than every negative
    y_true = np.array([0, 1])
    y_score = np.array([0.9, 0.1])
    assert abs(auc_roc(y_true, y_score) - 0.0) < 1e-6


def test_auc_roc_no_negatives():
    y_true = np.array([1, 1])
    y_score = np.array([0.9, 0.8])
    assert auc_roc(y_true, y_score) == 0.5


def test_f1_at_optimal_threshold_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    f1, thresh = f1_at_optimal_threshold(y_true, y_score)
    assert abs(f1 - 1.0) < 1e-6
    assert thresh == pytest.approx(0.8)


def test_f1_at_optimal_threshold_range():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_score = rng.uniform(0, 1, size=100)
    f1, thresh = f1_at_optimal_threshold(y_true, y_score)
    assert 0.0 <= f1 <= 1.0
    assert y_score.min() <= thresh <= y_score.max()
```

- [ ] **Step 2: Run tests to confirm FAIL**

```bash
pytest tests/test_evaluation.py -v
```

Expected: FAIL — `ImportError: cannot import name 'average_precision' from 'src.evaluation.metrics'`

- [ ] **Step 3: Implement metrics.py**

Replace `src/evaluation/metrics.py` entirely:

```python
"""Evaluation metrics for SFX hitfinder: AP, AUC-ROC, F1."""

from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (interpolation-free, sklearn-compatible).

    Returns 0.0 if there are no positive examples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    sorted_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[sorted_idx]

    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0

    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    recall = tp / n_pos
    recall_diff = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * recall_diff))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via trapezoidal integration.

    Returns 0.5 if all labels are the same class (AUC undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    sorted_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[sorted_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = np.r_[0.0, np.cumsum(y_sorted) / n_pos]
    fpr = np.r_[0.0, np.cumsum(1 - y_sorted) / n_neg]
    return float(np.trapz(tpr, fpr))


def f1_at_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """F1 score and the threshold that maximises it.

    Returns (best_f1, best_threshold). Iterates over all unique score values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(y_score)
    best_f1, best_thresh = 0.0, float(thresholds[0])

    for t in thresholds:
        y_pred = (y_score >= t).astype(float)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        denom_p = tp + fp
        denom_r = tp + fn
        if denom_p == 0 or denom_r == 0:
            continue

        p = tp / denom_p
        r = tp / denom_r
        if p + r == 0:
            continue

        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return float(best_f1), best_thresh
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/test_evaluation.py -v -k "average_precision or auc_roc or f1"
```

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_evaluation.py
git commit -m "feat: implement evaluation metrics (AP, AUC-ROC, F1)"
```

---

## Task 2: LODO Fold Definitions

**Files:**
- Modify: `src/evaluation/benchmark.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
from src.evaluation.benchmark import DETECTORS, build_lodo_folds


def test_detectors_constant():
    assert set(DETECTORS) == {"AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"}
    assert len(DETECTORS) == 4


def test_build_lodo_folds_count():
    folds = build_lodo_folds()
    assert len(folds) == 4


def test_build_lodo_folds_structure():
    folds = build_lodo_folds()
    for fold in folds:
        assert "fold_id" in fold
        assert "test_detector" in fold
        assert "train_detectors" in fold
        assert fold["test_detector"] not in fold["train_detectors"]
        assert len(fold["train_detectors"]) == 3


def test_build_lodo_folds_all_detectors_tested():
    folds = build_lodo_folds()
    tested = {f["test_detector"] for f in folds}
    assert tested == set(DETECTORS)


def test_build_lodo_folds_fold1():
    folds = build_lodo_folds()
    fold1 = next(f for f in folds if f["fold_id"] == 1)
    assert fold1["test_detector"] == "AGIPD"
    assert set(fold1["train_detectors"]) == {"JUNGFRAU_4M", "ePix10k", "Eiger4M"}
```

- [ ] **Step 2: Run tests to confirm FAIL**

```bash
pytest tests/test_evaluation.py -v -k "detectors or lodo_folds"
```

Expected: FAIL — `ImportError: cannot import name 'DETECTORS' from 'src.evaluation.benchmark'`

- [ ] **Step 3: Implement fold definitions in benchmark.py**

Replace `src/evaluation/benchmark.py` entirely:

```python
"""Cross-detector leave-one-detector-out evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import average_precision, auc_roc, f1_at_optimal_threshold

DETECTORS: list[str] = ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_IN_DOMAIN_TEST = "in_domain_test"
SPLIT_CROSS_DETECTOR = "cross_detector_eval"


def build_lodo_folds() -> list[dict]:
    """Return the 4 leave-one-detector-out fold definitions.

    Each fold: {"fold_id": int, "test_detector": str, "train_detectors": list[str]}
    """
    return [
        {
            "fold_id": i + 1,
            "test_detector": detector,
            "train_detectors": [d for d in DETECTORS if d != detector],
        }
        for i, detector in enumerate(DETECTORS)
    ]
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/test_evaluation.py -v -k "detectors or lodo_folds"
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark.py tests/test_evaluation.py
git commit -m "feat: add LODO fold definitions"
```

---

## Task 3: Session-Stratified Split and Artifact I/O

**Files:**
- Modify: `src/evaluation/benchmark.py`
- Modify: `tests/test_evaluation.py`

**Split artifact format:**

```json
{
  "fold": 1,
  "variant": "strict_lodo",
  "test_detector": "AGIPD",
  "splits": {
    "session_001": "train",
    "session_002": "val",
    "session_003": "in_domain_test",
    "session_004": "cross_detector_eval"
  }
}
```

Training detector sessions get `"train"` / `"val"` / `"in_domain_test"` (80/10/10).
Test detector sessions get `"cross_detector_eval"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
import json
import tempfile
from pathlib import Path

from src.evaluation.benchmark import (
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_session_stratified_split,
    load_split_artifact,
    save_split_artifact,
)


def _make_sessions(n_train: int, n_test: int) -> list[dict]:
    sessions = [
        {"session_id": f"train_sess_{i}", "detector": "JUNGFRAU_4M", "frame_count": 100 + i}
        for i in range(n_train)
    ]
    sessions += [
        {"session_id": f"test_sess_{i}", "detector": "AGIPD", "frame_count": 80 + i}
        for i in range(n_test)
    ]
    return sessions


def test_split_test_detector_sessions_held_out():
    sessions = _make_sessions(n_train=10, n_test=5)
    artifact = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    for sid, split in artifact["splits"].items():
        if sid.startswith("test_sess"):
            assert split == SPLIT_CROSS_DETECTOR


def test_split_train_sessions_get_three_way_split():
    sessions = _make_sessions(n_train=10, n_test=3)
    artifact = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    train_splits = {
        split for sid, split in artifact["splits"].items() if sid.startswith("train_sess")
    }
    assert train_splits.issubset({SPLIT_TRAIN, SPLIT_VAL, SPLIT_IN_DOMAIN_TEST})


def test_split_ratios_approximate():
    sessions = _make_sessions(n_train=100, n_test=20)
    artifact = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    train_splits = [
        s for sid, s in artifact["splits"].items() if sid.startswith("train_sess")
    ]
    n = len(train_splits)
    n_tr = sum(1 for s in train_splits if s == SPLIT_TRAIN)
    n_va = sum(1 for s in train_splits if s == SPLIT_VAL)
    n_te = sum(1 for s in train_splits if s == SPLIT_IN_DOMAIN_TEST)
    assert abs(n_tr / n - 0.80) < 0.05
    assert abs(n_va / n - 0.10) < 0.05
    assert abs(n_te / n - 0.10) < 0.05


def test_split_reproducible():
    sessions = _make_sessions(n_train=20, n_test=5)
    a1 = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    a2 = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    assert a1["splits"] == a2["splits"]


def test_split_artifact_metadata():
    sessions = _make_sessions(n_train=10, n_test=3)
    artifact = build_session_stratified_split(
        sessions, test_detector="AGIPD", fold=2, variant="strict_lodo", seed=42
    )
    assert artifact["fold"] == 2
    assert artifact["variant"] == "strict_lodo"
    assert artifact["test_detector"] == "AGIPD"
    assert "splits" in artifact


def test_save_load_split_artifact(tmp_path):
    sessions = _make_sessions(n_train=10, n_test=3)
    artifact = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    path = tmp_path / "fold1_strict.json"
    save_split_artifact(artifact, path)
    loaded = load_split_artifact(path)
    assert loaded == artifact
```

- [ ] **Step 2: Run tests to confirm FAIL**

```bash
pytest tests/test_evaluation.py -v -k "split"
```

Expected: FAIL — `ImportError: cannot import name 'build_session_stratified_split'`

- [ ] **Step 3: Append split functions to benchmark.py**

Add the following after `build_lodo_folds` in `src/evaluation/benchmark.py`:

```python
def build_session_stratified_split(
    sessions: list[dict],
    test_detector: str,
    ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
    fold: int | None = None,
    variant: str | None = None,
    seed: int = 42,
) -> dict:
    """Build a session-level split artifact.

    Sessions from test_detector get SPLIT_CROSS_DETECTOR.
    Remaining sessions are assigned greedy train/val/in_domain_test
    (sorted by frame_count descending, bucket fills in ratio order).
    """
    held_out = [s for s in sessions if s["detector"] == test_detector]
    train_pool = sorted(
        [s for s in sessions if s["detector"] != test_detector],
        key=lambda s: s["frame_count"],
        reverse=True,
    )

    bucket_names = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_IN_DOMAIN_TEST]
    bucket_targets = [r * len(train_pool) for r in ratios]
    bucket_counts = [0, 0, 0]
    splits: dict[str, str] = {}

    for s in train_pool:
        deficits = [bucket_targets[i] - bucket_counts[i] for i in range(3)]
        chosen = int(np.argmax(deficits))
        splits[s["session_id"]] = bucket_names[chosen]
        bucket_counts[chosen] += 1

    for s in held_out:
        splits[s["session_id"]] = SPLIT_CROSS_DETECTOR

    return {"fold": fold, "variant": variant, "test_detector": test_detector, "splits": splits}


def save_split_artifact(artifact: dict, path: str | Path) -> None:
    """Write split artifact to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)


def load_split_artifact(path: str | Path) -> dict:
    """Load split artifact from a JSON file."""
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/test_evaluation.py -v -k "split"
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark.py tests/test_evaluation.py
git commit -m "feat: session-stratified split builder and artifact I/O"
```

---

## Task 4: Fold Runner and Benchmark Aggregator

**Files:**
- Modify: `src/evaluation/benchmark.py`
- Modify: `tests/test_evaluation.py`

Convention: models output **logits** (pre-sigmoid). `run_on_loader` applies `torch.sigmoid` internally to get probabilities.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.benchmark import run_on_loader, run_fold, run_benchmark, SPLIT_CROSS_DETECTOR


def _make_perfect_loader() -> DataLoader:
    """DataLoader where logits [2.197, 1.386, -1.386, -2.197] give AP=1.0."""
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    images = torch.zeros(4, 1, 4, 4)
    return DataLoader(TensorDataset(images, y_true), batch_size=4)


class _PerfectModel(torch.nn.Module):
    """Returns logits whose sigmoid gives [0.9, 0.8, 0.2, 0.1]."""
    _LOGITS = torch.tensor([2.197, 1.386, -1.386, -2.197])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._LOGITS[: x.shape[0]].unsqueeze(1)


def test_run_on_loader_perfect_model():
    result = run_on_loader(_PerfectModel(), _make_perfect_loader(), device="cpu")
    assert abs(result["ap"] - 1.0) < 1e-4
    assert abs(result["auc_roc"] - 1.0) < 1e-4
    assert abs(result["f1"] - 1.0) < 1e-4
    assert "threshold" in result


def test_run_on_loader_keys():
    result = run_on_loader(_PerfectModel(), _make_perfect_loader(), device="cpu")
    assert set(result.keys()) == {"ap", "auc_roc", "f1", "threshold"}


def _fixed_factory(loader: DataLoader):
    return lambda session_ids: loader


def test_run_fold_returns_metrics_and_detector():
    artifact = {
        "fold": 1,
        "variant": "strict_lodo",
        "test_detector": "AGIPD",
        "splits": {"sess_a": SPLIT_CROSS_DETECTOR, "sess_b": SPLIT_TRAIN},
    }
    result = run_fold(_PerfectModel(), artifact, _fixed_factory(_make_perfect_loader()), device="cpu")
    assert result["test_detector"] == "AGIPD"
    assert "ap" in result


def test_run_benchmark_covers_all_folds():
    sessions = [
        {"session_id": f"sess_{d}_{i}", "detector": d, "frame_count": 100}
        for d in DETECTORS
        for i in range(5)
    ]
    folds = build_lodo_folds()
    artifacts = [
        build_session_stratified_split(
            sessions, test_detector=f["test_detector"], fold=f["fold_id"], seed=42
        )
        for f in folds
    ]
    results = run_benchmark(
        _PerfectModel(), artifacts, _fixed_factory(_make_perfect_loader()), device="cpu"
    )
    for fold_id in range(1, 5):
        assert f"fold_{fold_id}" in results
    assert "mean_ap" in results
    assert "std_ap" in results
```

- [ ] **Step 2: Run tests to confirm FAIL**

```bash
pytest tests/test_evaluation.py -v -k "run_on_loader or run_fold or run_benchmark"
```

Expected: FAIL — `ImportError: cannot import name 'run_on_loader'`

- [ ] **Step 3: Append fold runner functions to benchmark.py**

Add the following after `load_split_artifact` in `src/evaluation/benchmark.py`:

```python
def run_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Run model on a DataLoader; return ap, auc_roc, f1, threshold.

    Model must output logits (pre-sigmoid) with shape [batch, 1] or [batch].
    """
    model.to(device)
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    best_f1, threshold = f1_at_optimal_threshold(y_true, y_score)
    return {
        "ap": average_precision(y_true, y_score),
        "auc_roc": auc_roc(y_true, y_score),
        "f1": best_f1,
        "threshold": threshold,
    }


def run_fold(
    model: torch.nn.Module,
    split_artifact: dict,
    dataloader_factory: Callable[[list[str]], DataLoader],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on the held-out test-detector sessions for one fold.

    dataloader_factory(session_ids) must return a DataLoader over those sessions.
    """
    held_out_ids = [
        sid
        for sid, split in split_artifact["splits"].items()
        if split == SPLIT_CROSS_DETECTOR
    ]
    loader = dataloader_factory(held_out_ids)
    metrics = run_on_loader(model, loader, device)
    metrics["test_detector"] = split_artifact["test_detector"]
    return metrics


def run_benchmark(
    model: torch.nn.Module,
    split_artifacts: list[dict],
    dataloader_factory: Callable[[list[str]], DataLoader],
    device: str = "cpu",
) -> dict:
    """Run all folds and return per-fold results plus mean_ap and std_ap."""
    results: dict = {}
    for artifact in split_artifacts:
        fold_id = artifact["fold"]
        results[f"fold_{fold_id}"] = run_fold(model, artifact, dataloader_factory, device)

    ap_values = [v["ap"] for v in results.values()]
    results["mean_ap"] = float(np.mean(ap_values))
    results["std_ap"] = float(np.std(ap_values))
    return results
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
pytest tests/test_evaluation.py -v -k "run_on_loader or run_fold or run_benchmark"
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark.py tests/test_evaluation.py
git commit -m "feat: fold runner and benchmark aggregator"
```

---

## Task 5: Baseline Formatter, Config, and Docs

**Files:**
- Modify: `src/evaluation/benchmark.py`
- Modify: `tests/test_evaluation.py`
- Modify: `configs/base.yaml`
- Create: `data/splits/.gitkeep`
- Modify: `docs/eval_protocol.md`

- [ ] **Step 1: Write the failing test for format_results_table**

Append to `tests/test_evaluation.py`:

```python
from src.evaluation.benchmark import format_results_table


def test_format_results_table_contains_all_detectors():
    results = {
        "fold_1": {"ap": 0.85, "test_detector": "AGIPD"},
        "fold_2": {"ap": 0.82, "test_detector": "JUNGFRAU_4M"},
        "fold_3": {"ap": 0.78, "test_detector": "ePix10k"},
        "fold_4": {"ap": 0.90, "test_detector": "Eiger4M"},
        "mean_ap": 0.8375,
        "std_ap": 0.045,
    }
    table = format_results_table(results)
    for detector in ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]:
        assert detector in table
    assert "0.8375" in table or "0.84" in table


def test_format_results_table_with_oracle():
    results = {
        "fold_1": {"ap": 0.85, "test_detector": "AGIPD"},
        "fold_2": {"ap": 0.82, "test_detector": "JUNGFRAU_4M"},
        "fold_3": {"ap": 0.78, "test_detector": "ePix10k"},
        "fold_4": {"ap": 0.90, "test_detector": "Eiger4M"},
        "mean_ap": 0.8375,
        "std_ap": 0.045,
    }
    oracle = {"AGIPD": 0.95, "JUNGFRAU_4M": 0.93, "ePix10k": 0.88, "Eiger4M": 0.97}
    table = format_results_table(results, oracle_ap=oracle)
    assert "Oracle" in table
    assert "%" in table
```

- [ ] **Step 2: Run test to confirm FAIL**

```bash
pytest tests/test_evaluation.py -v -k "format_results_table"
```

Expected: FAIL — `ImportError: cannot import name 'format_results_table'`

- [ ] **Step 3: Append format_results_table to benchmark.py**

Add the following at the end of `src/evaluation/benchmark.py`:

```python
def format_results_table(
    results: dict,
    oracle_ap: dict[str, float] | None = None,
) -> str:
    """Format a results table showing per-fold AP and optional oracle comparison.

    oracle_ap: {detector_name: oracle_AP} — within-detector upper bound.
    """
    has_oracle = oracle_ap is not None
    header = f"{'Fold':<8} {'Detector':<15} {'AP':>8}"
    if has_oracle:
        header += f"  {'Oracle AP':>10}  {'Rel. Gap':>9}"

    sep = "-" * len(header)
    lines = [header, sep]

    for fold_id in range(1, len(DETECTORS) + 1):
        fold_key = f"fold_{fold_id}"
        fold_data = results.get(fold_key, {})
        ap = fold_data.get("ap", float("nan"))
        detector = fold_data.get("test_detector", DETECTORS[fold_id - 1])
        line = f"{fold_key:<8} {detector:<15} {ap:>8.4f}"
        if has_oracle:
            o_ap = oracle_ap.get(detector, float("nan"))
            if o_ap and o_ap > 0:
                gap = (o_ap - ap) / o_ap * 100
                line += f"  {o_ap:>10.4f}  {gap:>8.1f}%"
            else:
                line += f"  {'N/A':>10}  {'N/A':>9}"
        lines.append(line)

    lines.append(sep)
    mean_ap = results.get("mean_ap", float("nan"))
    std_ap = results.get("std_ap", float("nan"))
    lines.append(
        f"{'Mean':<8} {'':<15} {mean_ap:>8.4f}  +/- {std_ap:.4f}"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run full test suite to confirm all tests pass**

```bash
pytest tests/test_evaluation.py -v
```

Expected: All tests PASS (20+ tests)

- [ ] **Step 5: Add benchmark section to configs/base.yaml**

Open `configs/base.yaml` and append at the end:

```yaml

benchmark:
  split_seed: 42
  split_ratios: [0.80, 0.10, 0.10]
  variants: [strict_lodo, semi_supervised_lodo]
  detector_agnostic_max_relative_degradation: null  # set after Phase 4
  detector_agnostic_min_absolute_ap: null           # set after Phase 4
```

- [ ] **Step 6: Create data/splits directory**

```bash
touch data/splits/.gitkeep
```

- [ ] **Step 7: Write docs/eval_protocol.md**

Replace `docs/eval_protocol.md` entirely with:

```markdown
# Cross-Detector Leave-One-Detector-Out Benchmark Protocol

**Version:** 1.0 — Phase 1 finalization
**Spec:** `docs/superpowers/specs/2026-05-13-benchmark-protocol-design.md`
**Implementation:** `src/evaluation/benchmark.py`

---

## Overview

This protocol defines how the SFX hitfinder is assessed for detector-agnostic
generalization. The structure is Leave-One-Detector-Out (LODO): train on three
detector types, test on the fourth, repeated for all four detectors.

Two variants run in parallel:

| Variant | Description |
|---------|-------------|
| **Strict LODO** | Held-out detector contributes zero frames to any training step |
| **Semi-supervised LODO** | Held-out detector unlabeled frames added to MAE pretraining only; labels and test frames remain held out |

The comparison between variants is a scientific contribution: it tests whether
unlabeled target-domain exposure during SSL pretraining improves cross-detector
transfer.

---

## The Four Folds

| Fold | Test Detector | Train Detectors | Facility separation |
|------|--------------|----------------|---------------------|
| 1 | AGIPD | JUNGFRAU 4M, ePix10k, Eiger4M | EuXFEL vs LCLS + Synchrotron |
| 2 | JUNGFRAU 4M | AGIPD, ePix10k, Eiger4M | LCLS-CXI vs EuXFEL + LCLS + Synchrotron |
| 3 | ePix10k | AGIPD, JUNGFRAU 4M, Eiger4M | LCLS vs EuXFEL + LCLS-CXI + Synchrotron |
| 4 | Eiger4M | AGIPD, JUNGFRAU 4M, ePix10k | Synchrotron/SSX vs EuXFEL + LCLS |

Facility-level separation is the primary guarantee of clean splits. JUNGFRAU 4M
and ePix10k share LCLS but run in separate beamtimes with independent samples.

---

## Data Split Construction

Within the three training detectors, sessions are split **80 / 10 / 10**
(train / val / in-domain test). Splits happen at the **session level** — all
frames from one beamtime run stay together.

A session is identified by `(detector_type, facility, run_id)`.

**Algorithm (implemented in `build_session_stratified_split`):**

1. Mark all held-out detector sessions as `cross_detector_eval`
2. Sort remaining sessions by frame count (descending)
3. Greedily assign each session to the bucket most below its target proportion

**Reproducibility:** Split seed is `configs/base.yaml benchmark.split_seed`.
Artifacts are saved to `data/splits/fold<N>_<variant>_split.json` and committed
to git.

---

## Metrics

### Primary: Average Precision (AP)

Area under the precision-recall curve. Threshold-independent and robust to
class imbalance. SFX hit rates can be as low as 5%.

### Secondary (reported, not used for pass/fail)

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Ranking quality, full threshold range |
| F1 at optimal threshold | Single operating-point summary |
| Precision / Recall at optimal threshold | Operational breakdown |

### Aggregation

- Per-fold AP for all 4 folds
- **Mean AP ± std** as headline number
- Baseline comparison via `format_results_table(results, oracle_ap=...)`

### Baselines

| Baseline | Description |
|----------|-------------|
| Peakfinder8 | Threshold-based, current SFX standard |
| CrystFEL hit criterion | Bragg peak count threshold |
| Within-detector oracle | Trained and tested on same detector (upper bound) |

---

## Pass Criteria

A model is **detector-agnostic** only if on **every** fold:

1. **Relative degradation ≤ X%** — fold AP within X% of the within-detector oracle
2. **Absolute AP ≥ Y** — fold AP clears a minimum operational threshold

X and Y are set after Phase 4 results. Stored in `configs/base.yaml` as
`benchmark.detector_agnostic_max_relative_degradation` and
`benchmark.detector_agnostic_min_absolute_ap`.

Partial pass (3/4 folds) is reported but does not count as passing.

---

## Reproducibility Checklist

- [ ] `configs/base.yaml` `benchmark.split_seed` recorded
- [ ] Split artifacts committed to `data/splits/`
- [ ] W&B runs tagged `strict_lodo` or `semi_supervised_lodo`
- [ ] Oracle runs logged alongside LODO runs
- [ ] Precision-recall curves exported for all folds
```

- [ ] **Step 8: Commit everything**

```bash
git add src/evaluation/benchmark.py tests/test_evaluation.py \
        configs/base.yaml data/splits/.gitkeep docs/eval_protocol.md
git commit -m "feat: complete benchmark protocol — baselines, config, eval_protocol.md"
```
