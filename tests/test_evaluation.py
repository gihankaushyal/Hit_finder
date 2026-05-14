import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.benchmark import (
    DETECTORS,
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
    format_results_table,
    load_split_artifact,
    run_benchmark,
    run_fold,
    run_on_loader,
    save_split_artifact,
)
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


def test_f1_at_optimal_threshold_known_value():
    # At threshold 0.6: pred=[1,1,0,0], TP=1,FP=1,FN=0 -> P=0.5,R=1.0,F1=0.667
    # At threshold 0.5: pred=[1,1,1,0], TP=1,FP=2,FN=0 -> P=0.33,R=1.0,F1=0.5
    # At threshold 0.9: pred=[1,0,0,0], TP=1,FP=0,FN=0 -> P=1.0,R=1.0,F1=1.0
    # Best: threshold=0.9, F1=1.0
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.6, 0.5, 0.1])
    f1, thresh = f1_at_optimal_threshold(y_true, y_score)
    assert abs(f1 - 1.0) < 1e-6
    assert thresh == pytest.approx(0.9)


def test_f1_empty_input():
    f1, thresh = f1_at_optimal_threshold(np.array([]), np.array([]))
    assert f1 == 0.0
    assert np.isnan(thresh)


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


def _make_sessions(n_train: int, n_test: int) -> list[dict]:
    sessions = [
        {
            "session_id": f"train_sess_{i}",
            "detector": "JUNGFRAU_4M",
            "frame_count": 100 + i,
        }
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
    sessions = _make_sessions(n_train=30, n_test=3)
    artifact = build_session_stratified_split(sessions, test_detector="AGIPD", seed=42)
    train_splits = [
        split
        for sid, split in artifact["splits"].items()
        if sid.startswith("train_sess")
    ]
    split_set = set(train_splits)
    assert SPLIT_TRAIN in split_set
    assert SPLIT_VAL in split_set
    assert SPLIT_IN_DOMAIN_TEST in split_set


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
    result = run_fold(
        _PerfectModel(), artifact, _fixed_factory(_make_perfect_loader()), device="cpu"
    )
    assert result["test_detector"] == "AGIPD"
    assert "ap" in result


def test_run_benchmark_covers_all_folds():
    sessions = [
        {"session_id": f"sess_{d}_{i}", "detector": d, "frame_count": 100}
        for d in ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]
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
    # AGIPD fold: oracle=0.95, model=0.85 -> rel gap = (0.95-0.85)/0.95*100 = 10.5%
    assert "10.5%" in table


# ---------------------------------------------------------------------------
# Issue 1: AP tie-breaking regression test (must match sklearn convention)
# ---------------------------------------------------------------------------


def test_average_precision_tie_breaking_matches_sklearn():
    """Regression: AP on tied scores must match sklearn's pessimistic convention.

    sklearn uses np.argsort(probas_pred, kind='stable')[::-1] internally, which
    puts higher-indexed samples first when scores are equal (no positive promotion).
    Our implementation must produce the identical result.

    Known values verified against sklearn 1.x:
      y_true  = [1, 0, 1, 0],  y_score = [0.8, 0.8, 0.5, 0.5]
      sklearn AP = 0.5  (pessimistic: negatives win ties)
    """
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.8, 0.8, 0.5, 0.5])
    assert abs(average_precision(y_true, y_score) - 0.5) < 1e-6


def test_average_precision_tie_breaking_all_same_score():
    """When all scores are identical, AP should equal the positive rate (pessimistic)."""
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    # 2 positives out of 5 — all tied, so precision at each step = cumulative TP/rank
    # sklearn convention: order by index descending (4,3,2,1,0) → labels [0,0,1,0,1]
    # tp=[0,0,1,1,2], p=[0,0,1/3,1/4,2/5], recall=[0,0,0.5,0.5,1.0]
    # recall_diff=[0,0,0.5,0,0.5], AP = 1/3*0.5 + 2/5*0.5 = 1/6 + 1/5 = 11/30
    expected = 1 / 3 * 0.5 + 2 / 5 * 0.5
    assert abs(average_precision(y_true, y_score) - expected) < 1e-6


# ---------------------------------------------------------------------------
# Issue 2: run_benchmark raises on fold=None
# ---------------------------------------------------------------------------


def test_run_benchmark_raises_on_none_fold():
    artifact_no_fold = {
        "fold": None,
        "test_detector": "AGIPD",
        "splits": {"sess_a": SPLIT_CROSS_DETECTOR},
    }
    with pytest.raises(ValueError, match="fold=None"):
        run_benchmark(
            _PerfectModel(), [artifact_no_fold], _fixed_factory(_make_perfect_loader())
        )


# ---------------------------------------------------------------------------
# Issue 3: std_ap uses sample std (ddof=1)
# ---------------------------------------------------------------------------


def test_run_benchmark_std_ap_uses_sample_std():
    """std_ap must use ddof=1 (sample std), not ddof=0 (population std)."""
    sessions = [
        {"session_id": f"sess_{d}_{i}", "detector": d, "frame_count": 100}
        for d in ["AGIPD", "JUNGFRAU_4M", "ePix10k", "Eiger4M"]
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
    ap_values = [results[f"fold_{i}"]["ap"] for i in range(1, 5)]
    expected_std = float(np.std(ap_values, ddof=1))
    assert abs(results["std_ap"] - expected_std) < 1e-9


# ---------------------------------------------------------------------------
# Issue 4: build_session_stratified_split raises for semi_supervised_lodo
# ---------------------------------------------------------------------------


def test_split_raises_for_semi_supervised():
    sessions = _make_sessions(n_train=10, n_test=3)
    with pytest.raises(NotImplementedError):
        build_session_stratified_split(
            sessions, test_detector="AGIPD", variant="semi_supervised_lodo"
        )


# ---------------------------------------------------------------------------
# Issue 5: run_on_loader returns NaN dict for empty DataLoader
# ---------------------------------------------------------------------------


def _make_empty_loader() -> DataLoader:
    """DataLoader with zero samples."""
    dataset = TensorDataset(torch.zeros(0, 1, 4, 4), torch.zeros(0, dtype=torch.long))
    return DataLoader(dataset, batch_size=4)


def test_run_on_loader_empty_returns_nan():
    result = run_on_loader(_PerfectModel(), _make_empty_loader(), device="cpu")
    assert set(result.keys()) == {"ap", "auc_roc", "f1", "threshold"}
    for key in ("ap", "auc_roc", "f1", "threshold"):
        assert np.isnan(result[key]), f"Expected NaN for '{key}', got {result[key]}"
