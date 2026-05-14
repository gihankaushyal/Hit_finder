import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.benchmark import (
    DETECTORS,
    SPLIT_CROSS_DETECTOR,
    SPLIT_IN_DOMAIN_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    build_lodo_folds,
    build_session_stratified_split,
    load_split_artifact,
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
