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
