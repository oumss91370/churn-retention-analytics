"""Tests unitaires de `src/evaluation.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import (
    compute_metrics,
    confusion_to_dict,
    find_best_threshold,
    threshold_analysis,
)


def test_compute_metrics_perfect_prediction() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = y_true.copy()
    proba = y_true.astype(float)
    m = compute_metrics(y_true, y_pred, proba)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_all_zeros() -> None:
    y_true = np.array([0] * 9 + [1])
    y_pred = np.zeros_like(y_true)
    proba = np.zeros_like(y_true, dtype=float)
    m = compute_metrics(y_true, y_pred, proba)
    assert m["recall"] == 0.0


def test_compute_metrics_returns_all_keys() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_metrics(y_true, y_pred, proba)
    assert set(m).issuperset({"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"})


def test_confusion_to_dict_shape() -> None:
    cm = confusion_to_dict(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1]))
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_threshold_analysis_monotonic_recall() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.uniform(0, 1, size=200)
    table = threshold_analysis(y_true, y_proba)
    recalls = table["recall"].values
    assert recalls[0] >= recalls[-1]


def test_find_best_threshold_f1() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.2, 0.55, 0.6, 0.7, 0.85])
    thr = find_best_threshold(y_true, y_proba, metric="f1")
    assert 0.3 <= thr <= 0.6


def test_find_best_threshold_recall_with_precision_constraint() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    thr = find_best_threshold(y_true, y_proba, metric="recall", min_precision=0.5)
    assert thr <= 0.6


def test_threshold_analysis_columns() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_proba = np.array([0.2, 0.7, 0.3, 0.8, 0.55])
    table = threshold_analysis(y_true, y_proba)
    assert isinstance(table, pd.DataFrame)
    assert set(table.columns) == {"threshold", "precision", "recall", "f1"}
