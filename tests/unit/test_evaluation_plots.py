"""Tests des helpers de plot dans `src/evaluation.py`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation import (
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
)


def _data():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.uniform(0, 1, size=200)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


def test_save_confusion_matrix(tmp_path: Path) -> None:
    y_true, y_pred, _ = _data()
    p = tmp_path / "cm.png"
    out = save_confusion_matrix(y_true, y_pred, p, title="Test")
    assert out.exists() and out.stat().st_size > 0


def test_save_roc_curve(tmp_path: Path) -> None:
    y_true, _, y_proba = _data()
    p = tmp_path / "roc.png"
    out = save_roc_curve(y_true, y_proba, p)
    assert out.exists() and out.stat().st_size > 0


def test_save_pr_curve(tmp_path: Path) -> None:
    y_true, _, y_proba = _data()
    p = tmp_path / "pr.png"
    out = save_pr_curve(y_true, y_proba, p)
    assert out.exists() and out.stat().st_size > 0


def test_save_threshold_plot(tmp_path: Path) -> None:
    y_true, _, y_proba = _data()
    p = tmp_path / "thr.png"
    out = save_threshold_plot(y_true, y_proba, p)
    assert out.exists() and out.stat().st_size > 0


def test_find_best_threshold_recall_no_constraint() -> None:
    from src.evaluation import find_best_threshold

    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    thr = find_best_threshold(y, p, metric="recall")
    assert 0.0 <= thr <= 1.0


def test_compute_metrics_without_proba() -> None:
    from src.evaluation import compute_metrics

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert "roc_auc" not in m
    assert "pr_auc" not in m
