"""Tests d'intégration des helpers d'orchestration de `src/models.py`."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.data import clean, split
from src.evaluation import compute_metrics
from src.features import get_feature_names, make_features
from src.models import (
    _logreg,
    _predict_proba,
    _random_forest,
    _tuned_metrics,
    _xgboost,
    fit_preprocessor,
    save_artifacts,
    transform_test,
    tune_hyperparameters,
)


def test_fit_preprocessor_and_transform(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    X_train, X_test, _, _ = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, median = fit_preprocessor(X_train)
    X_test_proc = transform_test(pre, X_test, median)
    assert X_test_proc.shape[0] == X_test.shape[0]
    assert X_test_proc.shape[1] == X_train_proc.shape[1]


def test_tuned_metrics_returns_floor(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    X_train, X_test, y_train, y_test = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, median = fit_preprocessor(X_train)
    X_test_proc = transform_test(pre, X_test, median)

    model = _logreg(class_weight="balanced").fit(X_train_proc, y_train)
    threshold, proba, metrics = _tuned_metrics(model, X_test_proc, y_test)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert proba.shape[0] == X_test_proc.shape[0]


def test_predict_proba_decision_function_fallback(mock_df: pd.DataFrame) -> None:
    from sklearn.svm import LinearSVC

    cleaned = clean(mock_df)
    X_train, X_test, y_train, _ = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, median = fit_preprocessor(X_train)
    X_test_proc = transform_test(pre, X_test, median)

    model = LinearSVC(random_state=RANDOM_STATE, max_iter=1000).fit(X_train_proc, y_train)
    proba = _predict_proba(model, X_test_proc)
    assert proba.shape[0] == X_test_proc.shape[0]
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_tune_hyperparameters_logreg_returns_fitted(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    X_train, _, y_train, _ = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, _ = fit_preprocessor(X_train)

    est = _logreg(class_weight="balanced")
    fitted = tune_hyperparameters("logreg", est, X_train_proc, y_train)
    assert hasattr(fitted, "coef_")


def test_tune_hyperparameters_xgboost_small(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    X_train, _, y_train, _ = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, _ = fit_preprocessor(X_train)

    est = _xgboost()
    fitted = tune_hyperparameters("xgboost", est, X_train_proc, y_train, n_iter=3, cv_splits=2)
    assert hasattr(fitted, "predict_proba")


def test_save_artifacts_round_trip(mock_df: pd.DataFrame, tmp_path: Path, monkeypatch) -> None:
    cleaned = clean(mock_df)
    X_train, X_test, y_train, y_test = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, median = fit_preprocessor(X_train)

    model = _random_forest(class_weight="balanced")
    model.fit(X_train_proc, y_train)

    import src.models as models_mod

    monkeypatch.setattr(models_mod, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(models_mod, "PREPROCESSOR_PATH", tmp_path / "preprocessor.pkl")
    monkeypatch.setattr(models_mod, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(models_mod, "MODELS_DIR", tmp_path)

    save_artifacts(
        "random_forest",
        model,
        pre,
        median,
        {"f1": 0.4, "pr_auc": 0.3, "tn": 100, "fp": 20, "fn": 10, "tp": 30},
        0.4,
        get_feature_names(pre),
    )

    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "preprocessor.pkl").exists()
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_type"] == "random_forest"
    assert meta["decision_threshold"] == 0.4
