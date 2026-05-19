"""Tests d'intégration de `src/models.py` sur petit échantillon."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE
from src.data import clean
from src.evaluation import compute_metrics
from src.features import build_preprocessor, make_features
from src.models import (
    _logreg,
    _mlp,
    _random_forest,
    _xgboost,
    compare_imbalance_strategies,
    cross_validate_models,
    fit_preprocessor,
    select_final_model,
    transform_test,
)


@pytest.fixture(scope="module")
def small_split(mock_df: pd.DataFrame):
    cleaned = clean(mock_df)
    from src.data import split

    X_train, X_test, y_train, y_test = split(cleaned, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def test_train_baseline_returns_fitted(small_split) -> None:
    X_train, X_test, y_train, y_test = small_split
    pre, X_train_proc, median_rev = fit_preprocessor(X_train)
    model = _logreg(class_weight="balanced")
    model.fit(X_train_proc, y_train)
    assert hasattr(model, "coef_")
    proba = model.predict_proba(transform_test(pre, X_test, median_rev))[:, 1]
    assert proba.shape[0] == X_test.shape[0]


def test_train_random_forest_returns_fitted(small_split) -> None:
    X_train, _, y_train, _ = small_split
    pre, X_train_proc, _ = fit_preprocessor(X_train)
    model = _random_forest(class_weight="balanced")
    model.fit(X_train_proc, y_train)
    assert hasattr(model, "feature_importances_")


def test_train_xgboost_returns_fitted(small_split) -> None:
    X_train, _, y_train, _ = small_split
    pre, X_train_proc, _ = fit_preprocessor(X_train)
    model = _xgboost(scale_pos_weight=9.0)
    model.fit(X_train_proc, y_train)
    assert hasattr(model, "predict_proba")


def test_train_mlp_returns_fitted(small_split) -> None:
    X_train, _, y_train, _ = small_split
    pre, X_train_proc, _ = fit_preprocessor(X_train)
    model = _mlp()
    model.fit(X_train_proc, y_train)
    assert hasattr(model, "loss_")


def test_compare_imbalance_strategies_returns_dataframe(small_split) -> None:
    X_train, X_test, y_train, y_test = small_split
    pre, X_train_proc, median_rev = fit_preprocessor(X_train)
    X_test_proc = transform_test(pre, X_test, median_rev)
    df = compare_imbalance_strategies(X_train_proc, y_train, X_test_proc, y_test)
    assert "strategy" in df.columns
    assert "f1" in df.columns
    assert len(df) == 4


def test_cross_validate_models_returns_dataframe(small_split) -> None:
    X_train, _, y_train, _ = small_split
    pre, X_train_proc, _ = fit_preprocessor(X_train)
    estimators = {"logreg": _logreg(class_weight="balanced")}
    df = cross_validate_models(estimators, X_train_proc, y_train, n_splits=3)
    assert set(df.columns).issuperset({"model", "cv_f1_mean", "cv_pr_auc_mean"})


def test_select_final_model_picks_best() -> None:
    df = pd.DataFrame({"model": ["a", "b", "c"], "f1": [0.3, 0.5, 0.4]})
    assert select_final_model(df, primary="f1") == "b"
