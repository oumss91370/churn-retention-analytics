"""Tests des plots et explainers dans `src/explain.py`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.explain import (
    _to_dense,
    aggregate_business_importance,
    compute_shap_values,
    make_explainer,
    save_beeswarm,
    save_force_plot,
    save_global_bar_plot,
    to_business_label,
)


def _trained_tree_model():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
    return model, X, y


def test_to_dense_with_array() -> None:
    arr = np.zeros((3, 2))
    out = _to_dense(arr)
    assert out.shape == (3, 2)


def test_make_explainer_tree_and_compute() -> None:
    model, X, _ = _trained_tree_model()
    expl = make_explainer(model, X)
    shap_values = compute_shap_values(expl, X[:10])
    assert shap_values.shape[0] == 10
    assert shap_values.shape[1] == X.shape[1]


def test_make_explainer_linear_kernel() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    expl = make_explainer(model, X[:20])
    shap_values = compute_shap_values(expl, X[:3])
    assert shap_values.shape[0] == 3


def test_save_global_bar_plot(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    shap_values = rng.normal(size=(50, 6))
    feature_names = ["age", "nps_score", "csat_score", "monthly_logins", "tenure_months", "country_USA"]
    out = save_global_bar_plot(shap_values, feature_names, tmp_path / "bar.png", top=5)
    assert out.exists() and out.stat().st_size > 0


def test_save_beeswarm(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    shap_values = rng.normal(size=(50, 4))
    X = rng.normal(size=(50, 4))
    names = ["age", "nps_score", "csat_score", "monthly_logins"]
    out = save_beeswarm(shap_values, X, names, tmp_path / "bee.png", top=4)
    assert out.exists() and out.stat().st_size > 0


def test_save_force_plot(tmp_path: Path) -> None:
    shap_row = np.array([0.2, -0.1, 0.05, -0.3, 0.15])
    names = ["age", "nps_score", "csat_score", "monthly_logins", "tenure_months"]
    out = save_force_plot(shap_row, names, tmp_path / "force.png", title="Test", top=4)
    assert out.exists() and out.stat().st_size > 0


def test_aggregate_business_importance_ordered() -> None:
    rng = np.random.default_rng(42)
    shap_values = rng.normal(size=(20, 3))
    names = ["nps_score", "csat_score", "monthly_logins"]
    agg = aggregate_business_importance(shap_values, names)
    assert list(agg["variable"])[0] in {"nps_score", "csat_score", "monthly_logins"}
    assert (agg["abs_shap"].diff().dropna() <= 0).all()
