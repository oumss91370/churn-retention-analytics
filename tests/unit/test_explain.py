"""Tests unitaires de `src/explain.py`."""

from __future__ import annotations

import numpy as np

from src.explain import (
    aggregate_business_importance,
    to_business_label,
    top_factors_for_instance,
)


def test_top_factors_returns_k_items() -> None:
    shap_row = np.array([0.1, -0.5, 0.3, -0.05, 0.2])
    names = ["age", "nps_score", "csat_score", "tenure_months", "monthly_logins"]
    top = top_factors_for_instance(shap_row, names, k=3)
    assert len(top) == 3


def test_top_factors_sorted_by_abs_value() -> None:
    shap_row = np.array([0.1, -0.5, 0.3])
    names = ["a", "b", "c"]
    top = top_factors_for_instance(shap_row, names, k=3)
    assert top[0]["feature"] == "b"
    assert top[1]["feature"] == "c"
    assert top[2]["feature"] == "a"


def test_top_factors_directions() -> None:
    shap_row = np.array([0.4, -0.3, 0.1])
    names = ["a", "b", "c"]
    top = top_factors_for_instance(shap_row, names, k=3)
    assert top[0]["direction"] == "augmente"
    assert top[1]["direction"] == "diminue"


def test_to_business_label_known_feature() -> None:
    assert to_business_label("nps_score") == "Score NPS"
    assert to_business_label("payment_failures") == "Échecs de paiement"


def test_to_business_label_ohe() -> None:
    label = to_business_label("country_USA")
    assert "Pays" in label and "USA" in label


def test_to_business_label_fallback() -> None:
    assert to_business_label("unknown_feature_x") == "Unknown feature x"


def test_aggregate_business_importance_sums_ohe() -> None:
    shap_values = np.array(
        [
            [0.1, -0.2, 0.05, -0.1, 0.3],
            [0.2, 0.1, -0.05, 0.0, -0.2],
        ]
    )
    feature_names = [
        "country_USA",
        "country_UK",
        "country_Germany",
        "nps_score",
        "csat_score",
    ]
    agg = aggregate_business_importance(shap_values, feature_names)
    assert "country" in agg["variable"].values
    country_row = agg[agg["variable"] == "country"]
    expected = (abs(shap_values[:, 0]).mean() + abs(shap_values[:, 1]).mean() + abs(shap_values[:, 2]).mean())
    assert float(country_row["abs_shap"].iloc[0]) == float(np.round(expected, 12)) or abs(
        float(country_row["abs_shap"].iloc[0]) - expected
    ) < 1e-9
