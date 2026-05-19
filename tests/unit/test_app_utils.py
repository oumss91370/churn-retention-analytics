"""Tests unitaires de la logique métier du dashboard (`app/utils.py`)."""

from __future__ import annotations

import pytest

from app.utils import (
    feature_name_to_business_label,
    format_currency_eur,
    format_number,
    format_percent,
    proba_to_risk_level,
    recommendation_from_factors,
    revenue_at_risk,
)


def test_format_currency_eur_basic() -> None:
    assert format_currency_eur(12345.67) == "12 346 €"
    assert format_currency_eur(0) == "0 €"
    assert format_currency_eur(-150) == "-150 €"


def test_format_number_zero_decimals() -> None:
    assert format_number(12345) == "12 345"


def test_format_percent_one_decimal() -> None:
    assert format_percent(0.123) == "12,3 %"


def test_proba_to_risk_level_thresholds() -> None:
    assert proba_to_risk_level(0.10) == "faible"
    assert proba_to_risk_level(0.30) == "modéré"
    assert proba_to_risk_level(0.45) == "modéré"
    assert proba_to_risk_level(0.60) == "élevé"
    assert proba_to_risk_level(0.95) == "élevé"


def test_revenue_at_risk_sum() -> None:
    assert revenue_at_risk([0.5, 0.1, 0.9], [100, 200, 300]) == pytest.approx(0.5 * 100 + 0.1 * 200 + 0.9 * 300)


def test_revenue_at_risk_length_mismatch() -> None:
    with pytest.raises(ValueError):
        revenue_at_risk([0.1], [100, 200])


def test_feature_name_to_business_label_known() -> None:
    assert feature_name_to_business_label("payment_failures") == "Échecs de paiement"


def test_feature_name_to_business_label_ohe() -> None:
    label = feature_name_to_business_label("country_USA")
    assert "Pays" in label and "USA" in label


def test_feature_name_to_business_label_fallback() -> None:
    assert feature_name_to_business_label("xy_unknown") == "Xy unknown"


def test_recommendation_from_factors_positive_directions() -> None:
    factors = [
        {"feature": "csat_score", "label": "Score CSAT", "shap_value": 0.2, "direction": "augmente"},
        {"feature": "nps_score", "label": "Score NPS", "shap_value": 0.1, "direction": "augmente"},
    ]
    text = recommendation_from_factors(factors)
    assert "Score CSAT" in text
    assert "Score NPS" in text


def test_recommendation_from_factors_empty() -> None:
    assert "Aucune" in recommendation_from_factors([])
