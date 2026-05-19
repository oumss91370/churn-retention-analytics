"""Tests unitaires de `src/features.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import clean
from src.features import (
    apply_features,
    build_preprocessor,
    get_feature_names,
    make_features,
)


def test_tickets_per_month_formula() -> None:
    df = pd.DataFrame(
        {
            "support_tickets": [10, 0, 3],
            "tenure_months": [5, 12, 0],
            "payment_failures": [0, 0, 0],
            "total_revenue": [100, 200, 300],
            "usage_growth_rate": [0.1, 0.0, -0.1],
        }
    )
    out = make_features(df)
    expected = np.array([10 / 6, 0 / 13, 3 / 1])
    np.testing.assert_allclose(out["tickets_per_month"].values, expected)


def test_failed_payment_rate_formula() -> None:
    df = pd.DataFrame(
        {
            "support_tickets": [0, 0],
            "tenure_months": [10, 0],
            "payment_failures": [2, 1],
            "total_revenue": [100, 200],
            "usage_growth_rate": [0.0, 0.0],
        }
    )
    out = make_features(df)
    np.testing.assert_allclose(out["failed_payment_rate"].values, [2 / 11, 1 / 1])


def test_is_high_value_threshold() -> None:
    df = pd.DataFrame(
        {
            "support_tickets": [0, 0, 0, 0],
            "tenure_months": [10, 10, 10, 10],
            "payment_failures": [0, 0, 0, 0],
            "total_revenue": [100, 200, 300, 400],
            "usage_growth_rate": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = make_features(df)
    assert out["is_high_value"].tolist() == [0, 0, 1, 1]


def test_engagement_drop_signs() -> None:
    df = pd.DataFrame(
        {
            "support_tickets": [0, 0, 0],
            "tenure_months": [10, 10, 10],
            "payment_failures": [0, 0, 0],
            "total_revenue": [100, 200, 300],
            "usage_growth_rate": [-0.1, 0.0, 0.2],
        }
    )
    out = make_features(df)
    assert out["engagement_drop"].tolist() == [1, 0, 0]


def test_apply_features_uses_external_median() -> None:
    df = pd.DataFrame(
        {
            "support_tickets": [0, 0],
            "tenure_months": [5, 5],
            "payment_failures": [0, 0],
            "total_revenue": [100, 1_000],
            "usage_growth_rate": [0.1, 0.1],
        }
    )
    out = apply_features(df, revenue_median=500)
    assert out["is_high_value"].tolist() == [0, 1]


def test_preprocessor_fit_train_only(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    feat = make_features(cleaned)
    train = feat.iloc[:150]
    test = feat.iloc[150:]
    pre = build_preprocessor()
    pre.fit(train)

    scaler = pre.named_transformers_["num"]
    expected_means = train[scaler.feature_names_in_].mean().values
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)

    train_test = pd.concat([train, test])
    means_full = train_test[scaler.feature_names_in_].mean().values
    assert not np.allclose(scaler.mean_, means_full, rtol=1e-3)


def test_preprocessor_handles_unknown_category(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    feat = make_features(cleaned)
    pre = build_preprocessor()
    pre.fit(feat)

    novel = feat.iloc[:1].copy()
    novel["country"] = "Atlantide"
    transformed = pre.transform(novel)
    assert transformed.shape[0] == 1


def test_preprocessor_output_shape(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    feat = make_features(cleaned)
    pre = build_preprocessor()
    transformed = pre.fit_transform(feat)
    assert transformed.shape[0] == feat.shape[0]
    names = get_feature_names(pre)
    assert transformed.shape[1] == len(names)
