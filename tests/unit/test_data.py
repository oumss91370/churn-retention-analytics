"""Tests unitaires de `src/data.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ID_COL, RANDOM_STATE, TARGET
from src.data import clean, load_raw, split


def test_load_raw_shape() -> None:
    df = load_raw()
    assert df.shape == (10_000, 32)


def test_load_raw_columns() -> None:
    df = load_raw()
    assert TARGET in df.columns
    assert ID_COL in df.columns


def test_clean_drops_customer_id(mock_df: pd.DataFrame) -> None:
    out = clean(mock_df)
    assert ID_COL not in out.columns


def test_clean_complaint_type_fillna(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    assert cleaned["complaint_type"].isna().sum() == 0
    assert "None" in set(cleaned["complaint_type"].unique())


def test_clean_no_nan_after_clean(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_no_duplicates() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B"],
            "gender": ["Male", "Male", "Female"],
            "age": [30, 30, 40],
            "complaint_type": ["Billing", "Billing", np.nan],
            "churn": [0, 0, 1],
        }
    )
    out = clean(df)
    assert len(out) == 2


def test_split_stratified(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    X_train, X_test, y_train, y_test = split(cleaned, random_state=RANDOM_STATE)
    train_ratio = float(y_train.mean())
    test_ratio = float(y_test.mean())
    assert abs(train_ratio - test_ratio) < 0.05


def test_split_no_overlap(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    X_train, X_test, _, _ = split(cleaned, random_state=RANDOM_STATE)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_reproducible(mock_df: pd.DataFrame) -> None:
    cleaned = clean(mock_df)
    Xa, Xb, ya, yb = split(cleaned, random_state=RANDOM_STATE)
    Xa2, Xb2, ya2, yb2 = split(cleaned, random_state=RANDOM_STATE)
    pd.testing.assert_frame_equal(Xa, Xa2)
    pd.testing.assert_frame_equal(Xb, Xb2)
    pd.testing.assert_series_equal(ya, ya2)
    pd.testing.assert_series_equal(yb, yb2)
