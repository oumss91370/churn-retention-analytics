"""Scénarios critiques CRIT-01 à CRIT-04 : intégrité du pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE
from src.data import clean, split
from src.features import apply_features, build_preprocessor, make_features
from src.models import _logreg, fit_preprocessor


def test_crit01_no_data_leakage_in_preprocessing(mock_df: pd.DataFrame) -> None:
    """CRIT-01 : le preprocessor est fitté uniquement sur le train."""
    cleaned = clean(mock_df)
    X_train, X_test, _, _ = split(cleaned, random_state=RANDOM_STATE)
    X_train_feat = make_features(X_train)
    pre = build_preprocessor()
    pre.fit(X_train_feat)

    scaler = pre.named_transformers_["num"]
    expected = X_train_feat[scaler.feature_names_in_].mean().values
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-5)


def test_crit02_smote_only_on_train(mock_df: pd.DataFrame) -> None:
    """CRIT-02 : SMOTE ne touche pas le test set (sa distribution doit rester intacte)."""
    cleaned = clean(mock_df)
    X_train, X_test, y_train, y_test = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, median = fit_preprocessor(X_train)

    original_ratio = float(y_test.mean())
    X_train_resampled, y_train_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_proc, y_train)

    assert pytest.approx(original_ratio, abs=0.001) == float(y_test.mean())
    assert float(y_train_resampled.mean()) == pytest.approx(0.5, abs=0.05)


def test_crit03_model_serialization_round_trip(mock_df: pd.DataFrame, tmp_path: Path) -> None:
    """CRIT-03 : un modèle chargé produit exactement les mêmes prédictions."""
    cleaned = clean(mock_df)
    X_train, X_test, y_train, _ = split(cleaned, random_state=RANDOM_STATE)
    pre, X_train_proc, median = fit_preprocessor(X_train)

    model = _logreg(class_weight="balanced")
    model.fit(X_train_proc, y_train)

    artefact = tmp_path / "model.pkl"
    joblib.dump(model, artefact)
    loaded = joblib.load(artefact)

    X_test_feat = apply_features(X_test, median)
    X_test_proc = pre.transform(X_test_feat)

    np.testing.assert_array_equal(
        model.predict_proba(X_test_proc),
        loaded.predict_proba(X_test_proc),
    )


def test_crit04_split_reproducible(mock_df: pd.DataFrame) -> None:
    """CRIT-04 : deux exécutions avec random_state=42 donnent des splits identiques."""
    cleaned = clean(mock_df)
    Xa, Xb, ya, yb = split(cleaned, random_state=RANDOM_STATE)
    Xa2, Xb2, ya2, yb2 = split(cleaned, random_state=RANDOM_STATE)
    pd.testing.assert_frame_equal(Xa, Xa2)
    pd.testing.assert_frame_equal(Xb, Xb2)
    pd.testing.assert_series_equal(ya, ya2)
    pd.testing.assert_series_equal(yb, yb2)
