"""Feature engineering et pipeline de préprocessing."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les 4 features dérivées au DataFrame.

    Les features dérivées :
    - `tickets_per_month` : pression du support normalisée par ancienneté.
    - `failed_payment_rate` : taux d'échec de paiement normalisé.
    - `is_high_value` : indicateur binaire revenu total > médiane train.
    - `engagement_drop` : indicateur binaire pour usage_growth_rate négatif.

    La médiane utilisée pour `is_high_value` est calculée sur le DataFrame
    fourni : à passer uniquement sur le train, puis stocker la valeur pour
    appliquer le même seuil au test (voir `apply_features`).

    Args:
        df: DataFrame nettoyé.

    Returns:
        DataFrame enrichi des 4 features dérivées.
    """
    out = df.copy()

    tenure_safe = out["tenure_months"] + 1
    out["tickets_per_month"] = out["support_tickets"] / tenure_safe
    out["failed_payment_rate"] = out["payment_failures"] / tenure_safe

    revenue_median = out["total_revenue"].median()
    out["is_high_value"] = (out["total_revenue"] > revenue_median).astype(int)

    out["engagement_drop"] = (out["usage_growth_rate"] < 0).astype(int)

    return out


def apply_features(df: pd.DataFrame, revenue_median: float) -> pd.DataFrame:
    """Applique les features dérivées avec une médiane externe (fit sur train).

    Args:
        df: DataFrame nettoyé (typiquement le test set).
        revenue_median: médiane du revenu total apprise sur le train.

    Returns:
        DataFrame enrichi des 4 features dérivées.
    """
    out = df.copy()

    tenure_safe = out["tenure_months"] + 1
    out["tickets_per_month"] = out["support_tickets"] / tenure_safe
    out["failed_payment_rate"] = out["payment_failures"] / tenure_safe
    out["is_high_value"] = (out["total_revenue"] > revenue_median).astype(int)
    out["engagement_drop"] = (out["usage_growth_rate"] < 0).astype(int)

    return out


def build_preprocessor() -> ColumnTransformer:
    """Construit le ColumnTransformer numérique + catégoriel.

    - StandardScaler sur les variables numériques (originales + dérivées).
    - OneHotEncoder sur les variables catégorielles, `handle_unknown='ignore'`.

    Returns:
        ColumnTransformer non fitté.
    """
    numeric_cols = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_pipeline(estimator) -> Pipeline:
    """Encapsule preprocessing + estimateur dans un Pipeline sklearn.

    Args:
        estimator: estimateur sklearn (LogReg, RF, XGB, MLP, …).

    Returns:
        Pipeline non fitté.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retourne la liste des features en sortie du preprocessor fitté."""
    return list(preprocessor.get_feature_names_out())
