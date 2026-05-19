"""Chargement, nettoyage et découpage du dataset churn."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ID_COL, RANDOM_STATE, RAW_CSV, TARGET, TEST_SIZE

logger = logging.getLogger(__name__)


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Charge le CSV brut.

    Args:
        path: chemin vers le CSV ; par défaut `data/raw/customer_churn.csv`.

    Returns:
        DataFrame brut (10 000 × 32 sur le dataset fourni).
    """
    csv_path = Path(path) if path is not None else RAW_CSV
    df = pd.read_csv(csv_path)
    logger.info("Dataset chargé : shape=%s", df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage idempotent du DataFrame brut.

    - Supprime les doublons.
    - Supprime la colonne `customer_id` (identifiant non prédictif).
    - Remplace les NaN de `complaint_type` par la catégorie `"None"`.

    Args:
        df: DataFrame brut.

    Returns:
        DataFrame nettoyé, sans NaN, sans colonne identifiant.
    """
    out = df.copy()
    out = out.drop_duplicates().reset_index(drop=True)

    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])

    if "complaint_type" in out.columns:
        out["complaint_type"] = out["complaint_type"].fillna("None")

    return out


def split(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare le DataFrame en X_train, X_test, y_train, y_test stratifiés.

    Args:
        df: DataFrame nettoyé.
        target: nom de la colonne cible.
        test_size: proportion du test set.
        random_state: graine.

    Returns:
        Quadruplet (X_train, X_test, y_train, y_test).
    """
    if target not in df.columns:
        raise KeyError(f"Colonne cible absente : {target!r}")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def load_clean_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline pratique : chargement → clean → split.

    Returns:
        Quadruplet (X_train, X_test, y_train, y_test).
    """
    return split(clean(load_raw()))
