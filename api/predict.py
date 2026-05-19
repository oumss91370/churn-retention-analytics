"""Logique d'inférence partagée par l'API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
)
from src.explain import (
    compute_shap_values,
    make_explainer,
    top_factors_for_instance,
)
from src.features import apply_features, get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class InferenceContext:
    """Conteneur pour les artefacts chargés une seule fois."""

    model: object
    preprocessor: object
    metadata: dict
    feature_names: list[str]
    explainer: object | None = None


_CONTEXT: InferenceContext | None = None


def is_loaded() -> bool:
    return _CONTEXT is not None


def get_context() -> InferenceContext:
    if _CONTEXT is None:
        raise RuntimeError("Le modèle n'est pas chargé")
    return _CONTEXT


def load_artifacts(
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    metadata_path: Path = METADATA_PATH,
    build_explainer: bool = True,
) -> InferenceContext:
    """Charge le modèle, le preprocessor et les métadonnées en mémoire."""
    global _CONTEXT

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    if not Path(preprocessor_path).exists():
        raise FileNotFoundError(f"Preprocessor introuvable : {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    if Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    else:
        metadata = {
            "model_type": "unknown",
            "model_class": model.__class__.__name__,
            "decision_threshold": 0.5,
            "revenue_median_train": 0.0,
        }

    feature_names = metadata.get("feature_names") or get_feature_names(preprocessor)

    explainer = None
    if build_explainer:
        try:
            from src.data import load_clean_split

            X_train, _, _, _ = load_clean_split()
            X_train_feat = apply_features(X_train, metadata.get("revenue_median_train", 0.0))
            X_train_proc = preprocessor.transform(X_train_feat)
            X_dense = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else np.asarray(X_train_proc)
            rng = np.random.default_rng(42)
            sample_size = min(200, X_dense.shape[0])
            idx = rng.choice(X_dense.shape[0], size=sample_size, replace=False)
            explainer = make_explainer(model, X_dense[idx])
        except Exception as exc:  # pragma: no cover - fallback non testé
            logger.warning("Initialisation SHAP impossible : %s", exc)

    _CONTEXT = InferenceContext(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        feature_names=feature_names,
        explainer=explainer,
    )
    return _CONTEXT


def unload() -> None:
    """Réinitialise le contexte (utile pour les tests)."""
    global _CONTEXT
    _CONTEXT = None


def proba_to_risk_level(proba: float) -> str:
    """Convertit une probabilité en niveau de risque métier."""
    if proba < RISK_THRESHOLD_LOW:
        return "faible"
    if proba < RISK_THRESHOLD_HIGH:
        return "modéré"
    return "élevé"


def predict_one(payload: dict) -> dict:
    """Inférence sur un client unique.

    Args:
        payload: dictionnaire validé conforme à `ClientPayload`.

    Returns:
        Dictionnaire conforme à `PredictResponse`.
    """
    ctx = get_context()
    revenue_median = float(ctx.metadata.get("revenue_median_train", 0.0))
    threshold = float(ctx.metadata.get("decision_threshold", 0.5))

    df = pd.DataFrame([payload])
    df_feat = apply_features(df, revenue_median=revenue_median)
    X_proc = ctx.preprocessor.transform(df_feat)
    X_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)

    proba = float(ctx.model.predict_proba(X_proc)[0, 1]) if hasattr(ctx.model, "predict_proba") else float(
        ctx.model.predict(X_proc)[0]
    )
    prediction = int(proba >= threshold)

    top_factors: list[dict] = []
    if ctx.explainer is not None:
        try:
            shap_vals = compute_shap_values(ctx.explainer, X_dense)
            top_factors = top_factors_for_instance(shap_vals[0], ctx.feature_names, k=3)
        except Exception as exc:  # pragma: no cover - fallback robustesse
            logger.warning("Calcul SHAP individuel impossible : %s", exc)

    if not top_factors:
        importances = None
        if hasattr(ctx.model, "feature_importances_"):
            importances = np.asarray(ctx.model.feature_importances_)
        elif hasattr(ctx.model, "coef_"):
            coef = np.asarray(ctx.model.coef_).ravel()
            importances = np.abs(coef)
        if importances is not None and len(importances) == len(ctx.feature_names):
            order = np.argsort(-importances)[:3]
            from src.explain import to_business_label

            for idx in order:
                feat = ctx.feature_names[idx]
                top_factors.append(
                    {
                        "feature": feat,
                        "label": to_business_label(feat),
                        "shap_value": float(importances[idx]),
                        "direction": "augmente",
                    }
                )

    return {
        "churn_probability": proba,
        "churn_prediction": prediction,
        "risk_level": proba_to_risk_level(proba),
        "decision_threshold": threshold,
        "top_factors": top_factors,
    }
