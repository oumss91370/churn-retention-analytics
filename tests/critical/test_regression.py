"""Scénario critique CRIT-05 : performance non-régressée du modèle final."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.config import (
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from src.data import load_clean_split
from src.evaluation import compute_metrics
from src.features import apply_features

# Plancher réaliste calibré sur l'observation du dataset (PR-AUC plafonne ~0.30).
F1_FLOOR = 0.38
PR_AUC_FLOOR = 0.27
RECALL_FLOOR = 0.55


@pytest.mark.regression
def test_crit05_performance_regression() -> None:
    """Le modèle final doit dépasser un plancher réaliste de performance."""
    if not MODEL_PATH.exists():
        pytest.skip("Modèle non sérialisé : exécuter `python -m src.models` au préalable.")

    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROCESSOR_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    threshold = float(meta.get("decision_threshold", 0.5))
    revenue_median = float(meta.get("revenue_median_train", 0.0))

    _, X_test, _, y_test = load_clean_split()
    X_test_feat = apply_features(X_test, revenue_median)
    X_test_proc = pre.transform(X_test_feat)

    proba = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    metrics = compute_metrics(np.asarray(y_test), y_pred, proba)

    assert metrics["f1"] >= F1_FLOOR, f"F1 régressé : {metrics['f1']:.3f} < {F1_FLOOR}"
    assert metrics["pr_auc"] >= PR_AUC_FLOOR, f"PR-AUC régressé : {metrics['pr_auc']:.3f} < {PR_AUC_FLOOR}"
    assert metrics["recall"] >= RECALL_FLOOR, f"Recall régressé : {metrics['recall']:.3f} < {RECALL_FLOOR}"
