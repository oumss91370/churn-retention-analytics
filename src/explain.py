"""Interprétabilité SHAP et importance de features pour le modèle final."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


BUSINESS_LABELS: dict[str, str] = {
    "age": "Âge",
    "tenure_months": "Ancienneté (mois)",
    "monthly_logins": "Connexions / mois",
    "weekly_active_days": "Jours actifs / semaine",
    "avg_session_time": "Durée moyenne d'une session",
    "features_used": "Fonctionnalités utilisées",
    "usage_growth_rate": "Évolution d'usage",
    "last_login_days_ago": "Dernière connexion (jours)",
    "monthly_fee": "Abonnement mensuel",
    "total_revenue": "Revenu total cumulé",
    "payment_failures": "Échecs de paiement",
    "support_tickets": "Tickets support",
    "avg_resolution_time": "Temps moyen de résolution",
    "csat_score": "Score CSAT",
    "escalations": "Escalades support",
    "email_open_rate": "Taux d'ouverture e-mail",
    "marketing_click_rate": "Taux de clic marketing",
    "nps_score": "Score NPS",
    "referral_count": "Parrainages",
    "tickets_per_month": "Tickets / mois (normalisé)",
    "failed_payment_rate": "Taux d'échec de paiement",
    "is_high_value": "Client à forte valeur",
    "engagement_drop": "Baisse d'engagement",
    "gender": "Genre",
    "country": "Pays",
    "city": "Ville",
    "customer_segment": "Segment client",
    "signup_channel": "Canal d'inscription",
    "contract_type": "Type de contrat",
    "payment_method": "Moyen de paiement",
    "discount_applied": "Remise appliquée",
    "price_increase_last_3m": "Hausse de prix récente",
    "complaint_type": "Type de plainte",
    "survey_response": "Réponse au sondage",
}


def to_business_label(feature: str) -> str:
    """Convertit un nom technique (potentiellement avec préfixe OHE) en libellé métier."""
    base = feature
    for sep in ("_", "="):
        for cat in (
            "gender",
            "country",
            "city",
            "customer_segment",
            "signup_channel",
            "contract_type",
            "payment_method",
            "discount_applied",
            "price_increase_last_3m",
            "complaint_type",
            "survey_response",
        ):
            prefix = f"{cat}_"
            if base.startswith(prefix):
                value = base[len(prefix):]
                return f"{BUSINESS_LABELS.get(cat, cat)} = {value}"
    return BUSINESS_LABELS.get(base, base.replace("_", " ").capitalize())


def _to_dense(arr) -> np.ndarray:
    """Convertit potentiellement une matrice sparse en dense."""
    if hasattr(arr, "toarray"):
        return arr.toarray()
    return np.asarray(arr)


def make_explainer(model, X_background: np.ndarray):
    """Construit le SHAP explainer adapté au type de modèle."""
    model_name = type(model).__name__.lower()
    X_dense = _to_dense(X_background)

    if "xgb" in model_name or "forest" in model_name or "tree" in model_name or "gbm" in model_name:
        return shap.TreeExplainer(model)

    if hasattr(model, "predict_proba"):
        def predict_fn(data):
            return model.predict_proba(data)[:, 1]
    else:
        predict_fn = model.predict

    sample_size = min(100, X_dense.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_dense.shape[0], size=sample_size, replace=False)
    return shap.KernelExplainer(predict_fn, X_dense[idx])


def compute_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """Calcule les valeurs SHAP. Retourne un ndarray 2D (n_samples, n_features) pour la classe positive."""
    X_dense = _to_dense(X)
    raw = explainer.shap_values(X_dense)
    if isinstance(raw, list):
        if len(raw) == 2:
            arr = np.asarray(raw[1])
        else:
            arr = np.asarray(raw[0])
    else:
        arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr[:, :, 0]
    return arr


def top_factors_for_instance(
    shap_row: np.ndarray,
    feature_names: list[str],
    k: int = 3,
) -> list[dict]:
    """Retourne les top-k facteurs d'une prédiction individuelle.

    Args:
        shap_row: vecteur SHAP de longueur n_features.
        feature_names: noms techniques alignés.
        k: nombre de facteurs à retourner.

    Returns:
        Liste de dictionnaires triée par |shap| décroissant.
    """
    arr = np.asarray(shap_row).ravel()
    order = np.argsort(-np.abs(arr))[:k]
    out: list[dict] = []
    for idx in order:
        out.append(
            {
                "feature": feature_names[idx],
                "label": to_business_label(feature_names[idx]),
                "shap_value": float(arr[idx]),
                "direction": "augmente" if arr[idx] > 0 else "diminue",
            }
        )
    return out


def save_global_bar_plot(
    shap_values: np.ndarray,
    feature_names: list[str],
    path: Path | str,
    top: int = 10,
) -> Path:
    """Sauvegarde le SHAP summary bar plot."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)[:top]
    labels = [to_business_label(feature_names[i]) for i in order]
    values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(labels)), values[::-1])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Impact moyen sur la prédiction (|SHAP|)")
    ax.set_title("Top facteurs explicatifs — Global")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    path: Path | str,
    top: int = 10,
) -> Path:
    """Sauvegarde un summary beeswarm SHAP."""
    X_dense = _to_dense(X)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)[:top]
    sv = shap_values[:, order]
    xv = X_dense[:, order]
    labels = [to_business_label(feature_names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(42)
    for i in range(len(order)):
        jitter = rng.normal(scale=0.08, size=sv.shape[0])
        normalised = xv[:, i]
        if normalised.std() > 0:
            normalised = (normalised - normalised.min()) / (normalised.max() - normalised.min() + 1e-9)
        ax.scatter(
            sv[:, i],
            np.full(sv.shape[0], i) + jitter,
            c=normalised,
            cmap="coolwarm",
            s=8,
            alpha=0.6,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Valeur SHAP")
    ax.set_title("Distribution des contributions — Top features")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_force_plot(
    shap_row: np.ndarray,
    feature_names: list[str],
    path: Path | str,
    title: str,
    top: int = 8,
) -> Path:
    """Sauvegarde un force plot horizontal lisible (top contributeurs d'une instance)."""
    arr = np.asarray(shap_row).ravel()
    order = np.argsort(-np.abs(arr))[:top]
    contribs = arr[order]
    labels = [to_business_label(feature_names[i]) for i in order]
    colors = ["#d35454" if v > 0 else "#3a86ff" for v in contribs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(labels)), contribs[::-1], color=colors[::-1])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Contribution SHAP (négatif = vers reste, positif = vers churn)")
    ax.set_title(title)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def aggregate_business_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aggrège les valeurs |SHAP| par variable métier d'origine (en sommant les modalités OHE)."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    rows = []
    for name, val in zip(feature_names, mean_abs):
        base = name
        for cat in BUSINESS_LABELS:
            if name.startswith(f"{cat}_") and cat in (
                "gender",
                "country",
                "city",
                "customer_segment",
                "signup_channel",
                "contract_type",
                "payment_method",
                "discount_applied",
                "price_increase_last_3m",
                "complaint_type",
                "survey_response",
            ):
                base = cat
                break
        rows.append({"variable": base, "label": BUSINESS_LABELS.get(base, base), "abs_shap": float(val)})
    agg = (
        pd.DataFrame(rows)
        .groupby(["variable", "label"], as_index=False)["abs_shap"]
        .sum()
        .sort_values("abs_shap", ascending=False)
    )
    return agg
