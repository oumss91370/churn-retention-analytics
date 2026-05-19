"""Helpers du dashboard : formatages, conversions, labels métier."""

from __future__ import annotations


RISK_LOW_THRESHOLD = 0.30
RISK_HIGH_THRESHOLD = 0.60


BUSINESS_LABELS = {
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
    "payment_method": "Moyen de paiement",
    "payment_failures": "Échecs de paiement",
    "discount_applied": "Remise appliquée",
    "price_increase_last_3m": "Hausse de prix récente",
    "support_tickets": "Tickets support",
    "avg_resolution_time": "Temps moyen de résolution",
    "complaint_type": "Type de plainte",
    "csat_score": "Score CSAT",
    "escalations": "Escalades support",
    "email_open_rate": "Taux d'ouverture e-mail",
    "marketing_click_rate": "Taux de clic marketing",
    "nps_score": "Score NPS",
    "survey_response": "Réponse au sondage",
    "referral_count": "Parrainages",
    "gender": "Genre",
    "country": "Pays",
    "city": "Ville",
    "customer_segment": "Segment client",
    "signup_channel": "Canal d'inscription",
    "contract_type": "Type de contrat",
}


def format_currency_eur(amount: float) -> str:
    """Formate une somme en euros avec séparateur de milliers (espace fine)."""
    if amount is None:
        return "—"
    sign = "-" if amount < 0 else ""
    value = abs(float(amount))
    integer = int(round(value))
    formatted = f"{integer:,}".replace(",", " ")
    return f"{sign}{formatted} €"


def format_number(value: float | int, decimals: int = 0) -> str:
    """Formate un nombre avec séparateur de milliers et décimales optionnelles."""
    if value is None:
        return "—"
    if decimals == 0:
        return f"{int(round(float(value))):,}".replace(",", " ")
    return f"{float(value):,.{decimals}f}".replace(",", " ")


def format_percent(value: float, decimals: int = 1) -> str:
    """Formate un ratio (0-1) en pourcentage."""
    if value is None:
        return "—"
    return f"{float(value) * 100:.{decimals}f} %".replace(".", ",")


def proba_to_risk_level(proba: float) -> str:
    """Conversion probabilité → niveau de risque."""
    if proba < RISK_LOW_THRESHOLD:
        return "faible"
    if proba < RISK_HIGH_THRESHOLD:
        return "modéré"
    return "élevé"


def revenue_at_risk(probas: list[float], revenues: list[float]) -> float:
    """Somme pondérée du revenu à risque."""
    if len(probas) != len(revenues):
        raise ValueError("probas et revenues doivent avoir la même longueur")
    return float(sum(p * r for p, r in zip(probas, revenues)))


def feature_name_to_business_label(name: str) -> str:
    """Convertit un nom de feature en libellé métier (avec gestion du OHE)."""
    if name in BUSINESS_LABELS:
        return BUSINESS_LABELS[name]
    for cat, label in BUSINESS_LABELS.items():
        prefix = f"{cat}_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            return f"{label} = {value}"
    return name.replace("_", " ").capitalize()


def recommendation_from_factors(factors: list[dict]) -> str:
    """Construit une recommandation textuelle à partir des top facteurs SHAP."""
    if not factors:
        return "Aucune recommandation disponible : facteurs explicatifs indisponibles."
    actionable = [f for f in factors if f["direction"] == "augmente"]
    if not actionable:
        actionable = factors
    parts = []
    for f in actionable[:3]:
        label = feature_name_to_business_label(f.get("feature", ""))
        parts.append(f"• {label}")
    head = (
        "Priorités d'action suggérées (variables qui poussent vers le churn) :"
        if any(f["direction"] == "augmente" for f in factors)
        else "Variables les plus contributives :"
    )
    return head + "\n" + "\n".join(parts)
