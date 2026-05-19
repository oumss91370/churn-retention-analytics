"""Schémas Pydantic pour validation des payloads API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientPayload(BaseModel):
    """Représentation d'un client en entrée du modèle.

    Tous les champs sont obligatoires sauf `complaint_type` qui accepte `"None"`.
    Les bornes sont alignées sur le dataset d'entraînement.
    """

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    age: int = Field(ge=18, le=99)
    country: str = Field(min_length=1, max_length=64)
    city: str = Field(min_length=1, max_length=64)
    customer_segment: Literal["Individual", "SME", "Enterprise"]
    tenure_months: int = Field(ge=0, le=240)
    signup_channel: Literal["Web", "Mobile", "Referral"]
    contract_type: Literal["Monthly", "Quarterly", "Yearly"]
    monthly_logins: int = Field(ge=0, le=500)
    weekly_active_days: int = Field(ge=0, le=7)
    avg_session_time: float = Field(ge=0, le=600)
    features_used: int = Field(ge=0, le=100)
    usage_growth_rate: float = Field(ge=-1.0, le=5.0)
    last_login_days_ago: int = Field(ge=0, le=3650)
    monthly_fee: float = Field(ge=0, le=10000)
    total_revenue: float = Field(ge=0, le=1_000_000)
    payment_method: Literal["Card", "PayPal", "Bank Transfer"]
    payment_failures: int = Field(ge=0, le=200)
    discount_applied: Literal["Yes", "No"]
    price_increase_last_3m: Literal["Yes", "No"]
    support_tickets: int = Field(ge=0, le=500)
    avg_resolution_time: float = Field(ge=0, le=10000)
    complaint_type: Literal["Technical", "Billing", "Service", "None"] = "None"
    csat_score: float = Field(ge=1, le=5)
    escalations: int = Field(ge=0, le=200)
    email_open_rate: float = Field(ge=0, le=1)
    marketing_click_rate: float = Field(ge=0, le=1)
    nps_score: int = Field(ge=-100, le=100)
    survey_response: Literal["Satisfied", "Neutral", "Unsatisfied"]
    referral_count: int = Field(ge=0, le=1000)


class TopFactor(BaseModel):
    """Un facteur contributif à la prédiction."""

    feature: str
    label: str
    shap_value: float
    direction: Literal["augmente", "diminue"]


class PredictResponse(BaseModel):
    """Réponse d'un appel à `POST /predict`."""

    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: Literal[0, 1]
    risk_level: Literal["faible", "modéré", "élevé"]
    decision_threshold: float = Field(ge=0, le=1)
    top_factors: list[TopFactor]


class HealthResponse(BaseModel):
    """Réponse de `GET /health`."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Réponse de `GET /model-info`."""

    model_type: str
    model_class: str
    training_date: str
    decision_threshold: float
    metrics: dict[str, float | int | str]
    n_features_out: int
    revenue_median_train: float
    random_state: int
