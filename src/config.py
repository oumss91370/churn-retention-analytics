"""Constantes du projet : graine, chemins, listes de colonnes, seuils."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_CSV: Path = DATA_DIR / "raw" / "customer_churn.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

TARGET: str = "churn"
ID_COL: str = "customer_id"

NUMERIC_FEATURES: list[str] = [
    "age",
    "tenure_months",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
    "monthly_fee",
    "total_revenue",
    "payment_failures",
    "support_tickets",
    "avg_resolution_time",
    "csat_score",
    "escalations",
    "email_open_rate",
    "marketing_click_rate",
    "nps_score",
    "referral_count",
]

CATEGORICAL_FEATURES: list[str] = [
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
]

DERIVED_NUMERIC_FEATURES: list[str] = [
    "tickets_per_month",
    "failed_payment_rate",
    "is_high_value",
    "engagement_drop",
]

RISK_THRESHOLD_LOW: float = 0.30
RISK_THRESHOLD_HIGH: float = 0.60

MODEL_PATH: Path = MODELS_DIR / "final_model.pkl"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"


def ensure_dirs() -> None:
    """Crée les répertoires de sortie s'ils n'existent pas."""
    for directory in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
