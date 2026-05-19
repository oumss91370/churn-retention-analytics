"""Fixtures partagées de la suite de tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE  # noqa: E402  pylint: disable=wrong-import-position
from src.data import load_raw  # noqa: E402


def _generate_mock_data(n: int = 200, churn_rate: float = 0.15, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Génère un DataFrame synthétique au schéma identique au vrai dataset."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * churn_rate))
    n_neg = n - n_pos

    base = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "age": rng.integers(18, 75, size=n),
            "country": rng.choice(
                ["USA", "UK", "Germany", "India", "Canada", "Australia", "Bangladesh"],
                size=n,
            ),
            "city": rng.choice(
                ["Paris", "London", "Berlin", "Mumbai", "Toronto", "Sydney", "Dhaka"],
                size=n,
            ),
            "customer_segment": rng.choice(["Individual", "SME", "Enterprise"], size=n),
            "tenure_months": rng.integers(1, 60, size=n),
            "signup_channel": rng.choice(["Web", "Mobile", "Referral"], size=n),
            "contract_type": rng.choice(["Monthly", "Quarterly", "Yearly"], size=n),
            "monthly_logins": rng.integers(0, 55, size=n),
            "weekly_active_days": rng.integers(0, 8, size=n),
            "avg_session_time": rng.uniform(1, 40, size=n).round(2),
            "features_used": rng.integers(1, 15, size=n),
            "usage_growth_rate": rng.normal(0.0, 0.15, size=n).round(3),
            "last_login_days_ago": rng.integers(0, 80, size=n),
            "monthly_fee": rng.integers(10, 100, size=n),
            "total_revenue": rng.integers(10, 5000, size=n),
            "payment_method": rng.choice(["Card", "PayPal", "Bank Transfer"], size=n),
            "payment_failures": rng.integers(0, 6, size=n),
            "discount_applied": rng.choice(["Yes", "No"], size=n),
            "price_increase_last_3m": rng.choice(["Yes", "No"], size=n),
            "support_tickets": rng.integers(0, 8, size=n),
            "avg_resolution_time": rng.uniform(1, 60, size=n).round(2),
            "complaint_type": rng.choice(["Technical", "Billing", "Service", None], size=n),
            "csat_score": rng.uniform(1, 5, size=n).round(1),
            "escalations": rng.integers(0, 5, size=n),
            "email_open_rate": rng.uniform(0.1, 0.9, size=n).round(2),
            "marketing_click_rate": rng.uniform(0.01, 0.5, size=n).round(2),
            "nps_score": rng.integers(-100, 101, size=n),
            "survey_response": rng.choice(["Satisfied", "Neutral", "Unsatisfied"], size=n),
            "referral_count": rng.integers(0, 8, size=n),
        }
    )
    base["churn"] = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    base = base.sample(frac=1, random_state=seed).reset_index(drop=True)
    return base


@pytest.fixture(scope="session")
def mock_df() -> pd.DataFrame:
    """Dataset synthétique de 200 lignes pour les tests rapides."""
    return _generate_mock_data(n=200, churn_rate=0.15, seed=RANDOM_STATE)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Dataset brut réel (chargé une seule fois pour la session)."""
    return load_raw()


@pytest.fixture
def sample_payload() -> dict:
    """Payload complet et valide pour `POST /predict`."""
    return {
        "gender": "Female",
        "age": 38,
        "country": "USA",
        "city": "Paris",
        "customer_segment": "SME",
        "tenure_months": 12,
        "signup_channel": "Web",
        "contract_type": "Monthly",
        "monthly_logins": 22,
        "weekly_active_days": 4,
        "avg_session_time": 18.5,
        "features_used": 5,
        "usage_growth_rate": -0.05,
        "last_login_days_ago": 6,
        "monthly_fee": 45,
        "total_revenue": 540,
        "payment_method": "Card",
        "payment_failures": 1,
        "discount_applied": "No",
        "price_increase_last_3m": "Yes",
        "support_tickets": 2,
        "avg_resolution_time": 24.0,
        "complaint_type": "Billing",
        "csat_score": 3.2,
        "escalations": 0,
        "email_open_rate": 0.42,
        "marketing_click_rate": 0.18,
        "nps_score": -10,
        "survey_response": "Neutral",
        "referral_count": 1,
    }
