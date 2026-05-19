"""Tests end-to-end : un client entre, une décision business sort."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import MODEL_PATH

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not MODEL_PATH.exists():
        pytest.skip("Modèle non sérialisé")
    with TestClient(app) as test_client:
        yield test_client


def test_full_path_low_risk_to_high_risk(client: TestClient, sample_payload: dict) -> None:
    """Trajectoire complète : un client serein devient à risque quand on dégrade ses signaux."""
    serein = copy.deepcopy(sample_payload)
    serein.update(
        nps_score=85,
        csat_score=5.0,
        survey_response="Satisfied",
        last_login_days_ago=0,
        usage_growth_rate=0.4,
        payment_failures=0,
        support_tickets=0,
        contract_type="Yearly",
        price_increase_last_3m="No",
    )
    risque = copy.deepcopy(sample_payload)
    risque.update(
        nps_score=-80,
        csat_score=1.5,
        survey_response="Unsatisfied",
        last_login_days_ago=45,
        usage_growth_rate=-0.4,
        payment_failures=5,
        support_tickets=7,
        contract_type="Monthly",
        price_increase_last_3m="Yes",
    )

    r1 = client.post("/predict", json=serein).json()
    r2 = client.post("/predict", json=risque).json()

    assert r2["churn_probability"] > r1["churn_probability"]
    assert r2["risk_level"] in {"modéré", "élevé"}


def test_full_pipeline_response_schema(client: TestClient, sample_payload: dict) -> None:
    """Le contrat de réponse `POST /predict` est respecté pour un client réaliste."""
    r = client.post("/predict", json=sample_payload).json()
    expected = {"churn_probability", "churn_prediction", "risk_level", "decision_threshold", "top_factors"}
    assert expected.issubset(set(r.keys()))
    assert r["churn_prediction"] in {0, 1}
    assert r["risk_level"] in {"faible", "modéré", "élevé"}
    for factor in r["top_factors"]:
        assert set(factor.keys()).issuperset({"feature", "label", "shap_value", "direction"})
