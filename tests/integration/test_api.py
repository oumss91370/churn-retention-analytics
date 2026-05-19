"""Tests d'intégration de l'API FastAPI via TestClient (in-process)."""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import predict as predict_module
from api.main import app
from src.config import MODEL_PATH, PREPROCESSOR_PATH


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not MODEL_PATH.exists():
        pytest.skip("Modèle non sérialisé, lancer `python -m src.models` d'abord")
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] is not None


def test_predict_valid_payload(client: TestClient, sample_payload: dict) -> None:
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert "risk_level" in body
    assert "top_factors" in body


def test_predict_probability_in_range(client: TestClient, sample_payload: dict) -> None:
    r = client.post("/predict", json=sample_payload)
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_risk_level_consistency(client: TestClient, sample_payload: dict) -> None:
    payload = copy.deepcopy(sample_payload)
    payload["nps_score"] = 90
    payload["csat_score"] = 5.0
    payload["last_login_days_ago"] = 0
    payload["payment_failures"] = 0
    payload["support_tickets"] = 0
    payload["usage_growth_rate"] = 0.4
    payload["contract_type"] = "Yearly"
    payload["survey_response"] = "Satisfied"
    r = client.post("/predict", json=payload)
    body = r.json()
    assert body["risk_level"] in {"faible", "modéré", "élevé"}


def test_predict_top_factors_length(client: TestClient, sample_payload: dict) -> None:
    r = client.post("/predict", json=sample_payload)
    body = r.json()
    assert isinstance(body["top_factors"], list)
    assert 1 <= len(body["top_factors"]) <= 5


def test_predict_missing_field_returns_422(client: TestClient, sample_payload: dict) -> None:
    payload = copy.deepcopy(sample_payload)
    payload.pop("nps_score")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_type_returns_422(client: TestClient, sample_payload: dict) -> None:
    payload = copy.deepcopy(sample_payload)
    payload["age"] = "trente"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_out_of_range_returns_422(client: TestClient, sample_payload: dict) -> None:
    payload = copy.deepcopy(sample_payload)
    payload["nps_score"] = 200
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_unknown_enum_returns_422(client: TestClient, sample_payload: dict) -> None:
    payload = copy.deepcopy(sample_payload)
    payload["gender"] = "Other"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_idempotent(client: TestClient, sample_payload: dict) -> None:
    r1 = client.post("/predict", json=sample_payload).json()
    r2 = client.post("/predict", json=sample_payload).json()
    assert r1["churn_probability"] == pytest.approx(r2["churn_probability"], abs=1e-6)
    assert r1["churn_prediction"] == r2["churn_prediction"]


def test_model_info_returns_metadata(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"]
    assert body["training_date"]
    assert 0.0 <= body["decision_threshold"] <= 1.0


def test_crit06_api_matches_direct_inference(client: TestClient, sample_payload: dict) -> None:
    """CRIT-06 : la réponse API doit refléter exactement l'inférence directe."""
    ctx = predict_module.get_context()
    direct = predict_module.predict_one(sample_payload)
    api_response = client.post("/predict", json=sample_payload).json()
    assert api_response["churn_probability"] == pytest.approx(direct["churn_probability"], abs=1e-9)
    assert api_response["churn_prediction"] == direct["churn_prediction"]
    assert api_response["risk_level"] == direct["risk_level"]
