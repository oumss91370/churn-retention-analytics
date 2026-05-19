"""Application FastAPI exposant le modèle de churn."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

from api.predict import (
    InferenceContext,
    get_context,
    is_loaded,
    load_artifacts,
    predict_one,
    unload,
)
from api.schemas import (
    ClientPayload,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Charge les artefacts au démarrage et nettoie à l'arrêt."""
    try:
        load_artifacts()
        logger.info("Modèle chargé avec succès")
    except FileNotFoundError as exc:
        logger.warning("Modèle indisponible au démarrage : %s", exc)
    yield
    unload()


app = FastAPI(
    title="Churn Retention Analytics API",
    version="1.0.0",
    description=(
        "Service d'inférence du modèle de prédiction de churn. "
        "Le dashboard consomme cette API en HTTP."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirige vers la documentation Swagger."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Sonde de santé."""
    if not is_loaded():
        return HealthResponse(status="degraded", model_loaded=False, model_version=None)
    ctx: InferenceContext = get_context()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=str(ctx.metadata.get("model_type", "unknown")),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info() -> ModelInfoResponse:
    """Métadonnées du modèle chargé."""
    if not is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé",
        )
    ctx = get_context()
    meta = ctx.metadata
    metrics_clean = {
        k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
        for k, v in meta.get("metrics", {}).items()
    }
    return ModelInfoResponse(
        model_type=str(meta.get("model_type", "unknown")),
        model_class=str(meta.get("model_class", "unknown")),
        training_date=str(meta.get("training_date", "unknown")),
        decision_threshold=float(meta.get("decision_threshold", 0.5)),
        metrics=metrics_clean,
        n_features_out=int(meta.get("n_features_out", 0)),
        revenue_median_train=float(meta.get("revenue_median_train", 0.0)),
        random_state=int(meta.get("random_state", 42)),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: ClientPayload) -> PredictResponse:
    """Prédiction de churn pour un client."""
    if not is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé",
        )
    result = predict_one(payload.model_dump())
    return PredictResponse(**result)
