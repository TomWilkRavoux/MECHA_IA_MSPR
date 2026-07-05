"""Application FastAPI - exposition du LSTM de maintenance prédictive MECHA.

Lancement (depuis Tom/) :
    uv run uvicorn backend.api.main:app --reload

Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ml.prep import FEATURES, RISK_THRESHOLD, SEQ_LEN

from .inference import CRITICAL_PROBA, CRITICAL_RUL, service
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrajectoryResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charge les modèles au démarrage (échec explicite si models/ absent).
    try:
        service.load()
    except Exception as exc:  # noqa: BLE001 - on veut démarrer et le signaler via /health
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="MECHA - API Maintenance Prédictive (LSTM)",
    description=(
        "Expose le modèle LSTM (classification `at_risk` + régression `RUL`) "
        "pour l'exploitation métier : alertes et priorisation des interventions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Sonde de vivacité/readiness (utile pour Docker / CI / orchestration)."""
    return HealthResponse(
        status="ok" if service.ready else "degraded",
        models_loaded=service.ready,
        device=service.device,
        seq_len=SEQ_LEN,
        n_features=len(FEATURES),
        risk_threshold=RISK_THRESHOLD,
        critical_rul=CRITICAL_RUL,
        critical_proba=CRITICAL_PROBA,
    )


@app.get("/features", tags=["monitoring"])
def features() -> dict:
    """Liste ordonnée des variables attendues par cycle (contrat d'entrée)."""
    return {"seq_len": SEQ_LEN, "features": FEATURES}


def _predict_one(req: PredictRequest) -> PredictResponse:
    cycles = [c.values for c in req.cycles]
    out = service.predict(cycles)
    return PredictResponse(machine_id=req.machine_id, **out)


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Prédit l'état `at_risk` et le RUL d'une machine à partir de ses cycles."""
    if not service.ready:
        raise HTTPException(
            status_code=503, detail="Modèles non chargés (voir /health)."
        )
    return _predict_one(req)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Prédiction pour un lot de machines (supervision d'un parc / d'une ligne)."""
    if not service.ready:
        raise HTTPException(
            status_code=503, detail="Modèles non chargés (voir /health)."
        )
    return BatchPredictResponse(results=[_predict_one(m) for m in req.machines])


@app.post("/predict/trajectory", response_model=TrajectoryResponse, tags=["prediction"])
def predict_trajectory(req: PredictRequest) -> TrajectoryResponse:
    """Trajectoire cycle par cycle d'UNE machine (RUL et proba de risque au fil des cycles)."""
    if not service.ready:
        raise HTTPException(
            status_code=503, detail="Modèles non chargés (voir /health)."
        )
    points = service.predict_trajectory([c.values for c in req.cycles])
    return TrajectoryResponse(machine_id=req.machine_id, points=points)
