"""Schémas d'entrée / sortie de l'API (contrats Pydantic).

Une requête = les cycles capteurs d'UNE machine (série temporelle). Le service
reconstruit la fenêtre glissante attendue par le LSTM et renvoie l'état `at_risk`
et le RUL estimé, enrichis d'un niveau d'alerte métier.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ml.prep import FEATURES, RISK_THRESHOLD, SEQ_LEN


class Reading(BaseModel):
    """Relevé capteurs d'un cycle machine : les 24 variables du modèle.

    Les clés doivent correspondre exactement à `ml.prep.FEATURES`
    (3 `setting_*` + 21 capteurs). Valeurs numériques brutes (non normalisées) :
    la normalisation (StandardScaler entraîné) est appliquée côté serveur.
    """

    values: dict[str, float] = Field(
        ...,
        description=f"Dict {{nom_variable: valeur}} couvrant les {len(FEATURES)} features.",
    )

    @field_validator("values")
    @classmethod
    def _check_features(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [f for f in FEATURES if f not in v]
        if missing:
            raise ValueError(f"Variables manquantes: {missing}")
        return v


class PredictRequest(BaseModel):
    machine_id: str = Field(..., description="Identifiant machine (traçabilité).")
    cycles: list[Reading] = Field(
        ...,
        min_length=1,
        description=(
            f"Cycles ordonnés du plus ancien au plus récent. Le modèle utilise les "
            f"{SEQ_LEN} derniers ; si moins de {SEQ_LEN} sont fournis, le premier "
            f"cycle est répété (left-padding), comme à l'entraînement."
        ),
    )


class BatchPredictRequest(BaseModel):
    machines: list[PredictRequest] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    machine_id: str
    at_risk: bool = Field(..., description="Décision de la tête classification (proba >= 0.5).")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité classe 'à risque'.")
    rul_predicted: float = Field(..., description="RUL estimé (cycles restants avant défaillance).")
    alert_level: str = Field(..., description="ok | warning | critical (règle métier).")
    threshold: int = Field(RISK_THRESHOLD, description="Seuil at_risk utilisé.")
    n_cycles_used: int = Field(..., description="Nombre de cycles réellement utilisés (<= SEQ_LEN).")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class TrajectoryPoint(BaseModel):
    """Prédiction pour un cycle de la trajectoire (fenêtre glissante s'y terminant)."""

    cycle_index: int = Field(..., description="Position du cycle dans la série fournie (0-indexé).")
    at_risk: bool = Field(..., description="Décision de la tête classification (proba >= 0.5).")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité classe 'à risque'.")
    rul_predicted: float = Field(..., description="RUL estimé à ce cycle (cycles restants).")
    alert_level: str = Field(..., description="ok | warning | critical (règle métier).")


class TrajectoryResponse(BaseModel):
    machine_id: str
    points: list[TrajectoryPoint]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    seq_len: int
    n_features: int
    risk_threshold: int = Field(..., description="Seuil at_risk (RUL <= seuil).")
    critical_rul: int = Field(..., description="RUL en-deçà duquel l'alerte passe 'critical'.")
    critical_proba: float = Field(..., description="Proba au-delà de laquelle l'alerte passe 'critical'.")
