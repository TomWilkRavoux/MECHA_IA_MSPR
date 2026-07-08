"""Tests d'intégration de l'API d'exposition du LSTM.

Lancement (depuis Tom/) : uv run pytest
Les tests d'inférence utilisent un petit **fixture** (`tests/fixtures/`) et sont
ignorés si les modèles sont absents de models/. La validation du contrat d'entrée,
elle, ne dépend ni des modèles ni des données.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from ml.prep import FEATURES, MODELS_DIR

# Échantillon léger versionné (3 machines) : évite de dépendre du jeu de test complet
# (20 Mo, non versionné) et permet aux tests d'inférence de tourner en CI.
FIXTURE = Path(__file__).parent / "fixtures" / "mecha_test_sample.csv"

_MODELS_PRESENT = all(
    (MODELS_DIR / f).exists() for f in ("scaler.joblib", "lstm_classifier.pt", "lstm_regressor.pt")
)
needs_models = pytest.mark.skipif(not _MODELS_PRESENT, reason="modèles LSTM absents de models/")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample() -> pl.DataFrame:
    return pl.read_csv(FIXTURE)


def _cycles_for(df: pl.DataFrame, machine_id) -> list[dict]:
    g = df.filter(pl.col("machine_id") == machine_id).sort("cycle")
    return [{"values": {f: float(r[f]) for f in FEATURES}} for r in g.iter_rows(named=True)]


def test_health(client):
    body = client.get("/health").json()
    assert body["n_features"] == len(FEATURES)
    assert body["seq_len"] == 30
    # Seuils métier exposés pour le paramétrage du dashboard.
    assert body["risk_threshold"] == 30
    assert 0.0 <= body["critical_proba"] <= 1.0
    assert 0 < body["critical_rul"] <= body["risk_threshold"]


def test_features(client):
    body = client.get("/features").json()
    assert body["seq_len"] == 30
    assert body["features"] == FEATURES
    assert len(body["features"]) == 24


def test_missing_feature_returns_422(client):
    r = client.post("/predict", json={"machine_id": "X", "cycles": [{"values": {"T2": 1.0}}]})
    assert r.status_code == 422


def test_empty_cycles_returns_422(client):
    # `cycles` a min_length=1 -> une liste vide est rejetée par le contrat Pydantic.
    r = client.post("/predict", json={"machine_id": "X", "cycles": []})
    assert r.status_code == 422


def test_batch_empty_returns_422(client):
    # `machines` a min_length=1 -> un lot vide est rejeté.
    r = client.post("/predict/batch", json={"machines": []})
    assert r.status_code == 422


@needs_models
def test_predict_contract(client, sample):
    mid = sample["machine_id"][0]
    r = client.post("/predict", json={"machine_id": str(mid), "cycles": _cycles_for(sample, mid)})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert isinstance(body["at_risk"], bool)
    assert body["alert_level"] in {"ok", "warning", "critical"}
    # cohérence at_risk (classif) <-> alerte
    assert (body["alert_level"] == "ok") == (not body["at_risk"])


@needs_models
def test_batch(client, sample):
    ids = sample["machine_id"].unique().to_list()[:3]
    machines = [{"machine_id": str(m), "cycles": _cycles_for(sample, m)} for m in ids]
    r = client.post("/predict/batch", json={"machines": machines})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


@needs_models
def test_batch_matches_single(client, sample):
    # Le batch groupé (un seul forward pour tout le parc) doit renvoyer EXACTEMENT
    # le même résultat que N appels /predict unitaires, ordre d'entrée préservé.
    ids = sample["machine_id"].unique().to_list()[:3]
    machines = [{"machine_id": str(m), "cycles": _cycles_for(sample, m)} for m in ids]

    singles = [client.post("/predict", json=m).json() for m in machines]
    batch = client.post("/predict/batch", json={"machines": machines}).json()["results"]

    assert [b["machine_id"] for b in batch] == [str(m) for m in ids]
    assert batch == singles


@needs_models
def test_trajectory_contract(client, sample):
    mid = sample["machine_id"][0]
    cycles = _cycles_for(sample, mid)
    r = client.post("/predict/trajectory", json={"machine_id": str(mid), "cycles": cycles})
    assert r.status_code == 200
    points = r.json()["points"]
    # Un point par cycle observé, aligné sur l'ordre d'entrée.
    assert len(points) == len(cycles)
    assert [p["cycle_index"] for p in points] == list(range(len(cycles)))
    for p in points:
        assert 0.0 <= p["risk_probability"] <= 1.0
        assert p["alert_level"] in {"ok", "warning", "critical"}
        # cohérence at_risk (classif) <-> alerte
        assert (p["alert_level"] == "ok") == (not p["at_risk"])
