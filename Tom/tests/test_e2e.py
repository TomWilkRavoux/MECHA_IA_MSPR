"""Test end-to-end : l'API interrogée en HTTP dans son **conteneur**.

Contrairement à `test_api.py` (TestClient FastAPI en process), ces tests tapent une
instance réellement démarrée — image Docker `mecha-backend`, modèles LSTM embarqués —
via la couche réseau. Ils valident l'assemblage complet (image + serveur + inférence).

Volontairement **sans dépendance lourde** (ni polars ni ml.prep) : seul `requests`
est requis. La liste des features est lue depuis l'API elle-même (`/features`), le
jeu d'exemple depuis le fixture CSV via la stdlib. Le job CI `e2e` reste donc léger.

Lancement local (depuis Tom/) :
    docker compose up -d --build backend
    MECHA_E2E_URL=http://localhost:8000 uv run pytest tests/test_e2e.py

En CI, le job `e2e` démarre le conteneur puis pointe `MECHA_E2E_URL` dessus. Les tests
sont ignorés (skip) si l'URL n'est pas joignable, pour rester inoffensifs hors contexte E2E.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest
import requests

BASE_URL = os.environ.get("MECHA_E2E_URL", "http://localhost:8000")
FIXTURE = Path(__file__).parent / "fixtures" / "mecha_test_sample.csv"


def _api_reachable() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


# Skip propre si aucune API n'écoute (ex. `pytest` lancé sans conteneur).
pytestmark = pytest.mark.skipif(
    not _api_reachable(), reason=f"API E2E injoignable sur {BASE_URL}"
)


@pytest.fixture(scope="module")
def features() -> list[str]:
    return requests.get(f"{BASE_URL}/features", timeout=5).json()["features"]


def _cycles_for(machine_id: str, features: list[str]) -> list[dict]:
    """Cycles d'une machine du fixture, formatés pour /predict (stdlib csv, pas de polars)."""
    with FIXTURE.open() as f:
        rows = [r for r in csv.DictReader(f) if r["machine_id"] == machine_id]
    rows.sort(key=lambda r: int(r["cycle"]))
    return [{"values": {feat: float(r[feat]) for feat in features}} for r in rows]


def _first_machine_ids(n: int = 3) -> list[str]:
    with FIXTURE.open() as f:
        ids = list(dict.fromkeys(r["machine_id"] for r in csv.DictReader(f)))
    return ids[:n]


def test_health_over_http():
    body = requests.get(f"{BASE_URL}/health", timeout=5).json()
    assert body["models_loaded"] is True  # l'image embarque bien les artefacts
    assert body["n_features"] == 24
    assert body["seq_len"] == 30


def test_predict_over_http(features):
    mid = _first_machine_ids(1)[0]
    payload = {"machine_id": mid, "cycles": _cycles_for(mid, features)}
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert isinstance(body["at_risk"], bool)
    assert body["alert_level"] in {"ok", "warning", "critical"}
    assert isinstance(body["rul_predicted"], (int, float))


def test_batch_over_http(features):
    ids = _first_machine_ids(3)
    machines = [{"machine_id": m, "cycles": _cycles_for(m, features)} for m in ids]
    r = requests.post(f"{BASE_URL}/predict/batch", json={"machines": machines}, timeout=15)
    assert r.status_code == 200
    assert len(r.json()["results"]) == len(ids)


def test_contract_over_http():
    # Le contrat d'entrée est aussi appliqué de bout en bout : feature manquante -> 422.
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"machine_id": "X", "cycles": [{"values": {"T2": 1.0}}]},
        timeout=5,
    )
    assert r.status_code == 422
