"""Tests unitaires (fonctions pures, sans modèles ni serveur).

Couvre la règle métier d'alerte, la construction des fenêtres glissantes, et la
mise en forme des requêtes côté frontend. Ces tests s'exécutent partout (CI incluse),
indépendamment de la présence des artefacts modèles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from api_client import build_requests, build_single_request

from backend.api.inference import CRITICAL_PROBA, CRITICAL_RUL, ModelService, _alert_level
from ml.prep import SEQ_LEN


# ---------------------------------------------------------------------------
# Règle métier d'alerte (_alert_level)
# ---------------------------------------------------------------------------
def test_alert_level_ok_when_not_at_risk():
    # Une machine non "à risque" est toujours OK, quels que soient RUL/proba.
    assert _alert_level(False, rul=1.0, proba=0.99) == "ok"


def test_alert_level_warning():
    # À risque mais loin des seuils critiques -> à surveiller.
    assert _alert_level(True, rul=CRITICAL_RUL + 5, proba=CRITICAL_PROBA - 0.1) == "warning"


def test_alert_level_critical_by_rul():
    assert _alert_level(True, rul=CRITICAL_RUL, proba=0.1) == "critical"


def test_alert_level_critical_by_proba():
    assert _alert_level(True, rul=CRITICAL_RUL + 50, proba=CRITICAL_PROBA) == "critical"


# ---------------------------------------------------------------------------
# Fenêtre glissante (ModelService._window_ending_at)
# ---------------------------------------------------------------------------
def _markers(n: int, n_features: int = 3) -> np.ndarray:
    """Matrice (n, n_features) où chaque ligne i vaut i (marqueur d'ordre)."""
    return np.tile(np.arange(n, dtype="float32").reshape(-1, 1), (1, n_features))


def test_window_left_pads_short_trajectory():
    arr = _markers(5)
    window, n_used = ModelService._window_ending_at(arr, end=len(arr))
    assert window.shape == (SEQ_LEN, 3)
    assert n_used == 5
    # Left-pad : les (SEQ_LEN - 5) premières lignes répètent le 1er cycle réel (0)...
    assert np.all(window[: SEQ_LEN - 5] == 0.0)
    # ...et les 5 dernières sont la trajectoire réelle 0..4.
    assert np.array_equal(window[-5:], arr)


def test_window_takes_last_seq_len_when_long():
    arr = _markers(SEQ_LEN + 10)  # 40 cycles
    window, n_used = ModelService._window_ending_at(arr, end=len(arr))
    assert window.shape == (SEQ_LEN, 3)
    assert n_used == SEQ_LEN
    # Pas de padding : exactement les SEQ_LEN derniers cycles (10..39).
    assert np.array_equal(window, arr[-SEQ_LEN:])


def test_window_ending_mid_trajectory():
    arr = _markers(5)
    window, n_used = ModelService._window_ending_at(arr, end=3)
    assert n_used == 3
    assert np.array_equal(window[-3:], arr[:3])  # cycles 0,1,2


# ---------------------------------------------------------------------------
# Mise en forme des requêtes (api_client)
# ---------------------------------------------------------------------------
_FEATS = ["T2", "T24"]


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": ["M1", "M1", "M2"],
            "cycle": [2, 1, 5],  # volontairement désordonné
            "usine": ["A", "A", "B"],  # colonne méta -> ignorée
            "T2": [10.0, 11.0, 20.0],
            "T24": [30.0, 31.0, 40.0],
        }
    )


def test_build_single_request_sorts_and_filters_features():
    df = _sample_df()
    g = df[df["machine_id"] == "M1"]
    req = build_single_request("M1", g, _FEATS)
    assert req["machine_id"] == "M1"
    # Ordonné par cycle croissant (1 puis 2) -> T2 = 11.0 puis 10.0.
    assert [c["values"]["T2"] for c in req["cycles"]] == [11.0, 10.0]
    # Seules les features sont envoyées (pas de méta).
    assert set(req["cycles"][0]["values"]) == set(_FEATS)


def test_build_requests_groups_by_machine():
    reqs = build_requests(_sample_df(), _FEATS)
    assert {r["machine_id"] for r in reqs} == {"M1", "M2"}
    m1 = next(r for r in reqs if r["machine_id"] == "M1")
    assert len(m1["cycles"]) == 2
