"""Tests unitaires de la chaîne MLOps (gate de validation + producteur de flux).

Fonctions pures uniquement (pas de modèles, pas de serveur, pas de données) :
exécutables partout, CI incluse. Le comportement bout-en-bout de `ml.train` et du
producteur réseau est validé manuellement (cf. docs/mlops.md).
"""

from __future__ import annotations

import pandas as pd

from ml.registry import next_registry
from ml.validate_metrics import check
from sim.producer import build_request, history_upto, machine_frames, select_machines


# ---------------------------------------------------------------------------
# Gate de validation (ml.validate_metrics.check)
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    "_comment": "ignoré",
    "classification": {"f1": {"min": 0.82}, "roc_auc": {"min": 0.95}},
    "regression": {"RMSE": {"max": 30.0}},
}


def test_check_passes_when_within_bounds():
    metrics = {"classification": {"f1": 0.885, "roc_auc": 0.993}, "regression": {"RMSE": 26.3}}
    rows = check(metrics, _THRESHOLDS)
    assert all(r["ok"] for r in rows)
    assert len(rows) == 3  # la clé de commentaire est ignorée


def test_check_fails_on_min_regression():
    metrics = {"classification": {"f1": 0.70, "roc_auc": 0.993}, "regression": {"RMSE": 26.3}}
    rows = check(metrics, _THRESHOLDS)
    failed = [r for r in rows if not r["ok"]]
    assert len(failed) == 1 and failed[0]["metric"] == "f1"


def test_check_fails_on_max_regression():
    metrics = {"classification": {"f1": 0.885, "roc_auc": 0.993}, "regression": {"RMSE": 35.0}}
    failed = [r for r in check(metrics, _THRESHOLDS) if not r["ok"]]
    assert len(failed) == 1 and failed[0]["metric"] == "RMSE"


def test_check_missing_metric_is_failure():
    metrics = {"classification": {"f1": 0.885}, "regression": {"RMSE": 26.3}}  # roc_auc absent
    failed = [r for r in check(metrics, _THRESHOLDS) if not r["ok"]]
    assert [r["metric"] for r in failed] == ["roc_auc"]
    assert failed[0]["value"] is None


# ---------------------------------------------------------------------------
# Registre de modèles versionné (ml.registry.next_registry) — décision pure
# ---------------------------------------------------------------------------
def _entry(run_id: str) -> dict:
    return {"run_id": run_id, "metrics": {}}


def test_first_run_becomes_current_even_without_promote():
    # Sans run courant, le premier run enregistré devient courant (sinon rien servi).
    reg = next_registry({"current": None, "runs": []}, _entry("R1"), promote=False)
    assert reg["current"] == "R1"
    assert [r["run_id"] for r in reg["runs"]] == ["R1"]


def test_no_promote_keeps_previous_current():
    base = {"current": "R1", "runs": [_entry("R1")]}
    reg = next_registry(base, _entry("R2"), promote=False)
    assert reg["current"] == "R1"  # baseline/served inchangé
    assert {r["run_id"] for r in reg["runs"]} == {"R1", "R2"}


def test_promote_switches_current_and_dedups():
    base = {"current": "R1", "runs": [_entry("R1"), _entry("R2")]}
    reg = next_registry(base, _entry("R2"), promote=True)  # ré-enregistre R2
    assert reg["current"] == "R2"
    assert [r["run_id"] for r in reg["runs"]] == ["R1", "R2"]  # pas de doublon


# ---------------------------------------------------------------------------
# Producteur de flux (sim.producer)
# ---------------------------------------------------------------------------
def _stream_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": ["M1", "M1", "M1", "M2", "M2"],
            "cycle": [3, 1, 2, 1, 2],  # désordonné exprès
            "usine": ["Usine A"] * 3 + ["Usine B"] * 2,
            "ligne_production": ["L1"] * 5,
            "T2": [12.0, 10.0, 11.0, 20.0, 21.0],
            "T24": [32.0, 30.0, 31.0, 40.0, 41.0],
        }
    )


def test_select_machines_limit_and_filter():
    df = _stream_df()
    assert select_machines(df, n=1, usine=None) == ["M1"]
    assert select_machines(df, n=5, usine="Usine B") == ["M2"]


def test_machine_frames_sorted_by_cycle():
    frames = machine_frames(_stream_df(), ["M1"])
    assert frames["M1"]["cycle"].tolist() == [1, 2, 3]  # trié


def test_history_upto_returns_last_seq_len():
    frame = machine_frames(_stream_df(), ["M1"])["M1"]
    # Au tick 2, on ne voit que les cycles 1 et 2 (fenêtre bornée à seq_len=2 ici).
    hist = history_upto(frame, tick=2, seq_len=2)
    assert hist["cycle"].tolist() == [1, 2]
    # seq_len limite la fenêtre aux plus récents.
    assert history_upto(frame, tick=3, seq_len=2)["cycle"].tolist() == [2, 3]


def test_build_request_features_only_ordered():
    frame = machine_frames(_stream_df(), ["M1"])["M1"]
    req = build_request("M1", history_upto(frame, tick=3, seq_len=3), ["T2", "T24"])
    assert req["machine_id"] == "M1"
    assert [c["values"]["T2"] for c in req["cycles"]] == [10.0, 11.0, 12.0]  # ordre cycle
    assert set(req["cycles"][0]["values"]) == {"T2", "T24"}  # pas de méta
