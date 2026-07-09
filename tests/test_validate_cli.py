"""Tests du CLI de la gate de validation (ml.validate_metrics.main).

La logique pure `check()` est couverte par test_mlops ; on exerce ici le point
d'entrée : lecture des fichiers, codes de sortie et affichage du rapport.
"""

from __future__ import annotations

import json

import pytest

from ml import validate_metrics as vm


@pytest.fixture(autouse=True)
def _sandbox_root(tmp_path, monkeypatch):
    """Les chemins CLI sont contraints à ROOT (anti-traversal) : on fait de
    tmp_path la racine de confiance pour pouvoir y écrire les fixtures."""
    monkeypatch.setattr(vm, "ROOT", tmp_path)


_THRESHOLDS = {
    "_comment": "ignoré",
    "classification": {"f1": {"min": 0.82}},
    "regression": {"RMSE": {"max": 30.0}},
}


def _write(tmp_path, metrics):
    m = tmp_path / "metrics.json"
    t = tmp_path / "thresholds.json"
    m.write_text(json.dumps(metrics))
    t.write_text(json.dumps(_THRESHOLDS))
    return m, t


def test_main_returns_2_when_metrics_missing(tmp_path, capsys):
    t = tmp_path / "thresholds.json"
    t.write_text(json.dumps(_THRESHOLDS))
    code = vm.main(["--metrics", str(tmp_path / "absent.json"), "--thresholds", str(t)])
    assert code == 2
    assert "introuvable" in capsys.readouterr().out


def test_main_returns_0_when_conform(tmp_path, capsys):
    m, t = _write(
        tmp_path,
        {"classification": {"f1": 0.9}, "regression": {"RMSE": 25.0}, "mode": "test"},
    )
    code = vm.main(["--metrics", str(m), "--thresholds", str(t)])
    assert code == 0
    assert "conforme" in capsys.readouterr().out


def test_main_returns_1_on_regression(tmp_path, capsys):
    m, t = _write(tmp_path, {"classification": {"f1": 0.5}, "regression": {"RMSE": 25.0}})
    code = vm.main(["--metrics", str(m), "--thresholds", str(t)])
    assert code == 1
    out = capsys.readouterr().out
    assert "régression" in out and "ÉCHEC" in out


def test_main_reports_missing_metric_as_failure(tmp_path):
    m, t = _write(tmp_path, {"classification": {}, "regression": {"RMSE": 25.0}})
    assert vm.main(["--metrics", str(m), "--thresholds", str(t)]) == 1


def test_main_rejects_path_outside_root(tmp_path):
    """Anti-traversal : un chemin qui s'échappe de ROOT est refusé avant lecture."""
    _, t = _write(tmp_path, {"classification": {"f1": 0.9}, "regression": {"RMSE": 25.0}})
    with pytest.raises(SystemExit, match="hors du projet"):
        vm.main(["--metrics", str(tmp_path / ".." / "evil.json"), "--thresholds", str(t)])
