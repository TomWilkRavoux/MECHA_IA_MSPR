"""Tests du registre de modèles versionné (ml.registry).

Les chemins du module (MODELS_DIR / RUNS_DIR / REGISTRY_PATH) sont redirigés vers
un `tmp_path` isolé : aucun artefact réel n'est touché, aucun réseau, pas de torch.
"""

from __future__ import annotations

import re

import pytest

from ml import registry


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirige tous les chemins du registre vers un dossier temporaire."""
    models = tmp_path / "models"
    runs = models / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(registry, "MODELS_DIR", models)
    monkeypatch.setattr(registry, "RUNS_DIR", runs)
    monkeypatch.setattr(registry, "REGISTRY_PATH", models / "registry.json")
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    return models, runs


def test_new_run_id_is_sortable_utc():
    rid = registry.new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z", rid)


def test_load_registry_empty_when_absent(sandbox):
    assert registry.load_registry() == {"current": None, "runs": []}


def test_register_run_persists_and_promotes(sandbox):
    reg = registry.register_run("R1", metrics={"classification": {"f1": 0.9}}, promote=True)
    assert reg["current"] == "R1"
    # Relecture depuis le disque : le pointeur a bien été sérialisé.
    assert registry.current_run_id() == "R1"
    assert registry.load_registry()["runs"][0]["metrics"]["classification"]["f1"] == 0.9


def test_register_run_without_promote_keeps_baseline(sandbox):
    registry.register_run("R1", metrics={}, promote=True)
    registry.register_run("R2", metrics={}, promote=False)
    assert registry.current_run_id() == "R1"  # R1 reste servi


def test_set_current_rejects_unknown_run(sandbox):
    registry.register_run("R1", metrics={}, promote=True)
    with pytest.raises(ValueError, match="run inconnu"):
        registry.set_current("R404")


def test_set_current_none_returns_to_baseline(sandbox):
    registry.register_run("R1", metrics={}, promote=True)
    registry.set_current(None)
    assert registry.current_run_id() is None


def test_resolve_prefers_current_run_when_present(sandbox):
    models, runs = sandbox
    registry.register_run("R1", metrics={}, promote=True)
    run_file = runs / "R1" / "scaler.joblib"
    run_file.parent.mkdir(parents=True)
    run_file.write_text("x")
    assert registry.resolve("scaler.joblib") == run_file


def test_resolve_falls_back_to_baseline_when_run_file_absent(sandbox):
    models, _ = sandbox
    registry.register_run("R1", metrics={}, promote=True)  # run courant sans artefact
    assert registry.resolve("scaler.joblib") == models / "scaler.joblib"


def test_resolve_uses_baseline_without_registry(sandbox):
    models, _ = sandbox
    assert registry.resolve("metrics.json") == models / "metrics.json"


def test_promote_to_baseline_copies_existing_artifacts(sandbox):
    models, runs = sandbox
    src = runs / "R1"
    src.mkdir()
    (src / "lstm_classifier.pt").write_text("clf")
    (src / "metrics.json").write_text("{}")  # scaler/regressor absents -> ignorés
    copied = registry.promote_to_baseline("R1")
    names = {p.name for p in copied}
    assert names == {"lstm_classifier.pt", "metrics.json"}
    assert (models / "lstm_classifier.pt").read_text() == "clf"


def test_promote_to_baseline_unknown_run_raises(sandbox):
    with pytest.raises(ValueError, match="run introuvable"):
        registry.promote_to_baseline("R404")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_list_empty(sandbox, capsys):
    assert registry.main(["list"]) == 0
    assert "aucun run enregistré" in capsys.readouterr().out


def test_cli_list_with_runs(sandbox, capsys):
    registry.register_run("R1", metrics={"classification": {"f1": 0.9}}, promote=True)
    assert registry.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "R1" in out and "→" in out


def test_cli_promote_and_use_baseline(sandbox, capsys):
    registry.register_run("R1", metrics={}, promote=True)
    registry.register_run("R2", metrics={}, promote=False)
    assert registry.main(["promote", "R2"]) == 0
    assert registry.current_run_id() == "R2"
    assert registry.main(["use-baseline"]) == 0
    assert registry.current_run_id() is None


def test_cli_promote_to_baseline(sandbox, capsys):
    _, runs = sandbox
    src = runs / "R1"
    src.mkdir()
    (src / "scaler.joblib").write_text("s")
    registry.register_run("R1", metrics={}, promote=True)
    assert registry.main(["promote", "R1", "--to-baseline"]) == 0
    assert "baseline" in capsys.readouterr().out.lower()
