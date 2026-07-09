"""Tests unitaires des métriques et graphes d'évaluation (ml.metrics).

Fonctions pures et générateurs de figures matplotlib : aucune donnée réelle,
aucun modèle. Backend matplotlib forcé en `Agg` (sans affichage) pour la CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from ml import metrics


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def test_nasa_score_is_zero_on_perfect_prediction():
    y = np.array([10.0, 50.0, 100.0])
    assert metrics.nasa_score(y, y) == 0.0


def test_nasa_score_penalizes_late_more_than_early():
    y_true = np.array([50.0])
    # Convention NASA (d = pred - true) : RUL surestimé (d>0, panne détectée trop
    # tard) pénalisé plus fort que RUL sous-estimé (d<0, prédiction conservatrice).
    late = metrics.nasa_score(y_true, np.array([60.0]))  # d = +10 (surestimé)
    early = metrics.nasa_score(y_true, np.array([40.0]))  # d = -10 (sous-estimé)
    assert late > early > 0


def test_classification_metrics_perfect():
    y_true = [0, 1, 0, 1]
    m = metrics.classification_metrics(y_true, y_true)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert "roc_auc" not in m  # pas de proba fournie


def test_classification_metrics_with_proba_adds_roc_auc():
    m = metrics.classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], y_proba=[0.1, 0.9, 0.2, 0.8])
    assert m["roc_auc"] == 1.0


def test_regression_metrics_perfect():
    y = [10.0, 20.0, 30.0]
    m = metrics.regression_metrics(y, y)
    assert m["RMSE"] == 0.0
    assert m["MAE"] == 0.0
    assert m["R2"] == 1.0
    assert m["NASA"] == 0.0


def test_format_metrics_table_markdown():
    rows = {"LSTM": {"f1": 0.9, "RMSE": 25.0}, "RF": {"f1": 0.8, "RMSE": 30.0}}
    table = metrics.format_metrics_table(rows)
    lines = table.splitlines()
    assert lines[0] == "| Modèle | RMSE | f1 |"  # colonnes triées
    assert lines[1].startswith("|---")
    assert "| LSTM |" in table and "| RF |" in table
    assert "0.900" in table  # formatage à 3 décimales


def test_format_metrics_table_handles_missing_metric():
    table = metrics.format_metrics_table({"A": {"f1": 0.5}, "B": {"RMSE": 10.0}})
    assert "nan" in table  # métrique absente -> NaN formaté


# ---------------------------------------------------------------------------
# Graphes : on vérifie qu'une Figure est retournée (contenu délégué à matplotlib)
# ---------------------------------------------------------------------------
def _binary_sample():
    y_true = [0, 1, 0, 1, 1]
    y_proba = [0.2, 0.8, 0.3, 0.7, 0.9]
    y_pred = [0, 1, 0, 1, 1]
    return y_true, y_pred, y_proba


def test_plot_confusion_returns_figure():
    y_true, y_pred, _ = _binary_sample()
    assert isinstance(metrics.plot_confusion(y_true, y_pred), Figure)


def test_plot_roc_returns_figure():
    y_true, _, y_proba = _binary_sample()
    assert isinstance(metrics.plot_roc(y_true, y_proba), Figure)


def test_plot_pr_returns_figure():
    y_true, _, y_proba = _binary_sample()
    assert isinstance(metrics.plot_pr(y_true, y_proba), Figure)


def test_plot_feature_importance_returns_figure():
    names = ["a", "b", "c"]
    importances = [0.5, 0.3, 0.2]
    assert isinstance(metrics.plot_feature_importance(names, importances, top=2), Figure)


def test_plot_rul_scatter_returns_figure():
    assert isinstance(metrics.plot_rul_scatter([10, 20, 30], [12, 18, 33]), Figure)


def test_plot_error_hist_returns_figure():
    assert isinstance(metrics.plot_error_hist([10, 20, 30], [12, 18, 33]), Figure)


def test_plot_history_accepts_dict_and_object():
    fig_dict = metrics.plot_history({"loss": [1.0, 0.5], "val_loss": [1.1, 0.6]})
    assert isinstance(fig_dict, Figure)

    class _Hist:
        history = {"loss": [1.0, 0.4]}  # sans val_loss -> branche alternative

    assert isinstance(metrics.plot_history(_Hist()), Figure)


def test_save_fig_writes_png(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "FIG_DIR", tmp_path)
    fig = metrics.plot_rul_scatter([1, 2], [1, 2])
    metrics.save_fig(fig, "out.png")
    saved = tmp_path / "out.png"
    assert saved.exists() and saved.stat().st_size > 0
