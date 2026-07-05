"""Métriques et graphes partagés pour l'évaluation des modèles IA MECHA.

Deux tâches :
- Classification `at_risk` : accuracy, precision, recall, F1, ROC-AUC + graphes
  (matrice de confusion, ROC, précision/rappel).
- Régression `RUL` : RMSE, MAE, R² + **score asymétrique NASA C-MAPSS** qui
  pénalise davantage les prédictions en retard (RUL surestimé = panne détectée
  trop tard, coûteux pour MECHA).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.prep import FIG_DIR


# ----------------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------------
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score asymétrique NASA C-MAPSS (plus bas = mieux).

    Retard de prédiction (d<0, RUL surestimé) pénalisé plus fortement (exp(-d/13))
    que l'avance (d>0, RUL sous-estimé, exp(d/10)).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return m


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA": nasa_score(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Graphes (retournent une Figure matplotlib ; sauvegardables via save_fig)
# ----------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    """Sauvegarde une figure dans reports/figures/ (PNG haute résolution)."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")


def plot_confusion(y_true, y_pred, labels=("Normal", "À risque"), title="Matrice de confusion"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(y_true, y_proba, title="Courbe ROC"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr(y_true, y_proba, title="Courbe précision / rappel"):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(names, importances, top=20, title="Importance des variables"):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(order))))
    ax.barh([names[i] for i in order][::-1], np.asarray(importances)[order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_rul_scatter(y_true, y_pred, title="RUL prédit vs réel"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5)
    lim = [0, max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "--", color="red", linewidth=1, label="prédiction idéale")
    ax.set_xlabel("RUL réel (cycles)")
    ax.set_ylabel("RUL prédit (cycles)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_hist(y_true, y_pred, title="Distribution de l'erreur de RUL"):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(err, bins=40)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Erreur (prédit − réel) — négatif = retard (risqué)")
    ax.set_ylabel("Effectif")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_history(history, title="Historique d'entraînement (LSTM)"):
    """Courbes loss / val_loss d'un entraînement Keras (objet `history.history`)."""
    hist = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["loss"], label="entraînement")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def format_metrics_table(rows: dict[str, dict]) -> str:
    """Formate un dict {nom_modèle: {métrique: valeur}} en tableau markdown."""
    cols = sorted({k for m in rows.values() for k in m})
    header = "| Modèle | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for name, m in rows.items():
        vals = " | ".join(f"{m.get(c, float('nan')):.3f}" for c in cols)
        lines.append(f"| {name} | {vals} |")
    return "\n".join(lines)
