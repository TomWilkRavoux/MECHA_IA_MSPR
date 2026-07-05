"""Service d'inférence : chargement des modèles LSTM et prédiction.

Reproduit à l'identique le pré-traitement d'entraînement/évaluation
(`ml.prep.make_test_windows`) : normalisation StandardScaler, fenêtre des
`SEQ_LEN` derniers cycles, left-padding si la trajectoire est trop courte.
Deux têtes LSTM distinctes : classification (`at_risk`) et régression (`RUL`).
"""

from __future__ import annotations

import numpy as np
import torch

from ml.lstm import load_lstm
from ml.prep import FEATURES, RISK_THRESHOLD, SEQ_LEN, load_scaler

# Seuils métier pour la hiérarchisation des alertes (cf. CDC §5 : seuils/alertes).
CRITICAL_RUL = RISK_THRESHOLD // 2  # dégradation avancée -> intervention prioritaire
CRITICAL_PROBA = 0.75


def _alert_level(at_risk: bool, rul: float, proba: float) -> str:
    if not at_risk:
        return "ok"
    if rul <= CRITICAL_RUL or proba >= CRITICAL_PROBA:
        return "critical"
    return "warning"


class ModelService:
    """Charge scaler + 2 LSTM une seule fois et sert les prédictions."""

    def __init__(self) -> None:
        self.ready = False
        self.device = "cpu"
        self._scaler = None
        self._clf = None
        self._reg = None

    def load(self) -> None:
        """Charge les artefacts depuis models/. Lève si un artefact manque."""
        self._scaler = load_scaler()
        self._clf, self.device = load_lstm("lstm_classifier.pt")
        self._reg, _ = load_lstm("lstm_regressor.pt")
        self.ready = True

    def _build_window(self, cycles: list[dict[str, float]]) -> tuple[np.ndarray, int]:
        """(cycles bruts) -> fenêtre (SEQ_LEN, n_features) normalisée + n cycles réels."""
        arr = np.array(
            [[c[f] for f in FEATURES] for c in cycles], dtype="float32"
        )
        n_real = len(arr)
        arr = self._scaler.transform(arr)
        if n_real < SEQ_LEN:  # left-pad en répétant le premier cycle
            pad = np.repeat(arr[:1], SEQ_LEN - n_real, axis=0)
            arr = np.vstack([pad, arr])
        return arr[-SEQ_LEN:], min(n_real, SEQ_LEN)

    def predict(self, cycles: list[dict[str, float]]) -> dict:
        if not self.ready:
            raise RuntimeError("Modèles non chargés.")
        window, n_used = self._build_window(cycles)
        xt = torch.as_tensor(window[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            proba = float(torch.sigmoid(self._clf(xt)).item())
            rul = float(self._reg(xt).item())
        # `at_risk` = décision de la TÊTE DE CLASSIFICATION (seuil 0.5), cohérent
        # avec le protocole d'évaluation (eval_lstm_classifier.py). Le RUL est une
        # information de priorisation complémentaire fournie par la tête régression.
        at_risk = proba >= 0.5
        return {
            "at_risk": at_risk,
            "risk_probability": round(proba, 4),
            "rul_predicted": round(rul, 2),
            "alert_level": _alert_level(at_risk, rul, proba),
            "threshold": RISK_THRESHOLD,
            "n_cycles_used": n_used,
        }


service = ModelService()
