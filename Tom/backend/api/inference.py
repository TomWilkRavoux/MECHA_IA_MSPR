"""Service d'inférence : chargement des modèles LSTM et prédiction.

Reproduit à l'identique le pré-traitement d'entraînement/évaluation
(`ml.prep.make_test_windows`) : normalisation StandardScaler, fenêtre des
`SEQ_LEN` derniers cycles, left-padding si la trajectoire est trop courte.
Deux têtes LSTM distinctes : classification (`at_risk`) et régression (`RUL`).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ml.lstm import load_lstm
from ml.prep import FEATURES, RISK_THRESHOLD, SEQ_LEN, load_scaler

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "3")))

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

    def _normalize(self, cycles: list[dict[str, float]]) -> np.ndarray:
        """(cycles bruts) -> matrice (n_cycles, n_features) normalisée (StandardScaler)."""
        arr = np.array([[c[f] for f in FEATURES] for c in cycles], dtype="float32")
        return self._scaler.transform(arr)

    @staticmethod
    def _window_ending_at(arr: np.ndarray, end: int) -> tuple[np.ndarray, int]:
        """Fenêtre des SEQ_LEN cycles se terminant à `end` (exclu), left-paddée.

        `arr` est déjà normalisée. Si moins de SEQ_LEN cycles sont disponibles, on
        répète le premier cycle réel de la fenêtre (cohérent avec l'entraînement).
        """
        w = arr[max(0, end - SEQ_LEN):end]
        n_real = len(w)
        if n_real < SEQ_LEN:
            pad = np.repeat(w[:1], SEQ_LEN - n_real, axis=0)
            w = np.vstack([pad, w])
        return w, min(n_real, SEQ_LEN)

    def _infer(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(batch de fenêtres (n, SEQ_LEN, n_features)) -> (probas, ruls) numpy."""
        xt = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            proba = torch.sigmoid(self._clf(xt)).cpu().numpy()
            rul = self._reg(xt).cpu().numpy()
        return proba.reshape(-1), rul.reshape(-1)

    @staticmethod
    def _format(at_risk: bool, proba: float, rul: float) -> dict:
        """Construit le dict de sortie commun (prédiction + niveau d'alerte)."""
        return {
            "at_risk": at_risk,
            "risk_probability": round(proba, 4),
            "rul_predicted": round(rul, 2),
            "alert_level": _alert_level(at_risk, rul, proba),
        }

    def predict(self, cycles: list[dict[str, float]]) -> dict:
        if not self.ready:
            raise RuntimeError("Modèles non chargés.")
        arr = self._normalize(cycles)
        window, n_used = self._window_ending_at(arr, len(arr))
        proba, rul = self._infer(window[None, ...])
        # `at_risk` = décision de la TÊTE DE CLASSIFICATION (seuil 0.5), cohérent
        # avec le protocole d'évaluation (eval_lstm_classifier.py). Le RUL est une
        # information de priorisation complémentaire fournie par la tête régression.
        at_risk = bool(proba[0] >= 0.5)
        return {
            **self._format(at_risk, float(proba[0]), float(rul[0])),
            "threshold": RISK_THRESHOLD,
            "n_cycles_used": n_used,
        }

    def predict_many(self, machines_cycles: list[list[dict[str, float]]]) -> list[dict]:
        """Prédiction d'un lot de machines en **un seul passage batch**.

        Empile la fenêtre glissante de chaque machine et n'exécute qu'un forward
        pour tout le parc (au lieu de N forwards de batch 1). Résultat identique à
        N appels `predict`, ordre d'entrée préservé.
        """
        if not self.ready:
            raise RuntimeError("Modèles non chargés.")
        windows, n_used = [], []
        for cycles in machines_cycles:
            arr = self._normalize(cycles)
            w, n = self._window_ending_at(arr, len(arr))
            windows.append(w)
            n_used.append(n)
        # np.stack est sûr : _window_ending_at renvoie toujours (SEQ_LEN, n_features)
        # grâce au left-padding, donc toutes les fenêtres ont la même forme.
        probas, ruls = self._infer(np.stack(windows))
        return [
            {
                **self._format(bool(p >= 0.5), float(p), float(r)),
                "threshold": RISK_THRESHOLD,
                "n_cycles_used": n,
            }
            for p, r, n in zip(probas, ruls, n_used)
        ]

    def predict_trajectory(self, cycles: list[dict[str, float]]) -> list[dict]:
        """Prédiction cycle par cycle : une fenêtre glissante par cycle observé.

        Pour chaque cycle `i` (1..n), on prédit sur la fenêtre des SEQ_LEN cycles
        s'y terminant. Toutes les fenêtres sont évaluées en **un seul passage batch**.
        Renvoie un point par cycle (aligné sur l'ordre d'entrée via `cycle_index`).
        """
        if not self.ready:
            raise RuntimeError("Modèles non chargés.")
        arr = self._normalize(cycles)
        windows = np.stack([self._window_ending_at(arr, end)[0] for end in range(1, len(arr) + 1)])
        probas, ruls = self._infer(windows)
        return [
            {"cycle_index": i, **self._format(bool(p >= 0.5), float(p), float(r))}
            for i, (p, r) in enumerate(zip(probas, ruls))
        ]


service = ModelService()
