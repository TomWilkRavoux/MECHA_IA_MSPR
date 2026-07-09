"""Architecture LSTM partagée (source unique pour l'entraînement et l'évaluation).

Utilisée par `notebooks/03_lstm.py` (entraînement) et par les notebooks
`notebooks/analyse/eval_lstm_*.py` (évaluation), pour éviter toute divergence
de l'architecture entre entraînement et rechargement.
"""

from __future__ import annotations

import torch
from torch import nn

from ml.prep import FEATURES


class LSTMNet(nn.Module):
    """LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1).

    Sortie : un scalaire par fenêtre (logit pour la classification, RUL pour la
    régression). L'activation sigmoïde est appliquée à l'évaluation côté classif.
    """

    def __init__(self, n_features: int = len(FEATURES)):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])  # dernier pas de temps
        return self.head(out).squeeze(1)


def load_lstm(filename: str, device: str | None = None) -> tuple[LSTMNet, str]:
    """Recharge un LSTM entraîné depuis models/ et le met en mode évaluation."""
    from ml import registry  # résout le run courant (sinon baseline plate)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet().to(device)
    # weights_only=True : ne désérialise QUE des tenseurs (pas de pickle arbitraire),
    # protège contre l'exécution de code via un checkpoint malveillant (CWE-502).
    model.load_state_dict(
        torch.load(registry.resolve(filename), map_location=device, weights_only=True)
    )
    model.eval()
    return model, device
