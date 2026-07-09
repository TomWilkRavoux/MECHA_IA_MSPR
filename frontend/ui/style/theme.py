"""Constantes d'affichage partagées entre les écrans du dashboard.

Palette de statut alignée sur une palette validée (contraste + CVD) : la couleur
porte l'état, mais elle est **toujours** accompagnée d'un libellé texte
(`STATUS_LABEL`) - jamais la couleur seule.
"""

from __future__ import annotations

# --- Palette de statut (couleur + libellé : jamais la couleur seule) ---
ALERT_ORDER = ["critical", "warning", "ok"]
STATUS_COLOR = {"critical": "#d03b3b", "warning": "#eaa11e", "ok": "#0ca30c"}
STATUS_LABEL = {"critical": "Critique", "warning": "Surveiller", "ok": "Normal"}
# Teintes douces pour les pastilles (fond clair + encre foncée lisible).
STATUS_PILL = {
    "critical": ("#fbe4e4", "#a92a2a"),
    "warning": ("#fbeecd", "#7f5500"),
    "ok": ("#dff3e0", "#0a7a2f"),
}

# Repères d'alerte par défaut (miroir de backend/api/inference.py) si /health injoignable.
DEFAULT_THRESHOLDS = {"risk_threshold": 30, "critical_rul": 15, "critical_proba": 0.75}
