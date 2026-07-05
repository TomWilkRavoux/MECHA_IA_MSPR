"""Couche CSS de finition : rendu clair, sobre et « app métier » (cartes, pastilles).

`apply()` est appelé une fois au démarrage par `dashboard.py`. Le thème de base
(couleurs, typo) vit dans `.streamlit/config.toml` ; ici on ne fait que la finition
que le thème natif ne couvre pas.
"""

from __future__ import annotations

import streamlit as st

from .theme import STATUS_LABEL, STATUS_PILL

_CSS = """
<style>
/* --- Épure : on retire le bruit visuel de Streamlit --- */
[data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* --- Respiration --- */
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1320px;}

/* --- Titres --- */
h1 {font-weight: 600; letter-spacing: -0.01em;}
h2, h3 {font-weight: 600;}

/* --- Métriques présentées comme des cartes --- */
[data-testid="stMetric"] {
    background: #fcfcfb;
    border: 1px solid rgba(18, 16, 12, 0.08);
    border-radius: 10px;
    padding: 14px 16px 12px;
    box-shadow: 0 1px 2px rgba(18, 16, 12, 0.03);
}
[data-testid="stMetricValue"] {font-weight: 600;}

/* --- Onglets sobres, soulignement d'accent --- */
[data-baseweb="tab-list"] {gap: 2px; border-bottom: 1px solid #e4e2da;}
[data-baseweb="tab"] {font-weight: 500; padding-left: 14px; padding-right: 14px;}

/* --- Boutons & tableaux --- */
.stButton > button {border-radius: 8px; font-weight: 600;}
[data-testid="stDataFrame"] {border: 1px solid rgba(18, 16, 12, 0.08); border-radius: 10px;}

/* --- Pastilles de statut (couleur + libellé) --- */
.mecha-pill-label {font-size: 0.8rem; color: #52514e; margin-bottom: 5px;}
.mecha-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.4;
}
</style>
"""

_PILL_CSS = "".join(
    f".mecha-pill--{lvl} {{background: {bg}; color: {fg};}}"
    for lvl, (bg, fg) in STATUS_PILL.items()
)


def apply() -> None:
    """Injecte la feuille de style (une fois par run)."""
    st.markdown(_CSS.replace("</style>", _PILL_CSS + "</style>"), unsafe_allow_html=True)


def status_pill(level: str) -> str:
    """HTML d'une pastille de statut (couleur douce + libellé texte)."""
    return f'<span class="mecha-pill mecha-pill--{level}">{STATUS_LABEL[level]}</span>'
