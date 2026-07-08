"""Écran : barre latérale (connexion API, source de données, filtres)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from api_client import ApiClient

ROOT = Path(__file__).resolve().parents[3]  # screen/ -> ui/ -> frontend/ -> Tom/
DEMO_CSV = ROOT / "assets" / "KaggleDataset" / "mecha_test_classification.csv"
DEFAULT_API = os.environ.get("MECHA_API_URL", "http://localhost:8000")


def render() -> tuple[ApiClient, pd.DataFrame | None, int]:
    st.sidebar.title("MECHA")
    st.sidebar.caption("Maintenance prédictive / supervision du parc")

    api_url = st.sidebar.text_input("URL de l'API", value=DEFAULT_API)
    client = ApiClient(api_url)

    # État de santé du backend
    try:
        h = client.health()
        if h.get("models_loaded"):
            st.sidebar.success(f"API OK · device `{h['device']}` · seuil {h['risk_threshold']}")
        else:
            st.sidebar.warning("API joignable mais modèles non chargés.")
    except requests.RequestException:
        st.sidebar.error("API injoignable. Démarrez le backend puis rechargez.")

    st.sidebar.divider()
    st.sidebar.subheader("Source de données")
    src = st.sidebar.radio("Cycles machines", ["Jeu de démonstration", "Importer un CSV"], index=0)

    df: pd.DataFrame | None = None
    if src == "Jeu de démonstration":
        if DEMO_CSV.exists():
            df = pd.read_csv(DEMO_CSV)
            st.sidebar.caption(f"{DEMO_CSV.name} · {df['machine_id'].nunique()} machines")
        else:
            st.sidebar.error(f"Fichier de démo introuvable : {DEMO_CSV}")
    else:
        up = st.sidebar.file_uploader("CSV (machine_id, cycle, + variables capteurs)", type="csv")
        if up is not None:
            df = pd.read_csv(up)

    n_max = 40
    if df is not None:
        # Filtres métier (si les colonnes d'organisation sont présentes)
        for col, label in [
            ("usine", "Usine"),
            ("ligne_production", "Ligne de production"),
        ]:
            if col in df.columns:
                opts = sorted(df[col].dropna().unique().tolist())
                sel = st.sidebar.multiselect(label, opts, default=opts)
                df = df[df[col].isin(sel)]
        total = df["machine_id"].nunique()
        n_max = st.sidebar.slider("Machines à analyser", 1, int(total), min(40, int(total)))

    return client, df, n_max
