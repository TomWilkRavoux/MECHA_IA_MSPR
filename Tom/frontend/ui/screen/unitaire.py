"""Écran « Analyse unitaire » : prédiction à la demande d'une machine via `/predict`."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from api_client import ApiClient, build_single_request

from ..utils.components import (
    machine_cycles,
    render_prediction_metrics,
    render_trajectory_tabs,
)


def render(client: ApiClient, df: pd.DataFrame, thr: dict) -> None:
    """Analyse d'UNE machine à la demande (endpoint `/predict`), sans analyser le parc."""
    st.subheader("Analyse d'une machine (prédiction à la demande)")
    st.caption(
        "Choisissez une machine et interrogez le modèle via `/predict` - pas besoin d'analyser tout le parc."
    )

    machines = df["machine_id"].drop_duplicates().tolist()
    machine = st.selectbox("Machine à analyser", machines, key="single_machine")
    g = machine_cycles(df, machine)

    meta = [str(g[c].iloc[0]) for c in ["usine", "ligne_production"] if c in g.columns]
    if meta:
        st.caption(
            "Localisation : " + " · ".join(meta) + f" · {len(g)} cycles observés"
        )

    try:
        stats = client.predict(build_single_request(machine, g, client.features()))
    except requests.RequestException as exc:
        st.error(f"Prédiction indisponible (API) : {exc}")
        return

    render_prediction_metrics(stats, extra=True)
    st.divider()
    render_trajectory_tabs(client, machine, g, thr)
