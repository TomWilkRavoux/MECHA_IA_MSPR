"""Écran « Fiche machine » : détail d'une machine issue de l'analyse du parc."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import ApiClient

from ..utils.components import machine_cycles, render_prediction_metrics, render_trajectory_tabs


def render(client: ApiClient, res: pd.DataFrame, df: pd.DataFrame, thr: dict) -> None:
    """Fiche machine issue de l'analyse du parc (réutilise le résultat batch)."""
    st.subheader("Fiche machine")
    machine = st.selectbox("Machine", res["machine_id"].tolist(), key="detail_machine")
    render_prediction_metrics(res[res["machine_id"] == machine].iloc[0])
    render_trajectory_tabs(client, machine, machine_cycles(df, machine), thr,key_prefix="fiche")
