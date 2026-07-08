"""Écran « Par ligne » : synthèse agrégée par ligne de production (et usine)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ..style.theme import ALERT_ORDER, STATUS_COLOR


def render(res: pd.DataFrame) -> None:
    """Vue agrégée par ligne de production (et usine) : où concentrer la maintenance."""
    st.subheader("Synthèse par ligne de production")
    group_cols = [c for c in ["usine", "ligne_production"] if c in res.columns]
    if not group_cols:
        st.info("Le jeu de données ne contient pas de colonne `ligne_production` / `usine`.")
        return

    # Comptage des niveaux d'alerte + RUL moyen/min par ligne
    counts = (
        res.groupby(group_cols)["alert_level"].value_counts().unstack(fill_value=0).reset_index()
    )
    for lvl in ALERT_ORDER:
        if lvl not in counts.columns:
            counts[lvl] = 0
    rul = res.groupby(group_cols)["rul_predicted"].agg(["mean", "min"]).reset_index()
    agg = counts.merge(rul, on=group_cols)
    agg["Ligne"] = agg[group_cols].astype(str).agg(" · ".join, axis=1)
    agg = agg.sort_values("critical", ascending=False).reset_index(drop=True)

    # Bar chart empilé : répartition des alertes par ligne
    long = agg.melt(id_vars="Ligne", value_vars=ALERT_ORDER, var_name="alert_level", value_name="n")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("Ligne:N", sort=agg["Ligne"].tolist(), title=None),
            x=alt.X("n:Q", title="Nombre de machines", stack="zero"),
            color=alt.Color(
                "alert_level:N",
                scale=alt.Scale(domain=ALERT_ORDER, range=[STATUS_COLOR[a] for a in ALERT_ORDER]),
                legend=alt.Legend(title="Niveau d'alerte", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Ligne:N"),
                alt.Tooltip("alert_level:N", title="Niveau"),
                alt.Tooltip("n:Q", title="Machines"),
            ],
        )
        .properties(height=max(200, 34 * len(agg)))
    )
    st.altair_chart(chart, use_container_width=True)

    view = agg.rename(
        columns={
            "critical": "Critiques",
            "warning": "À surveiller",
            "ok": "Normales",
            "mean": "RUL moyen",
            "min": "RUL min",
        }
    )
    cols = ["Ligne", "Critiques", "À surveiller", "Normales", "RUL moyen", "RUL min"]
    st.dataframe(
        view[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "RUL moyen": st.column_config.NumberColumn(format="%.0f"),
            "RUL min": st.column_config.NumberColumn(format="%.0f"),
        },
    )
