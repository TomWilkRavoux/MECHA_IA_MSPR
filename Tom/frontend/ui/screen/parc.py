"""Écran « Vue parc » : KPI, priorisation des interventions, tableau du parc."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ..style.theme import ALERT_ORDER, STATUS_COLOR, STATUS_LABEL, STATUS_PILL

# Libellé de statut -> couleur d'encre (pour colorer la colonne « Alerte »).
_LABEL_FG = {STATUS_LABEL[k]: STATUS_PILL[k][1] for k in STATUS_LABEL}

# Libellé de la colonne « proba de risque » (réutilisé tooltip / tableau).
_PROBA_LABEL = "Proba risque"


def _render_kpis(res: pd.DataFrame) -> None:
    counts = res["alert_level"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Machines analysées", len(res))
    c2.metric(":red[Critiques]", int(counts.get("critical", 0)))
    c3.metric(":orange[À surveiller]", int(counts.get("warning", 0)))
    c4.metric(":green[Normales]", int(counts.get("ok", 0)))
    c5.metric("RUL min (cycles)", f"{res['rul_predicted'].min():.0f}")


def _render_urgency_chart(res: pd.DataFrame) -> None:
    st.subheader("Priorisation des interventions")
    st.caption("Machines classées par RUL croissant (la plus urgente en haut).")
    top = res.head(20).copy()
    top["Machine"] = top["machine_id"].astype(str)
    base = alt.Chart(top)
    bars = base.mark_bar(cornerRadius=4, height=alt.RelativeBandSize(0.7)).encode(
        y=alt.Y("Machine:N", sort=top["Machine"].tolist(), title=None),
        x=alt.X("rul_predicted:Q", title="RUL estimé (cycles)"),
        color=alt.Color(
            "alert_level:N",
            scale=alt.Scale(domain=ALERT_ORDER, range=[STATUS_COLOR[a] for a in ALERT_ORDER]),
            legend=alt.Legend(title="Niveau d'alerte", orient="top"),
        ),
        tooltip=[
            alt.Tooltip("machine_id:N", title="Machine"),
            alt.Tooltip("rul_predicted:Q", title="RUL", format=".1f"),
            alt.Tooltip("risk_probability:Q", title=_PROBA_LABEL, format=".2f"),
            alt.Tooltip("badge:N", title="Alerte"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, color="#555").encode(
        y=alt.Y("Machine:N", sort=top["Machine"].tolist()),
        x="rul_predicted:Q",
        text=alt.Text("rul_predicted:Q", format=".0f"),
    )
    st.altair_chart(
        (bars + labels).properties(height=max(240, 26 * len(top))), use_container_width=True
    )


def _render_table(res: pd.DataFrame) -> None:
    st.subheader("État du parc")
    cols = ["machine_id"] + [c for c in ["usine", "ligne_production"] if c in res.columns]
    cols += ["badge", "rul_predicted", "risk_probability"]
    view = res[cols].rename(
        columns={
            "machine_id": "Machine",
            "usine": "Usine",
            "ligne_production": "Ligne",
            "badge": "Alerte",
            "rul_predicted": "RUL (cycles)",
            "risk_probability": _PROBA_LABEL,
        }
    )
    styled = view.style.map(
        lambda v: f"color: {_LABEL_FG[v]}; font-weight: 600" if v in _LABEL_FG else "",
        subset=["Alerte"],
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RUL (cycles)": st.column_config.NumberColumn(format="%.1f"),
            _PROBA_LABEL: st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0
            ),
        },
    )
    st.download_button(
        "Exporter les alertes (CSV)",
        res.to_csv(index=False).encode(),
        file_name="alertes_mecha.csv",
        mime="text/csv",
    )


def render(res: pd.DataFrame) -> None:
    """Vue d'ensemble du parc analysé."""
    _render_kpis(res)
    st.divider()
    _render_urgency_chart(res)
    st.divider()
    _render_table(res)
