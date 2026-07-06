"""Composants d'affichage réutilisés par plusieurs écrans (fiche machine / analyse unitaire).

Extraction des cycles d'une machine, en-tête de stats, et les sous-onglets de
trajectoire (RUL / risque / capteurs).
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import requests
import streamlit as st
from api_client import ApiClient, build_single_request

from ..style.css import status_pill


def machine_cycles(df: pd.DataFrame, machine) -> pd.DataFrame:
    """Cycles d'une machine, ordonnés par `cycle` si la colonne existe."""
    g = df[df["machine_id"] == machine]
    return g.sort_values("cycle") if "cycle" in g.columns else g


def render_prediction_metrics(stats, extra: bool = False) -> None:
    """En-tête de stats d'une machine (accepte une Series `res` ou un dict `/predict`)."""
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="mecha-pill-label">Niveau d\'alerte</div>{status_pill(stats["alert_level"])}',
        unsafe_allow_html=True,
    )
    c2.metric("RUL estimé (dernier cycle)", f"{stats['rul_predicted']:.1f} cycles")
    c3.metric("Probabilité de risque", f"{stats['risk_probability']:.0%}")
    if extra:
        c4, c5 = st.columns(2)
        c4.metric("Cycles utilisés", stats.get("n_cycles_used", "-"))
        c5.metric("Seuil at_risk", f"{stats.get('threshold', '-')} cycles")


def _trajectory_chart(
    traj: pd.DataFrame, value: str, y_title: str, rules: list[tuple[float, str]]
):
    """Courbe `value` vs cycle + règles horizontales repères (seuil, valeur, libellé)."""
    line = (
        alt.Chart(traj)
        .mark_line(point=alt.OverlayMarkDef(size=18))
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y(f"{value}:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip(f"{value}:Q", title=y_title, format=".2f"),
            ],
        )
    )
    layers = [line]
    for y, label in rules:
        rule_df = pd.DataFrame({"y": [y], "label": [label]})
        layers.append(
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[5, 4], color="#888")
            .encode(y="y:Q")
        )
        layers.append(
            alt.Chart(rule_df)
            .mark_text(align="left", dx=4, dy=-4, color="#888")
            .encode(y="y:Q", text="label:N")
        )
    return alt.layer(*layers).properties(height=260)


def render_trajectory_tabs(
    client: ApiClient, machine, g: pd.DataFrame, thr: dict
) -> None:
    """Sous-onglets Régression / Classification / Capteurs pour une machine donnée."""
    # Trajectoire cycle par cycle via l'API (fenêtre glissante côté serveur)
    try:
        points = client.predict_trajectory(
            build_single_request(machine, g, client.features())
        )
    except requests.RequestException as exc:
        st.error(f"Trajectoire indisponible (API) : {exc}")
        return
    traj = pd.DataFrame(points)
    if "cycle" in g.columns and len(g) == len(traj):
        traj["cycle"] = g["cycle"].to_numpy()  # aligné : même ordre (tri par cycle)
    else:
        traj["cycle"] = traj["cycle_index"]

    tab_rul, tab_clf, tab_sensors = st.tabs(
        ["Régression (RUL)", "Classification (risque)", "Capteurs"]
    )

    with tab_rul:
        st.caption(
            f"RUL prédit à chaque cycle. La machine bascule **à risque** sous {thr['risk_threshold']} "
            f"cycles, et **critique** sous {thr['critical_rul']}."
        )
        st.altair_chart(
            _trajectory_chart(
                traj,
                "rul_predicted",
                "RUL estimé (cycles)",
                [
                    (
                        thr["risk_threshold"],
                        f"seuil à risque ({thr['risk_threshold']})",
                    ),
                    (thr["critical_rul"], f"critique ({thr['critical_rul']})"),
                ],
            ),
            use_container_width=True,
        )

    with tab_clf:
        st.caption(
            f"Probabilité `at_risk` à chaque cycle. Décision au-dessus de 0.5, "
            f"alerte **critique** au-dessus de {thr['critical_proba']}."
        )
        st.altair_chart(
            _trajectory_chart(
                traj,
                "risk_probability",
                "Probabilité de risque",
                [
                    (0.5, "décision (0.5)"),
                    (thr["critical_proba"], f"critique ({thr['critical_proba']})"),
                ],
            ),
            use_container_width=True,
        )

    with tab_sensors:
        # Petits multiples : évolution de quelques capteurs (un axe par capteur)
        sensors = [s for s in ["T24", "T50", "P30", "Nf"] if s in g.columns]
        if "cycle" in g.columns and sensors:
            long = g.melt(
                id_vars="cycle",
                value_vars=sensors,
                var_name="Capteur",
                value_name="Valeur",
            )
            chart = (
                alt.Chart(long)
                .mark_line()
                .encode(
                    x=alt.X("cycle:Q", title="Cycle"),
                    y=alt.Y("Valeur:Q", title=None, scale=alt.Scale(zero=False)),
                    facet=alt.Facet("Capteur:N", columns=2, title=None),
                )
                .resolve_scale(y="independent")
                .properties(width=300, height=160)
            )
            st.altair_chart(chart, use_container_width=False)
        else:
            st.info("Aucun capteur de démonstration disponible pour cette machine.")
