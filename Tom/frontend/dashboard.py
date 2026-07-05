"""Dashboard de maintenance prédictive MECHA (Streamlit) - point d'entrée.

Interface métier d'exploitation du modèle LSTM (CDC §5) : supervision d'un parc
de machines, priorisation des interventions par RUL et niveau d'alerte. Le
dashboard consomme l'API REST du backend - il ne charge aucun modèle localement.

Ce fichier ne fait qu'**orchestrer les écrans**. L'interface est organisée en
package `ui/` : `ui/screen/` (barre latérale + un module par onglet),
`ui/style/` (thème + CSS), `ui/utils/` (appels API + composants partagés).

Lancement (depuis Tom/, backend démarré) :
    uv run streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import requests
import streamlit as st
from ui.screen import ligne, machine, parc, sidebar, unitaire
from ui.style import css
from ui.utils.api import get_thresholds, run_predictions

st.set_page_config(
    page_title="MECHA - Maintenance prédictive", page_icon="🔧", layout="wide"
)

PARK_HINT = "Cliquez sur **Analyser le parc** pour lancer les prédictions sur l'ensemble sélectionné."


def main() -> None:
    css.apply()
    client, df, n_max = sidebar.render()
    st.title("Maintenance prédictive / supervision du parc")
    st.caption(
        "Prédiction de l'état `at_risk` et du RUL (LSTM) pour prioriser la maintenance."
    )

    if df is None:
        st.info(
            "Sélectionnez une source de données dans la barre latérale pour démarrer."
        )
        return

    thr = get_thresholds(client)

    if st.button("Analyser le parc", type="primary"):
        with st.spinner("Interrogation du modèle via l'API…"):
            try:
                st.session_state["res"] = run_predictions(client, df, n_max)
                st.session_state["df"] = df
            except requests.RequestException as exc:
                st.error(f"Échec de l'appel à l'API : {exc}")

    tab_parc, tab_ligne, tab_machine, tab_unit = st.tabs(
        ["Vue parc", "Par ligne", "Fiche machine", "Analyse unitaire"]
    )
    res = st.session_state.get("res")

    with tab_parc:
        if res is None:
            st.info(PARK_HINT)
        else:
            parc.render(res)
    with tab_ligne:
        if res is None:
            st.info(PARK_HINT)
        else:
            ligne.render(res)
    with tab_machine:
        if res is None:
            st.info(PARK_HINT)
        else:
            machine.render(client, res, st.session_state["df"], thr)
    with tab_unit:
        unitaire.render(client, df, thr)


main()
