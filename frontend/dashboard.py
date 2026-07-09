"""Dashboard de maintenance prédictive MECHA (Streamlit) - point d'entrée.

Interface métier d'exploitation du modèle LSTM (CDC §5) : supervision d'un parc
de machines, priorisation des interventions par RUL et niveau d'alerte. Le
dashboard consomme l'API REST du backend - il ne charge aucun modèle localement.

Ce fichier ne fait qu'**orchestrer les écrans**. L'interface est organisée en
package `ui/` : `ui/screen/` (barre latérale + un module par onglet),
`ui/style/` (thème + CSS), `ui/utils/` (appels API + composants partagés).

Lancement (depuis la racine, backend démarré) :
    uv run streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import requests
import streamlit as st
from ui.screen import ligne, machine, parc, sidebar, unitaire
from ui.style import css
from ui.utils.api import get_thresholds, run_predictions

st.set_page_config(page_title="MECHA - Maintenance prédictive", page_icon="🔧", layout="wide")

PARK_HINT = (
    "Cliquez sur **Analyser le parc** pour lancer les prédictions sur l'ensemble sélectionné."
)

# Écrans du dashboard. On navigue à écran unique (un seul rendu par rerun) plutôt
# qu'avec `st.tabs` : les onglets rendent TOUT leur contenu à chaque run et laissent
# le navigateur masquer/afficher en CSS, ce qui, sous latence (API lente en prod),
# fait « fuiter » le contenu d'un onglet dans l'onglet affiché (bleed de deltas).
SCREENS = ["Vue parc", "Par ligne", "Fiche machine", "Analyse unitaire"]


def main() -> None:
    css.apply()
    client, df, n_max = sidebar.render()
    st.title("Maintenance prédictive / supervision du parc")
    st.caption("Prédiction de l'état `at_risk` et du RUL (LSTM) pour prioriser la maintenance.")

    if df is None:
        st.info("Sélectionnez une source de données dans la barre latérale pour démarrer.")
        return

    thr = get_thresholds(client)

    if st.button("Analyser le parc", type="primary"):
        with st.spinner("Interrogation du modèle via l'API…"):
            try:
                st.session_state["res"] = run_predictions(client, df, n_max)
                st.session_state["df"] = df
            except requests.RequestException as exc:
                st.error(f"Échec de l'appel à l'API : {exc}")

    screen = (
        st.segmented_control(
            "Navigation", SCREENS, default=SCREENS[0], key="nav", label_visibility="collapsed"
        )
        or SCREENS[0]  # `None` quand l'utilisateur déselectionne : on retombe sur la vue parc
    )
    res = st.session_state.get("res")

    # Rendu à écran unique : seul l'écran sélectionné existe dans l'arbre à ce rerun.
    if screen == "Analyse unitaire":
        # Ne dépend pas de l'analyse du parc : prédiction à la demande via /predict.
        unitaire.render(client, df, thr)
    elif res is None:
        st.info(PARK_HINT)
    elif screen == "Vue parc":
        parc.render(res)
    elif screen == "Par ligne":
        ligne.render(res)
    elif screen == "Fiche machine":
        machine.render(client, res, st.session_state["df"], thr)


main()
