"""Dashboard de maintenance prédictive MECHA (Streamlit).

Interface métier d'exploitation du modèle LSTM (CDC §5) : supervision d'un parc
de machines, priorisation des interventions par RUL et niveau d'alerte. Le
dashboard consomme l'API REST du backend — il ne charge aucun modèle localement.

Lancement (depuis Tom/, backend démarré) :
    uv run streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

from api_client import META_COLS, ApiClient, build_requests

ROOT = Path(__file__).resolve().parent.parent  # dossier Tom/
DEMO_CSV = ROOT / "assets" / "KaggleDataset" / "mecha_test_classification.csv"
DEFAULT_API = os.environ.get("MECHA_API_URL", "http://localhost:8000")

# --- Palette de statut (icône + libellé + couleur : jamais la couleur seule) ---
ALERT_ORDER = ["critical", "warning", "ok"]
STATUS_COLOR = {"critical": "#C62828", "warning": "#ED9B00", "ok": "#2E7D32"}
STATUS_BADGE = {"critical": "🔴 Critique", "warning": "🟠 Surveiller", "ok": "🟢 Normal"}

st.set_page_config(page_title="MECHA — Maintenance prédictive", page_icon="🔧", layout="wide")


# ---------------------------------------------------------------------------
# Barre latérale : connexion API + source de données + filtres
# ---------------------------------------------------------------------------
def sidebar() -> tuple[ApiClient, pd.DataFrame | None, int]:
    st.sidebar.title("🔧 MECHA")
    st.sidebar.caption("Maintenance prédictive — supervision du parc")

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
        for col, label in [("usine", "Usine"), ("ligne_production", "Ligne de production")]:
            if col in df.columns:
                opts = sorted(df[col].dropna().unique().tolist())
                sel = st.sidebar.multiselect(label, opts, default=opts)
                df = df[df[col].isin(sel)]
        total = df["machine_id"].nunique()
        n_max = st.sidebar.slider("Machines à analyser", 1, int(total), min(40, int(total)))

    return client, df, n_max


# ---------------------------------------------------------------------------
# Appel API + mise en forme des résultats
# ---------------------------------------------------------------------------
def run_predictions(client: ApiClient, df: pd.DataFrame, n_max: int) -> pd.DataFrame:
    features = client.features()
    ids = df["machine_id"].drop_duplicates().head(n_max).tolist()
    sub = df[df["machine_id"].isin(ids)]

    results = client.predict_batch(build_requests(sub, features))
    res = pd.DataFrame(results)

    # Enrichit avec l'usine / la ligne (1re occurrence par machine)
    meta_cols = [c for c in ["usine", "ligne_production"] if c in df.columns]
    if meta_cols:
        meta = df.groupby("machine_id", sort=False)[meta_cols].first().reset_index()
        res = res.merge(meta, on="machine_id", how="left")

    res["badge"] = res["alert_level"].map(STATUS_BADGE)
    return res.sort_values("rul_predicted").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
def render_kpis(res: pd.DataFrame) -> None:
    counts = res["alert_level"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Machines analysées", len(res))
    c2.metric("🔴 Critiques", int(counts.get("critical", 0)))
    c3.metric("🟠 À surveiller", int(counts.get("warning", 0)))
    c4.metric("🟢 Normales", int(counts.get("ok", 0)))
    c5.metric("RUL min (cycles)", f"{res['rul_predicted'].min():.0f}")


def render_urgency_chart(res: pd.DataFrame) -> None:
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
            alt.Tooltip("risk_probability:Q", title="Proba risque", format=".2f"),
            alt.Tooltip("badge:N", title="Alerte"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, color="#555").encode(
        y=alt.Y("Machine:N", sort=top["Machine"].tolist()),
        x="rul_predicted:Q",
        text=alt.Text("rul_predicted:Q", format=".0f"),
    )
    st.altair_chart((bars + labels).properties(height=max(240, 26 * len(top))), use_container_width=True)


def render_table(res: pd.DataFrame) -> None:
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
            "risk_probability": "Proba risque",
        }
    )
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RUL (cycles)": st.column_config.NumberColumn(format="%.1f"),
            "Proba risque": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
        },
    )
    st.download_button(
        "⬇️ Exporter les alertes (CSV)",
        res.to_csv(index=False).encode(),
        file_name="alertes_mecha.csv",
        mime="text/csv",
    )


def render_detail(res: pd.DataFrame, df: pd.DataFrame) -> None:
    st.subheader("Fiche machine")
    machine = st.selectbox("Machine", res["machine_id"].tolist())
    row = res[res["machine_id"] == machine].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Niveau d'alerte", STATUS_BADGE[row["alert_level"]])
    c2.metric("RUL estimé", f"{row['rul_predicted']:.1f} cycles")
    c3.metric("Probabilité de risque", f"{row['risk_probability']:.0%}")

    # Petits multiples : évolution de quelques capteurs (un axe par capteur)
    g = df[df["machine_id"] == machine]
    sensors = [s for s in ["T24", "T50", "P30", "Nf"] if s in g.columns]
    if "cycle" in g.columns and sensors:
        long = g.melt(id_vars="cycle", value_vars=sensors, var_name="Capteur", value_name="Valeur")
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


def main() -> None:
    client, df, n_max = sidebar()
    st.title("Maintenance prédictive — supervision du parc")
    st.caption("Prédiction de l'état `at_risk` et du RUL (LSTM) pour prioriser la maintenance.")

    if df is None:
        st.info("Sélectionnez une source de données dans la barre latérale pour démarrer.")
        return

    if st.button("▶️ Analyser le parc", type="primary"):
        with st.spinner("Interrogation du modèle via l'API…"):
            try:
                st.session_state["res"] = run_predictions(client, df, n_max)
                st.session_state["df"] = df
            except requests.RequestException as exc:
                st.error(f"Échec de l'appel à l'API : {exc}")
                return

    if "res" not in st.session_state:
        st.info("Cliquez sur **Analyser le parc** pour lancer les prédictions.")
        return

    res = st.session_state["res"]
    render_kpis(res)
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        render_urgency_chart(res)
    with right:
        render_detail(res, st.session_state["df"])
    st.divider()
    render_table(res)


main()
