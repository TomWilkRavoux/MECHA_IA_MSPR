"""Appels API non-visuels : seuils métier (`/health`) et prédiction batch du parc."""

from __future__ import annotations

import pandas as pd
import requests

from api_client import ApiClient, build_requests

from ..style.theme import DEFAULT_THRESHOLDS, STATUS_LABEL


def get_thresholds(client: ApiClient) -> dict:
    """Récupère les seuils métier depuis /health (repli sur les valeurs par défaut)."""
    try:
        h = client.health()
        return {k: h.get(k, DEFAULT_THRESHOLDS[k]) for k in DEFAULT_THRESHOLDS}
    except requests.RequestException:
        return DEFAULT_THRESHOLDS


def run_predictions(client: ApiClient, df: pd.DataFrame, n_max: int) -> pd.DataFrame:
    """Appel `/predict/batch` sur les `n_max` premières machines + mise en forme."""
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

    res["badge"] = res["alert_level"].map(STATUS_LABEL)
    return res.sort_values("rul_predicted").reset_index(drop=True)
