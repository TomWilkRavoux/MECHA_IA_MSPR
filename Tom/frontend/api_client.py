"""Client HTTP de l'API de maintenance prédictive.

Le frontend est **totalement découplé** du modèle : il ne connaît que les
endpoints REST du backend (aucun import de `ml/` ni de `torch`). Cela reflète
l'architecture cible du CDC (§5) : IA exposée en service, consommée par l'appli.
"""

from __future__ import annotations

import pandas as pd
import requests

# Colonnes d'identification (non envoyées au modèle) présentes dans les CSV MECHA.
META_COLS = ["machine_id", "subset", "usine", "ligne_production", "unit", "cycle"]


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def features(self) -> list[str]:
        r = requests.get(f"{self.base_url}/features", timeout=self.timeout)
        r.raise_for_status()
        return r.json()["features"]

    def predict(self, machine: dict) -> dict:
        """Prédiction pour UNE machine (endpoint `/predict`) -> stats + niveau d'alerte."""
        r = requests.post(
            f"{self.base_url}/predict",
            json=machine,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def predict_batch(self, machines: list[dict]) -> list[dict]:
        r = requests.post(
            f"{self.base_url}/predict/batch",
            json={"machines": machines},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]

    def predict_trajectory(self, machine: dict) -> list[dict]:
        """Trajectoire cycle par cycle d'une machine -> liste de points prédits."""
        r = requests.post(
            f"{self.base_url}/predict/trajectory",
            json=machine,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["points"]


def build_single_request(machine_id, g: pd.DataFrame, features: list[str]) -> dict:
    """(cycles d'UNE machine) -> corps `/predict` ou `/predict/trajectory`.

    Ordonne par `cycle` et construit un cycle `{values: {feature: valeur}}` par
    ligne. Les colonnes méta sont ignorées.
    """
    g = g.sort_values("cycle") if "cycle" in g.columns else g
    cycles = [{"values": {f: float(row[f]) for f in features}} for _, row in g.iterrows()]
    return {"machine_id": str(machine_id), "cycles": cycles}


def build_requests(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """(DataFrame de cycles machine) -> corps `/predict/batch`.

    Regroupe par `machine_id` et délègue à `build_single_request` pour chaque machine.
    """
    return [
        build_single_request(machine_id, g, features)
        for machine_id, g in df.groupby("machine_id", sort=False)
    ]
