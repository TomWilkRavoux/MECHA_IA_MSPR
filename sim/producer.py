"""Producteur de flux temps réel simulé pour MECHA.

Émule la remontée continue des capteurs : rejoue un CSV de trajectoires machines
**cycle par cycle**, et à chaque « tick » interroge l'API (`/predict/batch`) comme
le ferait une supervision temps réel. Restitue une vue parc vivante en console et
journalise les alertes (`warning`/`critical`) dans `output/alertes_live.csv`.

C'est la démonstration du **chemin temps réel** de l'architecture : dans le
prototype, le connecteur IoT/SCADA est remplacé par ce rejeu de CSV cadencé ;
l'aval (API, seuils, alertes) est identique à ce qu'il serait en production.

Prérequis : le backend doit tourner (`uv run uvicorn backend.api.main:app`).

Usage (depuis la racine) :
    uv run python -m sim.producer                      # 8 machines, 1 cycle/s
    uv run python -m sim.producer --machines 5 --interval 0.3 --max-ticks 60
    uv run python -m sim.producer --usine Usine_1 --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from ml.prep import FEATURES, SEQ_LEN

ROOT = Path(__file__).resolve().parent.parent  # racine du projet
DEFAULT_CSV = ROOT / "assets" / "KaggleDataset" / "mecha_test_classification.csv"
DEFAULT_OUT = ROOT / "output" / "alertes_live.csv"
META_COLS = ["machine_id", "subset", "usine", "ligne_production", "unit", "cycle"]
ALERT_COLS = [
    "timestamp",
    "tick",
    "machine_id",
    "usine",
    "ligne_production",
    "alert_level",
    "risk_probability",
    "rul_predicted",
]


# ----------------------------------------------------------------------------
# Validation des entrées CLI (anti-SSRF / anti-traversal)
# ----------------------------------------------------------------------------
def _validate_api_url(url: str) -> str:
    """Valide l'URL de l'API avant tout appel réseau : schéma http(s) + hôte (anti-SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SystemExit(f"URL d'API invalide : {url!r} (attendu http(s)://hôte[:port]).")
    return url.rstrip("/")


# ----------------------------------------------------------------------------
# Préparation du jeu rejoué (fonctions pures, testables sans réseau)
# ----------------------------------------------------------------------------
def select_machines(df: pd.DataFrame, n: int, usine: str | None) -> list[str]:
    """Choisit jusqu'à `n` machines (filtrées par usine si fournie)."""
    if usine and "usine" in df.columns:
        df = df[df["usine"].astype(str) == usine]
    ids = list(dict.fromkeys(df["machine_id"].tolist()))  # ordre préservé, dédupliqué
    return ids[:n]


def machine_frames(df: pd.DataFrame, ids: list[str]) -> dict[str, pd.DataFrame]:
    """Découpe le DataFrame en {machine_id: trajectoire triée par cycle}."""
    frames = {}
    for mid in ids:
        g = df[df["machine_id"] == mid]
        frames[mid] = g.sort_values("cycle").reset_index(drop=True)
    return frames


def history_upto(frame: pd.DataFrame, tick: int, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    """Les `seq_len` derniers cycles observés jusqu'au tick `tick` (1-indexé)."""
    seen = frame.iloc[:tick]
    return seen.iloc[-seq_len:]


def build_request(machine_id: str, rows: pd.DataFrame, features: list[str]) -> dict:
    """(cycles observés d'une machine) -> corps `/predict` (via /predict/batch)."""
    cycles = [{"values": {f: float(r[f]) for f in features}} for _, r in rows.iterrows()]
    return {"machine_id": str(machine_id), "cycles": cycles}


# ----------------------------------------------------------------------------
# Boucle de simulation
# ----------------------------------------------------------------------------
_LEVEL_STYLE = {
    "ok": "\033[32m● ok\033[0m",
    "warning": "\033[33m▲ warning\033[0m",
    "critical": "\033[31m■ critical\033[0m",
}


def _wait_backend(api_url: str, retries: int = 10) -> None:
    for _ in range(retries):
        try:
            h = requests.get(f"{api_url}/health", timeout=5).json()
            if h.get("models_loaded"):
                print(f"Backend prêt (device={h.get('device')}, seq_len={h.get('seq_len')}).")
                return
        except requests.RequestException:
            pass
        print("  … attente du backend")
        time.sleep(2)
    raise SystemExit(f"Backend injoignable/non prêt sur {api_url} (démarrer l'API d'abord).")


_ALERT_ORDER = {"ok": 0, "warning": 1, "critical": 2}


def _build_batch(tick: int, ids: list[str], frames: dict[str, pd.DataFrame]):
    """Lot `/predict/batch` du tick courant + liste des machines encore actives."""
    batch, active = [], []
    for mid in ids:
        if tick <= len(frames[mid]):
            rows = history_upto(frames[mid], tick)
            batch.append(build_request(mid, rows, FEATURES))
            active.append(mid)
    return batch, active


def _log_alert_rises(active, results, meta, last_level, writer, fh, tick):
    """Compte les niveaux et journalise les MONTÉES d'alerte. Retourne (counts, n_journalisées)."""
    counts = {"ok": 0, "warning": 0, "critical": 0}
    logged = 0
    for mid in active:
        res = results[mid]
        lvl = res["alert_level"]
        counts[lvl] = counts.get(lvl, 0) + 1
        # Journalise à la MONTÉE d'alerte (ok->warning/critical ou warning->critical).
        if _ALERT_ORDER[lvl] > _ALERT_ORDER[last_level.get(mid, "ok")]:
            writer.writerow(
                {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "tick": tick,
                    "machine_id": mid,
                    "usine": meta[mid].get("usine", ""),
                    "ligne_production": meta[mid].get("ligne_production", ""),
                    "alert_level": lvl,
                    "risk_probability": res["risk_probability"],
                    "rul_predicted": res["rul_predicted"],
                }
            )
            fh.flush()
            logged += 1
        last_level[mid] = lvl
    return counts, logged


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Flux temps réel simulé -> API MECHA.")
    p.add_argument("--api-url", default=os.getenv("MECHA_API_URL", "http://localhost:8000"))
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--machines", type=int, default=8, help="Nombre de machines simulées.")
    p.add_argument("--usine", default=None, help="Filtrer sur une usine.")
    p.add_argument("--interval", type=float, default=1.0, help="Secondes entre deux cycles.")
    p.add_argument("--max-ticks", type=int, default=None, help="Arrêt après N cycles.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    api_url = _validate_api_url(args.api_url)

    if not args.csv.exists():
        raise SystemExit(f"CSV introuvable : {args.csv}")

    df = pd.read_csv(args.csv)
    ids = select_machines(df, args.machines, args.usine)
    if not ids:
        raise SystemExit("Aucune machine sélectionnée (vérifier --usine).")
    frames = machine_frames(df, ids)
    meta = {mid: frames[mid].iloc[0] for mid in ids}
    horizon = max(len(f) for f in frames.values())
    if args.max_ticks:
        horizon = min(horizon, args.max_ticks)

    _wait_backend(api_url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    fh = args.out.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=ALERT_COLS)
    if new_file:
        writer.writeheader()

    print(f"\nSimulation : {len(ids)} machines · {horizon} cycles · {args.interval}s/cycle")
    print(f"Alertes journalisées → {args.out.relative_to(ROOT)}\n")

    last_level: dict[str, str] = {}
    n_alerts = 0
    try:
        for tick in range(1, horizon + 1):
            batch, active = _build_batch(tick, ids, frames)
            if not batch:
                break

            r = requests.post(f"{api_url}/predict/batch", json={"machines": batch}, timeout=30)
            r.raise_for_status()
            results = {res["machine_id"]: res for res in r.json()["results"]}

            counts, logged = _log_alert_rises(active, results, meta, last_level, writer, fh, tick)
            n_alerts += logged

            crit = [m for m in active if results[m]["alert_level"] == "critical"]
            line = (
                f"tick {tick:3d}/{horizon}  actives={len(active):2d}  "
                f"{_LEVEL_STYLE['ok']} {counts['ok']:2d}  "
                f"{_LEVEL_STYLE['warning']} {counts['warning']:2d}  "
                f"{_LEVEL_STYLE['critical']} {counts['critical']:2d}"
            )
            if crit:
                line += "   →critiques: " + ", ".join(str(m) for m in crit[:6])
            print(line)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        fh.close()

    print(
        f"\nTerminé — {n_alerts} montée(s) d'alerte journalisée(s) dans {args.out.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
