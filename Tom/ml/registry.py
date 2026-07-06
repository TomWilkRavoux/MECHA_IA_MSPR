"""Registre de modèles versionné (brique MLOps « Registry »).

Chaque ré-entraînement (`ml/train.py`) écrit ses artefacts dans un dossier daté
`models/runs/<run_id>/` (2 têtes LSTM + scaler + metrics.json), et le pointeur
`models/registry.json` désigne le run **courant** — celui que servent le backend
et l'évaluation.

Les artefacts **plats** `models/*.pt` / `models/scaler.joblib` restent la
**BASELINE immuable committée** : `train.py` ne les écrase jamais. Si aucun run
n'est enregistré (ex. en CI, où `runs/` est ignoré par git), `resolve()` retombe
sur la baseline — le backend conteneurisé et le gate CI fonctionnent à
l'identique, sans registre.

Promouvoir un run **en production** est une action explicite et traçable :
`--to-baseline` copie les artefacts du run courant sur les fichiers plats
committés (à commiter, donc reviewable), ce qui referme la boucle CI → prod.

Usage (depuis Tom/) :
    uv run python -m ml.registry list                # liste les runs + le courant
    uv run python -m ml.registry promote <run_id>    # déplace le pointeur courant
    uv run python -m ml.registry use-baseline         # sert de nouveau la baseline plate
    uv run python -m ml.registry promote <run_id> --to-baseline   # copie -> baseline (prod)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # dossier Tom/
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Artefacts qui composent un modèle servable (têtes + scaler + rapport).
ARTIFACTS = ("lstm_classifier.pt", "lstm_regressor.pt", "scaler.joblib", "metrics.json")


# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
def new_run_id() -> str:
    """Identifiant de run trié chronologiquement (UTC, sûr pour un nom de dossier)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(run_id: str) -> Path:
    """Dossier des artefacts d'un run (créé à la demande)."""
    return RUNS_DIR / run_id


# ----------------------------------------------------------------------------
# Pointeur (registry.json)
# ----------------------------------------------------------------------------
def load_registry() -> dict:
    """Lit le pointeur ; retourne un registre vide s'il n'existe pas encore."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"current": None, "runs": []}


def _save_registry(reg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")


def next_registry(reg: dict, entry: dict, *, promote: bool) -> dict:
    """Décision **pure** : insère/remplace `entry` et fixe le run courant.

    Le pointeur bascule sur le nouveau run si `promote` est vrai **ou** si aucun
    run n'était courant (premier run). Factorisé hors des E/S pour être testable.
    """
    run_id = entry["run_id"]
    runs = [r for r in reg.get("runs", []) if r.get("run_id") != run_id] + [entry]
    current = run_id if (promote or reg.get("current") is None) else reg.get("current")
    return {"current": current, "runs": runs}


def register_run(run_id: str, *, metrics: dict, promote: bool = True) -> dict:
    """Enregistre un run entraîné dans le pointeur et le promeut (par défaut)."""
    entry = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": f"runs/{run_id}",
        "metrics": {
            "classification": metrics.get("classification", {}),
            "regression": metrics.get("regression", {}),
        },
    }
    reg = next_registry(load_registry(), entry, promote=promote)
    _save_registry(reg)
    return reg


def current_run_id() -> str | None:
    """Run actuellement servi, ou None (= baseline plate)."""
    return load_registry().get("current")


def set_current(run_id: str | None) -> dict:
    """Fixe le run courant. `None` rebascule sur la baseline plate."""
    reg = load_registry()
    known = {r["run_id"] for r in reg.get("runs", [])}
    if run_id is not None and run_id not in known:
        raise ValueError(f"run inconnu : {run_id!r} (connus : {sorted(known) or 'aucun'})")
    reg["current"] = run_id
    _save_registry(reg)
    return reg


# ----------------------------------------------------------------------------
# Résolution d'artefact (consommée par prep / lstm / validate_metrics)
# ----------------------------------------------------------------------------
def resolve(filename: str) -> Path:
    """Chemin d'un artefact : run courant s'il est défini **et présent**, sinon baseline.

    Ce fallback garantit que le backend et la CI fonctionnent même sans registre
    (les fichiers plats committés restent la source par défaut).
    """
    run_id = current_run_id()
    if run_id:
        candidate = run_dir(run_id) / filename
        if candidate.exists():
            return candidate
    return MODELS_DIR / filename


# ----------------------------------------------------------------------------
# Promotion vers la baseline (mise en production explicite)
# ----------------------------------------------------------------------------
def promote_to_baseline(run_id: str) -> list[Path]:
    """Copie les artefacts d'un run sur les fichiers plats committés (baseline).

    Action volontairement explicite : les fichiers plats sont versionnés par git,
    donc cette copie doit être commitée — la mise en production reste reviewable.
    """
    src = run_dir(run_id)
    if not src.is_dir():
        raise ValueError(f"run introuvable : {src.relative_to(ROOT)}")
    copied = []
    for name in ARTIFACTS:
        s = src / name
        if s.exists():
            shutil.copy2(s, MODELS_DIR / name)
            copied.append(MODELS_DIR / name)
    return copied


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _cmd_list(_args) -> int:
    reg = load_registry()
    current = reg.get("current")
    runs = reg.get("runs", [])
    print(f"Courant : {current or 'baseline (fichiers plats models/*)'}\n")
    if not runs:
        print("  (aucun run enregistré — seule la baseline est disponible)")
        return 0
    print(f"  {'':1} {'run_id':<18} {'F1':>6} {'RMSE':>7}  créé")
    for r in runs:
        mark = "→" if r["run_id"] == current else " "
        clf = r.get("metrics", {}).get("classification", {})
        rul = r.get("metrics", {}).get("regression", {})
        f1 = clf.get("f1")
        rmse = rul.get("RMSE")
        f1s = f"{f1:.3f}" if isinstance(f1, (int, float)) else "  -  "
        rmses = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "  -  "
        print(f"  {mark} {r['run_id']:<18} {f1s:>6} {rmses:>7}  {r.get('created_at', '?')}")
    return 0


def _cmd_promote(args) -> int:
    set_current(args.run_id)
    print(f"Run courant → {args.run_id}")
    if args.to_baseline:
        copied = promote_to_baseline(args.run_id)
        rels = ", ".join(p.name for p in copied)
        print(f"Copié sur la baseline committée : {rels}")
        print("→ commit ces fichiers pour mettre le modèle en production.")
    return 0


def _cmd_use_baseline(_args) -> int:
    set_current(None)
    print("Run courant → baseline (fichiers plats models/*).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Registre de modèles versionné MECHA.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Liste les runs et le run courant.").set_defaults(fn=_cmd_list)

    pp = sub.add_parser("promote", help="Fixe le run courant (pointeur servi).")
    pp.add_argument("run_id")
    pp.add_argument("--to-baseline", action="store_true",
                    help="Copie aussi les artefacts sur la baseline plate committée (prod).")
    pp.set_defaults(fn=_cmd_promote)

    sub.add_parser("use-baseline", help="Rebascule le pointeur sur la baseline plate.").set_defaults(fn=_cmd_use_baseline)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
