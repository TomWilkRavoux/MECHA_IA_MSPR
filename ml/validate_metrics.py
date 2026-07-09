"""Gate de validation des performances (intégration continue).

Compare `models/metrics.json` (produit par `ml/train.py`) aux seuils minimaux de
`ml/metrics_thresholds.json` et **échoue (code de sortie 1) en cas de régression**.
C'est le garde-fou « qualité modèle » de la chaîne : on ne merge pas un modèle
dont les performances passent sous le contrat.

Volontairement **sans dépendance** (stdlib seule) : le job CI qui l'exécute n'a
besoin ni de torch ni des données, il lit le rapport de métriques committé.

Usage (depuis la racine) :
    uv run python -m ml.validate_metrics
    python -m ml.validate_metrics --metrics models/metrics.json --thresholds ml/metrics_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ml import registry  # stdlib seule : ne casse pas l'exécution CI sans torch/données

ROOT = Path(__file__).resolve().parent.parent  # racine du projet
# Résout le run courant si un registre existe, sinon la baseline plate committée.
# En CI (runs/ ignoré par git), c'est bien le metrics.json committé qui est validé.
DEFAULT_METRICS = registry.resolve("metrics.json")
DEFAULT_THRESHOLDS = ROOT / "ml" / "metrics_thresholds.json"


def check(metrics: dict, thresholds: dict) -> list[dict]:
    """Confronte les métriques aux seuils. Retourne la liste des lignes de contrôle.

    Chaque ligne : {group, metric, value, bound, op, ok}. Une valeur manquante est
    traitée comme un échec (le contrat exige la métrique).
    """
    rows: list[dict] = []
    for group, rules in thresholds.items():
        if group.startswith("_"):  # clés de commentaire
            continue
        observed = metrics.get(group, {})
        for metric, rule in rules.items():
            value = observed.get(metric)
            if "min" in rule:
                bound, op = rule["min"], ">="
                ok = value is not None and value >= bound
            else:
                bound, op = rule["max"], "<="
                ok = value is not None and value <= bound
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "value": value,
                    "op": op,
                    "bound": bound,
                    "ok": ok,
                }
            )
    return rows


def _fmt(v) -> str:
    return "manquant" if v is None else f"{v:.3f}"


def _resolve_under_root(path: Path) -> Path:
    """Canonicalise un chemin CLI et le contraint au dépôt (anti-traversal, CWE-22).

    `--metrics` / `--thresholds` viennent de la ligne de commande : le chemin est
    résolu (`os.path.realpath`) puis doit rester sous `ROOT` — un chemin fabriqué
    (`../../…`) est rejeté avant tout accès disque. Les tests substituent `ROOT`
    par leur répertoire temporaire.
    """
    base = os.path.realpath(ROOT)
    resolved = os.path.realpath(path)
    if not resolved.startswith(base + os.sep):
        raise SystemExit(f"Chemin hors du projet refusé : {path}")
    return Path(resolved)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gate de validation des métriques modèle.")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    args = p.parse_args(argv)
    metrics_path = _resolve_under_root(args.metrics)
    thresholds_path = _resolve_under_root(args.thresholds)

    if not metrics_path.exists():
        print(f"ERREUR : {args.metrics} introuvable. Lancer d'abord `python -m ml.train`.")
        return 2

    metrics = json.loads(metrics_path.read_text())
    thresholds = json.loads(thresholds_path.read_text())
    rows = check(metrics, thresholds)

    gen = metrics.get("generated_at", "?")
    print(f"Validation du modèle — rapport du {gen} (mode: {metrics.get('mode', '?')})\n")
    print(f"  {'métrique':<22} {'valeur':>10}  {'seuil':>12}   état")
    print(f"  {'-' * 22} {'-' * 10}  {'-' * 12}   ----")
    for r in rows:
        label = f"{r['group']}.{r['metric']}"
        bound = f"{r['op']} {r['bound']}"
        state = "OK" if r["ok"] else "ÉCHEC"
        print(f"  {label:<22} {_fmt(r['value']):>10}  {bound:>12}   {state}")

    failures = [r for r in rows if not r["ok"]]
    if failures:
        print(f"\n✗ {len(failures)} régression(s) détectée(s) — modèle refusé.")
        return 1
    print(f"\n✓ {len(rows)} contrôles passés — modèle conforme au contrat de performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
