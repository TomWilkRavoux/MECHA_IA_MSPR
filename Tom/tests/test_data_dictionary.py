"""Validation du dictionnaire de données.

Vérifie que `docs/dictionnaire_donnees.md` documente **exactement** les colonnes du
schéma réel (source de vérité : `ml.prep`), sans oubli ni colonne fantôme. Le doc est
la source humaine ; ce test empêche toute dérive silencieuse doc ↔ données.
"""

from __future__ import annotations

import re
from pathlib import Path

from ml.prep import FEATURES, KEYS

DOC = Path(__file__).resolve().parents[1] / "docs" / "02-donnees" / "dictionnaire_donnees.md"

# Colonnes attendues : identité (KEYS) + réglages/capteurs (FEATURES) + cibles + vérité terrain.
EXPECTED = set(KEYS) | set(FEATURES) | {"RUL", "at_risk", "RUL_true"}

# 1re cellule d'une ligne de tableau contenant un identifiant `col` (backticks, mot simple).
_FIRST_COL = re.compile(r"^`([A-Za-z0-9_]+)`$")


def _documented_columns() -> set[str]:
    cols: set[str] = set()
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 2:
            continue
        m = _FIRST_COL.match(cells[1])  # cells[0] est vide (avant le 1er '|')
        if m:
            cols.add(m.group(1))
    return cols


def test_doc_exists():
    assert DOC.exists(), f"Dictionnaire introuvable : {DOC}"


def test_dictionary_matches_schema():
    documented = _documented_columns()
    missing = EXPECTED - documented       # colonnes du schéma non documentées
    extra = documented - EXPECTED         # colonnes documentées inexistantes
    assert not missing, f"Colonnes non documentées : {sorted(missing)}"
    assert not extra, f"Colonnes documentées mais absentes du schéma : {sorted(extra)}"


def test_feature_count():
    # Garde-fou : 3 réglages + 21 capteurs = 24 features.
    assert len(FEATURES) == 24
