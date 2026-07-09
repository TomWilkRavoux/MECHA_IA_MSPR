# Stratégie de tests — Partie IA (MECHA / MSPR TPRE841)

> Détaille la **suite de tests** du projet : ce que couvre chaque fichier, à quel
> niveau de la pyramide il se place, ses dépendances et le job CI qui l'exécute.
> Complète [`ci-cd.md`](ci-cd.md) §2.1 (orchestration) et [`mlops.md`](../04-mlops/mlops.md)
> (gates qualité). Tous les tests vivent dans [`../../tests/`](../../tests).

## 1. Principes

- **Priorité aux fonctions pures.** La majorité des tests portent sur des fonctions
  sans effet de bord (règles métier, transformations de données, décisions du
  registre, gate de validation). Ils tournent **partout**, sans GPU, sans réseau,
  sans le dataset complet.
- **Pyramide de tests** : beaucoup d'unitaires rapides, quelques tests d'intégration
  (contrat d'API en process), une poignée de tests end-to-end (API conteneurisée par
  HTTP).
- **Aucune dépendance au jeu de données complet** : les tests qui ont besoin de
  données utilisent soit des DataFrames construits **en mémoire**, soit un **fixture
  léger** versionné ([`tests/fixtures/mecha_test_sample.csv`](../../tests/fixtures),
  3 machines).
- **Dégradation propre** : les tests qui exigent les artefacts modèles ou une API
  réseau se *skippent* automatiquement quand ceux-ci sont absents, au lieu d'échouer.

## 2. La pyramide

```
        e2e (test_e2e.py)                API réelle dans son conteneur, via HTTP
      ─────────────────────────          (job CI « e2e », après build-images)
     intégration (test_api.py)           TestClient FastAPI en process
    ───────────────────────────────      (job CI « test-ia »)
   unitaires (test_units, test_mlops,     fonctions pures : règles d'alerte, fenêtre
   test_prep, test_metrics,               glissante, préparation, métriques, registre,
   test_registry, test_validate_cli,      gate de validation, dictionnaire de données
   test_data_dictionary)                  (job CI « test-ia »)
```

## 3. Vue d'ensemble des fichiers

| Fichier | Niveau | Tests | Ce qu'il couvre | Dépendances | Job CI |
|---|---|---|---|---|---|
| [`test_units.py`](../../tests/test_units.py) | unitaire | 9 | Règle métier d'alerte, fenêtre glissante d'inférence, mise en forme des requêtes frontend | aucune (fonctions pures) | `test-ia` |
| [`test_mlops.py`](../../tests/test_mlops.py) | unitaire | 11 | Gate de validation (`check`), décision du registre (`next_registry`), producteur de flux (`sim.producer`) | aucune | `test-ia` |
| [`test_prep.py`](../../tests/test_prep.py) | unitaire | 11 | Préparation des données : clipping RUL, label `at_risk`, split anti-fuite, X/y, scaler, fenêtres glissantes | aucune (DataFrames en mémoire) | `test-ia` |
| [`test_metrics.py`](../../tests/test_metrics.py) | unitaire | 15 | Métriques classification/régression, score NASA, tableau markdown, générateurs de figures matplotlib | matplotlib (backend `Agg`) | `test-ia` |
| [`test_registry.py`](../../tests/test_registry.py) | unitaire | 15 | Registre de modèles : persistance, promotion, résolution d'artefact, copie vers baseline, CLI | aucune (chemins redirigés vers `tmp_path`) | `test-ia` |
| [`test_validate_cli.py`](../../tests/test_validate_cli.py) | unitaire | 4 | Point d'entrée du gate de validation (codes de sortie, rapport, fichier manquant) | aucune (stdlib) | `test-ia` |
| [`test_data_dictionary.py`](../../tests/test_data_dictionary.py) | unitaire | 3 | Cohérence doc ↔ schéma : le dictionnaire de données documente exactement les colonnes réelles | doc `dictionnaire_donnees.md` | `test-ia` |
| [`test_api.py`](../../tests/test_api.py) | intégration | 9 | Contrat de l'API FastAPI en process (validation d'entrée + inférence sur fixture) | FastAPI TestClient ; inférence sous `@needs_models` | `test-ia` |
| [`test_e2e.py`](../../tests/test_e2e.py) | end-to-end | 4 | API **conteneurisée** interrogée par HTTP (assemblage image + serveur + modèles) | `requests` + conteneur démarré | `e2e` |

**Total : 81 tests.**

## 4. Détail par fichier

### 4.1 Niveau unitaire

#### `test_units.py` — règles et transformations « cœur service »
- **Règle d'alerte** (`_alert_level`) : une machine non « à risque » est toujours `ok` ;
  sinon `warning`/`critical` selon les seuils `CRITICAL_RUL` / `CRITICAL_PROBA`.
- **Fenêtre glissante** (`ModelService._window_ending_at`) : left-pad des trajectoires
  trop courtes, troncature aux `SEQ_LEN` derniers cycles, fenêtre en milieu de trajectoire.
- **Requêtes frontend** (`api_client.build_single_request` / `build_requests`) : tri par
  cycle, filtrage des colonnes méta, regroupement par machine.

#### `test_mlops.py` — chaîne MLOps (décisions pures)
- **Gate de validation** (`validate_metrics.check`) : réussite dans les bornes, échec sur
  `min`/`max`, métrique manquante = échec.
- **Registre** (`registry.next_registry`) : le premier run devient courant, `promote=False`
  conserve le run servi, `promote=True` bascule le pointeur et dédoublonne.
- **Producteur de flux** (`sim.producer`) : sélection/filtrage de machines, tri par cycle,
  fenêtre `history_upto`, construction de requête.

#### `test_prep.py` — préparation des données (`ml.prep`)
- `clip_rul` (clipping piecewise), `add_at_risk` (seuil 30), `split_by_machine`
  (**anti-fuite** : aucune machine à cheval train/val), `to_xy`, `fit_scaler`,
  `make_windows` / `make_test_windows` (formes, left-pad, alignement de la cible sur le
  dernier cycle). DataFrames polars construits en mémoire — aucun CSV requis.

#### `test_metrics.py` — métriques & graphes (`ml.metrics`)
- **Métriques** : classification (accuracy/precision/recall/F1/ROC-AUC), régression
  (RMSE/MAE/R²), **score asymétrique NASA** (RUL surestimé pénalisé plus fort), tableau
  markdown comparatif.
- **Graphes** : chaque `plot_*` retourne une `Figure` matplotlib ; `save_fig` écrit un PNG.
  Backend `Agg` forcé (sans affichage) pour la CI.

#### `test_registry.py` — registre de modèles versionné (`ml.registry`)
- Persistance (`register_run` → relecture disque), promotion, **résolution d'artefact**
  (run courant présent → baseline en fallback), rejet d'un run inconnu, copie vers la
  baseline, et le **CLI** (`list` / `promote` / `use-baseline` / `--to-baseline`).
- Les chemins du module (`MODELS_DIR`, `RUNS_DIR`, `REGISTRY_PATH`) sont redirigés vers un
  `tmp_path` isolé : aucun artefact réel n'est touché.

#### `test_validate_cli.py` — point d'entrée du gate (`ml.validate_metrics.main`)
- Fichier de métriques absent → code `2` ; modèle conforme → code `0` ; régression ou
  métrique manquante → code `1`. La logique pure `check()` est, elle, couverte par
  `test_mlops.py`.

#### `test_data_dictionary.py` — cohérence doc ↔ schéma
- Le dictionnaire de données ([`dictionnaire_donnees.md`](../02-donnees/dictionnaire_donnees.md))
  documente **exactement** les colonnes du schéma réel (`ml.prep`) : ni oubli, ni colonne
  fantôme. Garde-fou : 3 réglages + 21 capteurs = 24 features.

### 4.2 Niveau intégration — `test_api.py`
- **Contrat d'entrée** (sans modèles ni données) : `/health` et `/features` exposent les
  seuils métier ; une feature manquante ou une liste vide renvoie **422** (validation
  Pydantic).
- **Inférence** (sous `@needs_models`) : `/predict`, `/predict/batch`, `/predict/trajectory`
  sur le fixture — bornes des probabilités, cohérence `at_risk` ↔ niveau d'alerte, et
  **égalité batch = N appels unitaires** (un seul forward pour tout le parc).

### 4.3 Niveau end-to-end — `test_e2e.py`
- Interroge une **API réellement démarrée** (image `mecha-backend`, modèles embarqués) via
  la couche réseau : `/health`, `/predict`, `/predict/batch`, et le rejet **422** de bout
  en bout. Ne dépend que de `requests` ; la liste des features est lue depuis l'API elle-même.

## 5. Marqueurs & exécution conditionnelle

| Marqueur | Où | Effet |
|---|---|---|
| `@needs_models` | `test_api.py` | *Skip* les tests d'inférence si `models/*.pt` + `scaler.joblib` sont absents. En CI ils sont **committés**, donc ces tests s'exécutent réellement. |
| `pytestmark = skipif(not _api_reachable())` | `test_e2e.py` | *Skip* tout le fichier si aucune API n'écoute sur `MECHA_E2E_URL` — inoffensif quand `pytest` collecte tout sans conteneur. |

## 6. Lancer les tests

```bash
# Depuis la racine — toute la suite
uv run pytest -q

# Avec couverture (comme le job CI test-ia)
uv run pytest -q --cov=ml --cov=backend --cov-report=term-missing

# Un seul fichier / un seul test
uv run pytest tests/test_prep.py -q
uv run pytest tests/test_registry.py::test_resolve_falls_back_to_baseline_when_run_file_absent

# End-to-end (nécessite le conteneur démarré)
docker compose up -d --build backend
MECHA_E2E_URL=http://localhost:8000 uvx --with requests --with pytest pytest tests/test_e2e.py
docker compose down
```

## 7. Couverture

La couverture est mesurée sur le **cœur IA** exécuté par les tests (`ml/`, `backend/`) et
transmise à SonarCloud (`coverage.xml`). Sont **exclus de la mesure**
(cf. [`sonar-project.properties`](../../sonar-project.properties)) :

- `tests/**`, `**/__init__.py` ;
- `frontend/**`, `sim/**` (hors périmètre du cœur IA) ;
- `ml/train.py` : pipeline d'entraînement (torch + dataset C-MAPSS complet), **validé
  manuellement / via le job `e2e`**, pas par des tests unitaires.

Le Quality Gate SonarCloud exige **≥ 80 % de couverture sur le code nouveau**.
