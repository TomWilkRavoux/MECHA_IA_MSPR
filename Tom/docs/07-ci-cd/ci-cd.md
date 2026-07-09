# Intégration & livraison continues (CI/CD) MECHA / MSPR TPRE841

> Répond à l'exigence du CDC : *« appliquer l'intégration continue »* et
> *« principes de déploiement »*. Complète [`mlops.md`](../04-mlops/mlops.md) (le gate
> qualité modèle, décrit ici côté orchestration) et
> [`deploiement_vps.md`](../06-deploiement/deploiement_vps.md) (mise en prod).

L'outil est **CircleCI** (fichier unique [`.circleci/config.yml`](../../../.circleci/config.yml),
à la racine du dépôt). Toute la partie IA vit dans `Tom/`, aussi les jobs Python se
placent en `working_directory: ~/project/Tom`.

La chaîne couvre quatre familles de risques du cours : **gouvernance de branche**,
**sécurité** (secrets + dépendances), **tests** (unitaires → intégration → E2E), et
**qualité** (lint Ruff + analyse SonarCloud + gate de métriques modèle). Une **livraison**
(build des images sur tag) complète l'intégration.

## 1. Vue d'ensemble — deux workflows

```
 WORKFLOW « ci »  (à chaque push / pull request)
 ─────────────────────────────────────────────────────────────────────
   block-merge-rules ─┐  (gouvernance gitflow)
   scan-secrets       │
   lint               │   jobs en parallèle
   deps-audit         │
   validate-model     │
   test-ia ───────────┼──▶ build-images ──▶ e2e ──▶ sonarcloud
                            (build Docker)   (HTTP)   (qualité + couverture)

 WORKFLOW « cd »  (uniquement au push d'un tag vX.Y.Z)
 ─────────────────────────────────────────────────────────────────────
   build-images   (build-only des images de release, sans push)
```

L'**orchestration** repose sur les dépendances `requires:` :
- les jobs rapides et indépendants tournent **en parallèle** (fail-fast : un lint ou un
  test cassé remonte tout de suite) ;
- `build-images` n'est lancé **que si `test-ia` passe** (inutile de construire une image
  sur un code dont les tests échouent) ;
- `e2e` n'est lancé **que si `build-images` réussit** (il a besoin de l'image) ;
- `sonarcloud` clôt la chaîne : il **attend `test-ia`** (dont il consomme la couverture)
  **et `e2e`**, pour n'analyser qu'un code déjà validé.

## 2. Détail des jobs

| Job | Exécuteur | Rôle | Bloquant ? |
|---|---|---|---|
| `block-merge-rules` | `cimg/base` | Applique le **gitflow** : seules `release/*` et `hotfix/*` peuvent viser `main` ; `develop` et `hotfix/*` vers `integration`. | ✅ |
| `scan-secrets` | `cimg/node` | **gitleaks** : détecte des secrets committés (clés, tokens). | ✅ |
| `lint` | `cimg/python:3.12` | **Ruff** (`ruff check`) : erreurs de code, imports inutilisés/mal triés, modernisation (`pyupgrade`), bugs probables (`flake8-bugbear`). | ✅ |
| `test-ia` | `cimg/python:3.12` | Installe les deps depuis le **lockfile** (`uv sync --extra cpu`) et lance **pytest** (contrat d'API, unitaires, dictionnaire de données) avec couverture. | ✅ |
| `validate-model` | `cimg/python:3.12` | **Gate qualité modèle** : `ml/validate_metrics.py` refuse un modèle régressant sous `ml/metrics_thresholds.json`. Stdlib seule, aucune donnée. Détail dans [`mlops.md`](../04-mlops/mlops.md) §3. | ✅ |
| `deps-audit` | `cimg/python:3.12` | **pip-audit** : vulnérabilités connues (CVE) des dépendances, en complément de gitleaks. | ✅ (voir §5) |
| `build-images` | `machine` (Docker) | `docker compose build` : valide que les images backend & frontend se construisent (conteneurisation, CDC §8.1). | ✅ |
| `e2e` | `machine` (Docker) | Démarre l'API **dans son conteneur** (modèles LSTM embarqués) et vérifie une prédiction **de bout en bout par HTTP** (`tests/test_e2e.py`). | ✅ |
| `sonarcloud` | `cimg/python:3.12` | **SonarCloud** : analyse statique (bugs, code smells, duplication, maintenabilité) + **couverture** (`coverage.xml` de `test-ia`). En fin de chaîne. | ⚠️ garde sur `SONAR_TOKEN` (voir §5.1) |

### 2.1 La pyramide de tests

Trois niveaux, du plus isolé au plus intégré :

```
        e2e (tests/test_e2e.py)          API réelle dans son conteneur, via HTTP
      ─────────────────────────          (job e2e, après build-images)
     intégration (tests/test_api.py)     TestClient FastAPI en process
    ───────────────────────────────      (job test-ia)
   unitaires (test_units, test_mlops,     fonctions pures : règles d'alerte,
   test_data_dictionary)                  fenêtre glissante, gate, registre…
```

- `test-ia` exécute les niveaux **unitaire + intégration**. Les modèles d'inférence
  étant **committés** (`models/*.pt`, `scaler.joblib`), les tests marqués `@needs_models`
  s'y exécutent réellement (ils ne se *skippent* que si les artefacts sont absents).
- `e2e` exécute le niveau **end-to-end** : `test_e2e.py` ne dépend que de `requests`
  (la liste des features est lue depuis l'API `/features`, le jeu d'exemple depuis le
  fixture CSV). Il *skip* proprement si aucune API n'écoute, donc il reste inoffensif
  quand `test-ia` collecte l'ensemble des tests.

## 3. Gestion des dépendances — `uv` + extras CPU/GPU

Le projet est géré par **uv** (`pyproject.toml` + `uv.lock`). La CI installe donc les
dépendances **depuis le lockfile** (`uv sync --frozen`) : une seule source de vérité,
versions reproductibles, plus de liste maintenue à la main dans la config.

`torch` a la particularité d'exister en deux variantes exclusives, exposées en **extras** :

| Extra | Index PyTorch | Usage |
|---|---|---|
| `cu128` | `download.pytorch.org/whl/cu128` | poste de dev avec **GPU NVIDIA** |
| `cpu` | `download.pytorch.org/whl/cpu` | **CI** et serving : torch CPU, **sans** la pile CUDA (plus léger) |

```bash
uv sync --extra cu128     # dev GPU
uv sync --extra cpu       # CI / sans GPU
```

La CI utilise **toujours `--extra cpu`** : le runner CircleCI n'a pas de GPU et cela évite
de télécharger plusieurs Go de wheels CUDA. Le dossier de cache uv (`~/.cache/uv`) est
mis en cache par CircleCI (clé sur le hash de `uv.lock`) pour accélérer les runs suivants.

## 4. Déclenchement & exécution

### Quand ça tourne
- **Workflow `ci`** : à chaque `push` et sur chaque **pull request** (tous les jobs).
- **Workflow `cd`** : **uniquement** au push d'un **tag** `vX.Y.Z` (ex. `v1.0.0`).
  Rappel CircleCI : un job ne se déclenche jamais sur un tag sans filtre `tags:` explicite ;
  le workflow `cd` filtre `branches: ignore /.*/` + `tags: only /^v\d+\.\d+\.\d+$/`.

### Reproduire chaque job en local (depuis `Tom/`)

```bash
# lint
uvx ruff check .

# test-ia (unitaires + intégration, torch CPU)
uv sync --extra cpu
uv run pytest -q --cov=ml --cov=backend

# validate-model (gate qualité modèle, stdlib seule)
uv run python -m ml.validate_metrics

# deps-audit (scan de vulnérabilités des dépendances)
uv run --with pip-audit pip-audit

# build-images
docker compose build

# e2e (API conteneurisée + test HTTP)
docker compose up -d --build backend
#   … attendre le healthcheck (curl http://localhost:8000/health) …
MECHA_E2E_URL=http://localhost:8000 uvx --with requests --with pytest pytest tests/test_e2e.py
docker compose down

# sonarcloud (nécessite SONAR_TOKEN + projet SonarCloud, cf. §5.1)
uv run pytest --cov=ml --cov=backend --cov-report=xml   # produit coverage.xml
SONAR_TOKEN=xxxxx sonar-scanner   # sonar-scanner CLI, lit sonar-project.properties
```

### Déclencher une « release » (workflow `cd`)

```bash
git tag v1.0.0
git push origin v1.0.0     # → CircleCI lance le workflow cd (build des images)
```

## 5. Sécurité & choix d'implémentation

- **Deux angles de sécurité** : `scan-secrets` (gitleaks, secrets dans le code) **et**
  `deps-audit` (pip-audit, CVE des dépendances) — deux risques distincts du cours.
- **`deps-audit` bloquant** : toute CVE connue fait échouer la CI. Les transitifs
  vulnérables (`urllib3`, `idna`, `setuptools`) sont maintenus à des versions corrigées
  via un plancher de sécurité déclaré dans `pyproject.toml` (`[tool.uv]
  constraint-dependencies`) — sans les ajouter comme dépendances directes. Si un jour
  une CVE transitive **non corrigeable** apparaît, on l'exclut explicitement et de façon
  traçable (`pip-audit --ignore-vuln PYSEC-XXXX-XX`) plutôt que de re-désactiver le gate.
- **`lint` = `ruff check` seul** (pas `ruff format --check`) : le formatage automatique
  toucherait de nombreux fichiers et créerait des conflits avec les branches en cours.
  Le formatage reste disponible en local (`uvx ruff format`) et pourra devenir un gate
  une fois les branches convergées.

### 5.1 Configuration SonarCloud

Le job `sonarcloud` s'appuie sur [`sonar-project.properties`](../../sonar-project.properties)
(clé projet, organisation, sources, chemin de couverture). Pour l'activer :

1. Créer le projet sur [sonarcloud.io](https://sonarcloud.io) (import du dépôt GitHub) et
   vérifier `sonar.projectKey` / `sonar.organization` dans le fichier de properties.
2. Générer un **token** (My Account > Security) et l'ajouter comme variable
   **`SONAR_TOKEN`** dans CircleCI (Project Settings > Environment Variables).

Tant que `SONAR_TOKEN` n'est **pas** défini, le job **s'ignore proprement** (message +
sortie 0) : la CI ne casse pas avant que SonarCloud soit branché. La couverture affichée
provient du `coverage.xml` généré par `test-ia` (chemins relatifs via
`[tool.coverage.run] relative_files = true`) et transmis au job via un **workspace** CircleCI.

## 6. Ce qui reste « cible » (perspectives)

| Brique cible | Rôle | État |
|---|---|---|
| **Push registry** (Docker Hub / GHCR) | publier l'image taggée | la CD est **build-only** ; le push nécessite des credentials à configurer dans CircleCI |
| **Déploiement automatique VPS** | dérouler [`deploy.sh`](../../deploy.sh) via SSH sur tag | manuel aujourd'hui (cf. [`deploiement_vps.md`](../06-deploiement/deploiement_vps.md)) |
| **Test de charge** (Locust sur `/predict/batch`) | tenue en charge | hors périmètre actuel |

> Le message : l'**intégration continue** (gouvernance, sécurité, tests unitaires →
> intégration → E2E, lint, gate modèle) et l'**amorce de livraison** (build sur tag)
> sont en place et tournent ; l'industrialisation complète consiste à *fermer la boucle*
> avec le push registry puis le déploiement automatique.
