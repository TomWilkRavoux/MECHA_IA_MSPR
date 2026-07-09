# MECHA - Partie IA (MSPR TPRE841)

Maintenance prédictive sur dataset **NASA C-MAPSS** (reconditionné au vocabulaire MECHA).
Deux tâches : **classification `at_risk`** (RUL ≤ 30 cycles) et **régression `RUL`**.

Documentation métier détaillée (données + modèles) : [`docs/README.md`](docs/README.md).

---

## 1. Prérequis

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)** (gestionnaire d'environnement/paquets)
- **GPU NVIDIA optionnel** : `torch` est décliné en deux extras mutuellement exclusifs,
  `cu128` (CUDA, développement sur GPU) et `cpu` (CI / serving). Choisir l'un des deux
  au `uv sync`.

```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Depuis la racine : créer le venv et installer les dépendances
uv sync --extra cu128   # poste de dev avec GPU NVIDIA
# ou
uv sync --extra cpu     # sans GPU (CI, serving) : torch CPU, pas de pile CUDA
```

---

## 2. Dossiers à créer (non commités)

Le `.gitignore` exclut les données, modèles et rapports. Après un `git clone`,
il faut **recréer ces dossiers et y déposer les données** avant de lancer quoi que ce soit :

| Dossier | Statut | Contenu | Créé par |
|---|---|---|---|
| `assets/` | **à fournir** | données brutes + jeux d'entraînement/test | **manuel** |
| `models/` | auto | modèles entraînés (`*.joblib`, `*.pt`) | `ml/prep.py` au run |
| `reports/figures/` | auto | figures d'évaluation (`*.png`) | `ml/prep.py` au run |

`models/` et `reports/figures/` sont créés automatiquement au premier import de `ml.prep`
(voir `MODELS_DIR.mkdir(...)` / `FIG_DIR.mkdir(...)`). **Seul `assets/` est à préparer à la main.**

### Contenu minimal de `assets/` pour entraîner les modèles

Les notebooks lisent leurs données dans `assets/KaggleDataset/` via `ml/prep.py` :

```
assets/KaggleDataset/
├── mecha_train_classification.csv   # entraînement - contient at_risk + RUL
├── mecha_train_rul.csv              # entraînement régression (cible RUL)
├── mecha_test_classification.csv    # test (features seules, machines tronquées)
└── mecha_rul_true.csv               # vérité terrain : RUL au dernier cycle test
```

Ces 4 fichiers suffisent pour lancer les 3 notebooks de modélisation.

> Les CSV bruts C-MAPSS (`CMaps/`, `CMapsCsv/`) et les données MECHA
> (`drive-download-.../`, `output/`) ne sont nécessaires que pour rejouer la
> **pré-analyse** et la **fusion** (`preAnalyse/`). Récupérer l'archive du projet
> (Drive / Kaggle) et décompresser dans `assets/`.

---

## 3. Lancer le projet

Toutes les commandes se lancent **depuis la racine** (les chemins de `ml/prep.py` sont
relatifs à ce dossier).

### Notebooks de modélisation (`notebooks/`)

Chaque notebook entraîne, sauvegarde les modèles dans `models/` et les figures dans
`reports/figures/`.

```bash
# 1. Random Forest (classification + régression)
uv run marimo edit notebooks/01_random_forest.py

# 2. Baseline linéaire + XGBoost
uv run marimo edit notebooks/02_gradient_boosting_baseline.py

# 3. LSTM (PyTorch, GPU si dispo)
uv run marimo edit notebooks/03_lstm.py
```

> `marimo edit` ouvre le notebook interactif dans le navigateur.
> Pour une exécution non interactive : `uv run marimo run notebooks/01_random_forest.py`.

### Notebooks d'évaluation (`notebooks/analyse/`)

À lancer **après** avoir entraîné les modèles (ils rechargent les `*.joblib` / `*.pt`
de `models/`) :

```bash
uv run marimo edit notebooks/analyse/eval_rf_classifier.py
# ... idem pour eval_xgb_*, eval_baseline_*, eval_lstm_*
```

### Pré-analyse & fusion des données (`preAnalyse/`)

Optionnel - reproduit l'exploration et la construction des jeux MECHA à partir des
CSV bruts :

```bash
uv run marimo edit preAnalyse/analyseGlobal.py       # analyse exploratoire
uv run marimo edit preAnalyse/fusionDatasetFinal.py  # consolidation MECHA
```

---

## 4. Exposition du modèle - API REST (LSTM)

Le modèle **LSTM** (retenu comme meilleur, cf. `docs/README.md`) est exposé via une
**API REST FastAPI** pour l'exploitation métier (maintenance / supervision), conformément
au CDC §5. Elle sert les deux têtes du modèle : **classification `at_risk`** et
**régression `RUL`**, enrichies d'un **niveau d'alerte** (`ok` / `warning` / `critical`).

> Prérequis : modèles entraînés présents dans `models/` (`scaler.joblib`,
> `lstm_classifier.pt`, `lstm_regressor.pt`) - sinon lancer d'abord le notebook `03_lstm`.

```bash
# Depuis la racine - lance le serveur (http://localhost:8000)
uv run uvicorn backend.api.main:app --reload
```

- **Doc interactive (Swagger)** : http://localhost:8000/docs
- **Endpoints** :
  - `GET /health` - état du service + modèles chargés (sonde Docker/CI)
  - `GET /features` - liste ordonnée des 24 variables attendues par cycle
  - `POST /predict` - prédiction pour **une** machine (série de cycles)
  - `POST /predict/batch` - prédiction pour un **lot** de machines

### Exemple d'appel

Le corps envoie les cycles ordonnés d'une machine (valeurs **brutes** : la
normalisation est appliquée côté serveur). Le modèle utilise les 30 derniers cycles
(left-padding si moins).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "machine_id": "M001",
        "cycles": [
          {"values": {"setting_1": -0.7, "setting_2": 0.0, "setting_3": 100.0,
                      "T2": 518.67, "T24": 643.0, "...": 0.0, "W32": 23.4}}
        ]
      }'
```

Réponse :

```json
{
  "machine_id": "M001",
  "at_risk": true,
  "risk_probability": 0.97,
  "rul_predicted": 19.45,
  "alert_level": "critical",
  "threshold": 30,
  "n_cycles_used": 30
}
```

**Règle d'alerte** (paramétrable dans `backend/api/inference.py`) : `at_risk` provient de la
tête classification (proba ≥ 0.5, cohérent avec l'évaluation) ; l'alerte est `critical`
si `RUL ≤ 15` **ou** `proba ≥ 0.75`, `warning` si `at_risk` sans être critique, sinon `ok`.

---

## 5. Interface métier - Dashboard (Streamlit)

Dashboard de supervision qui **consomme l'API** (aucun modèle chargé côté frontend) :
priorisation des interventions par RUL, niveaux d'alerte, KPI parc et fiche machine
(CDC §5 : tableaux de bord, alertes, indicateurs).

> Prérequis : le backend (§4) doit tourner. L'URL de l'API est configurable dans la
> barre latérale (défaut `http://localhost:8000`, surchargée par `MECHA_API_URL`).

```bash
# Depuis la racine, dans un 2e terminal (backend déjà lancé)
uv run streamlit run frontend/dashboard.py
```

Ouvre http://localhost:8501. Source de données : **jeu de démonstration** (le jeu de
test C-MAPSS de `assets/`) ou **import CSV** (colonnes `machine_id`, `cycle` + variables
capteurs). Cliquer sur **Analyser le parc** pour lancer les prédictions.

Contenu : KPI (machines critiques / à surveiller / normales, RUL min), graphique de
priorisation (machines triées par RUL croissant, code couleur d'alerte), tableau du
parc exportable en CSV, et fiche machine (RUL, probabilité, tendance capteurs).

---

## 6. Tests

```bash
uv run pytest        # tests d'intégration de l'API (tests/)
```

Les tests d'inférence sont automatiquement ignorés si `models/` ne contient pas les
artefacts LSTM ; la validation du contrat d'entrée, elle, tourne sans les modèles.

---

## 7. Chaîne MLOps (entraînement automatisé, gate, flux temps réel)

Détails et justifications : [`docs/mlops.md`](docs/04-mlops/mlops.md).

### Ré-entraînement reproductible

Version opérationnelle du notebook `03_lstm` : prépare, entraîne les 2 têtes, évalue et
**journalise les métriques** dans `models/metrics.json`.

```bash
uv run python -m ml.train                 # entraîne → évalue → journalise
uv run python -m ml.train --eval-only     # recharge les artefacts, évalue seulement
uv run python -m ml.train --quick         # smoke : 2 epochs (vérifie le pipeline)
```

### Gate qualité (CI)

Refuse un modèle dont les performances régressent sous `ml/metrics_thresholds.json`
(job CircleCI `validate-model`). Sans dépendance ni données : lit le rapport committé.

```bash
uv run python -m ml.validate_metrics      # exit 1 si un seuil n'est pas tenu
```

### Simulation de flux temps réel

Rejoue un CSV cycle par cycle vers l'API (émule les capteurs). Vue parc vivante +
journal `output/alertes_live.csv`. **Prérequis : backend lancé (§4).**

```bash
uv run python -m sim.producer --machines 8 --interval 1.0
```

---

## 8. Structure du projet

```
MECHA_IA_MSPR/          # racine du dépôt
├── ml/                 # cœur IA partagé (prep, métriques, modèle LSTM)
│   ├── train.py        #   entraînement reproductible → models/ + metrics.json
│   └── validate_metrics.py  # gate qualité (seuils metrics_thresholds.json)
├── backend/            # application backend
│   └── api/            # API REST FastAPI (exposition du LSTM)
├── frontend/           # dashboard Streamlit (consomme l'API)
├── sim/                # simulation de flux temps réel (producer.py → API)
├── tests/              # tests unitaires + intégration API + MLOps (pytest)
├── notebooks/          # 3 notebooks de modélisation + analyse/ (évaluation)
├── preAnalyse/         # exploration & fusion des données brutes
├── docs/               # documentation technique (données + modèles)
├── assets/             # données à fournir (non commité)
├── models/             # modèles entraînés (auto, non commité)
├── reports/figures/    # figures d'évaluation (auto, non commité)
├── pyproject.toml      # dépendances (uv)
└── uv.lock
```
