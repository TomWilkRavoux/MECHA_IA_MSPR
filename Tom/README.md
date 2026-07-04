# MECHA — Partie IA (MSPR TPRE841)

Maintenance prédictive sur dataset **NASA C-MAPSS** (reconditionné au vocabulaire MECHA).
Deux tâches : **classification `at_risk`** (RUL ≤ 30 cycles) et **régression `RUL`**.

Documentation métier détaillée (données + modèles) : [`docs/README.md`](docs/README.md).

---

## 1. Prérequis

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)** (gestionnaire d'environnement/paquets)
- **GPU NVIDIA optionnel** : le LSTM tourne sur CUDA si dispo (index PyTorch `cu128`
  déclaré dans `pyproject.toml`), sinon fallback CPU.

```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Depuis Tom/ : créer le venv et installer les dépendances
uv sync
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
Tom/assets/KaggleDataset/
├── mecha_train_classification.csv   # entraînement — contient at_risk + RUL
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

Toutes les commandes se lancent **depuis `Tom/`** (les chemins de `ml/prep.py` sont
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

Optionnel — reproduit l'exploration et la construction des jeux MECHA à partir des
CSV bruts :

```bash
uv run marimo edit preAnalyse/analyseGlobal.py       # analyse exploratoire
uv run marimo edit preAnalyse/fusionDatasetFinal.py  # consolidation MECHA
```

---

## 4. Structure du projet

```
Tom/
├── ml/                 # socle partagé (prep, métriques, modèle LSTM)
├── notebooks/          # 3 notebooks de modélisation + analyse/ (évaluation)
├── preAnalyse/         # exploration & fusion des données brutes
├── docs/               # documentation technique (données + modèles)
├── assets/             # données à fournir (non commité)
├── models/             # modèles entraînés (auto, non commité)
├── reports/figures/    # figures d'évaluation (auto, non commité)
├── pyproject.toml      # dépendances (uv)
└── uv.lock
```
