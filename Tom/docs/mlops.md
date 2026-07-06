# Automatisation de la chaîne (MLOps) MECHA / MSPR TPRE841

> Répond au point d'industrialisation du CDC : *« principes de déploiement et
> d'exploitation, perspectives d'industrialisation »* et *« appliquer l'intégration
> continue »*. Complète [`architecture.md`](architecture.md) (schéma global).

Ce document décrit comment la chaîne *données → modèle → service* est **automatisée** :
un ré-entraînement reproductible en une commande, un **garde-fou qualité** en intégration
continue, et une **simulation de flux temps réel** qui démontre le chemin d'exploitation.

## 1. Vue d'ensemble

```
                LOCAL (données présentes)                     CI (sans données)
   ┌─────────────────────────────────────────┐        ┌──────────────────────────┐
   │ ml/train.py                              │        │ ml/validate_metrics.py   │
   │  prep → train (2 têtes LSTM) → eval      │        │  lit models/metrics.json │
   │  → models/*.pt + scaler.joblib           │  git   │  vs ml/metrics_thresholds│
   │  → models/metrics.json  ────────────────────push─▶│  → échoue si régression  │
   └─────────────────────────────────────────┘        └──────────────────────────┘

   ┌─────────────────────────────────────────┐
   │ sim/producer.py  (backend en marche)     │   flux temps réel simulé :
   │  rejoue un CSV cycle par cycle           │   CSV ──tick──▶ /predict/batch
   │  → vue parc live + output/alertes_live   │        ▲ émule capteurs IoT/SCADA
   └─────────────────────────────────────────┘
```

## 2. Ré-entraînement reproductible  `ml/train.py`

Convertit le notebook `notebooks/03_lstm.py` en **commande unique et traçable** (les
notebooks restent la version pédagogique/exploratoire ; `train.py` est la version
opérationnelle).

```bash
uv run python -m ml.train                 # prep → entraîne → évalue → journalise
uv run python -m ml.train --eval-only     # recharge les artefacts, évalue seulement
uv run python -m ml.train --epochs 10 --seed 7
uv run python -m ml.train --quick         # smoke : 2 epochs (vérifie que le pipeline tourne)
```

Le pipeline reproduit **à l'identique** le prétraitement d'entraînement (`ml/prep.py` :
split par machine anti-fuite, `StandardScaler` ajusté sur le train, fenêtres glissantes
`SEQ_LEN=30`, clip RUL) puis entraîne les deux têtes (BCEWithLogits pondéré pour
`at_risk`, MSE pour `RUL`) avec early stopping.

**Sorties :** chaque ré-entraînement écrit un **run versionné** `models/runs/<run_id>/`
(2 têtes LSTM + `scaler.joblib` + `metrics.json`) et fait pointer le registre dessus —
il **n'écrase jamais** la baseline plate committée (cf. §2 bis).

> **Reproductibilité vérifiée** : `--eval-only` sur les artefacts committés redonne les
> chiffres du [`README.md`](README.md) (F1 0.885, recall 0.969, AUC 0.993, RMSE 26.3,
> R² 0.74), ce qui garantit que `train.py` et les notebooks décrivent bien le même modèle.

## 2 bis. Registre de modèles versionné `ml/registry.py`

Pour ne **jamais perdre le modèle de référence**, la sauvegarde est découplée en deux niveaux :

- **Baseline immuable** : `models/lstm_classifier.pt`, `models/lstm_regressor.pt`,
  `models/scaler.joblib`, `models/metrics.json` — les fichiers **plats, committés** dans
  git (~300 Ko). C'est le contrat de perf servi par le backend conteneurisé et validé en
  CI. `train.py` ne les touche **jamais**.
- **Runs versionnés** : `models/runs/<run_id>/` (un dossier daté par ré-entraînement) +
  le pointeur `models/registry.json` qui désigne le run **courant**. Ces deux-là sont
  **locaux** (ignorés par git) : ce sont les expériences, pas le contrat.

`resolve()` (consommé par le backend, l'éval et le gate) sert le **run courant s'il est
défini et présent, sinon la baseline** — donc sans registre (ex. en CI, ou image Docker),
tout retombe sur la baseline committée, comportement inchangé.

```bash
uv run python -m ml.train                 # entraîne -> runs/<id>/, promeut courant
uv run python -m ml.train --no-promote    # entraîne sans changer le modèle servi
uv run python -m ml.registry list         # runs + courant (→ marque le courant)
uv run python -m ml.registry promote <run_id>          # bascule le pointeur servi
uv run python -m ml.registry use-baseline              # resert la baseline plate
uv run python -m ml.registry promote <run_id> --to-baseline   # copie -> baseline (= prod)
```

**Mise en production explicite** : `--to-baseline` copie les artefacts du run sur les
fichiers plats committés ; c'est la seule opération qui modifie la baseline, elle est
donc **reviewable en git** (commit) avant d'atterrir dans l'image Docker.

## 3. Garde-fou qualité en CI `ml/validate_metrics.py`

Job CircleCI `validate-model` : compare `models/metrics.json` aux seuils de
`ml/metrics_thresholds.json` et **fait échouer le pipeline en cas de régression**.

| Métrique | Seuil | Justification |
|---|---|---|
| `classification.f1` | ≥ 0.82 | détection fiable des machines à risque |
| `classification.recall` | ≥ 0.90 | **priorité métier** : ne pas rater une machine en fin de vie |
| `classification.roc_auc` | ≥ 0.95 | pouvoir discriminant global |
| `regression.RMSE` | ≤ 30.0 | erreur RUL bornée (en cycles) |
| `regression.MAE` | ≤ 22.0 | erreur moyenne acceptable |
| `regression.R2` | ≥ 0.65 | variance expliquée |

Volontairement **sans dépendance** (stdlib seule) et **sans données** : la CI lit le
rapport committé. Workflow associé : *ré-entraîner en local (`train.py` régénère artefacts
+ `metrics.json` ensemble) → commit → le gate valide en CI*.

```bash
uv run python -m ml.validate_metrics       # code de sortie 1 si un seuil n'est pas tenu
```

## 4. Simulation de flux temps réel `sim/producer.py`

Émule la remontée continue des capteurs : rejoue un CSV **cycle par cycle** et, à chaque
tick, interroge `/predict/batch` comme le ferait une supervision temps réel. L'aval (API,
seuils, alertes) est **identique à ce qu'il serait en production** seul l'amont (le
connecteur IoT/SCADA) est remplacé par le rejeu du CSV.

```bash
# 1) backend en marche :   uv run uvicorn backend.api.main:app
# 2) flux simulé :
uv run python -m sim.producer --machines 8 --interval 1.0
uv run python -m sim.producer --machines 40 --interval 0.1 --usine "Usine A"
```

À chaque tick, une vue parc vivante s'affiche (compteurs `ok` / `warning` / `critical`,
machines critiques) et les **montées d'alerte** (`ok→warning→critical`) sont journalisées
dans `output/alertes_live.csv` (horodatage, tick, machine, usine, ligne, proba, RUL)
exactement le livrable qu'exploiterait l'équipe maintenance.

## 5. Ce qui reste « cible » (perspectives)

Cette chaîne automatise la boucle *entraînement → validation → service*. L'industrialisation
complète ajouterait, autour de briques déjà en place :

| Brique cible | Rôle | S'appuie sur l'existant |
|---|---|---|
| Ingestion (OPC-UA/MQTT → Kafka) | remplacer le rejeu CSV par un vrai flux | `sim/producer.py` (même contrat d'API) |
| Orchestration (Airflow/Prefect) | planifier le ré-entraînement | `ml/train.py` (déjà une commande) |
| Registry (MLflow) | historiser modèles + métriques hors git, UI de comparaison | **amorcé** : `ml/registry.py` (runs versionnés + pointeur courant, §2 bis) |
| Monitoring de dérive (Evidently) | déclencher un re-train sur dérive | seuils + jeu de référence |
| CI → CT | entraînement continu | gate `validate-model` déjà en CI |

> Le message : les briques **serving** (FastAPI) et **CI + gate qualité** existent et
> tournent ; l'industrialisation consiste à *fermer la boucle* avec ingestion,
> orchestration et monitoring de dérive.

## 6. Limites

- La CI **ne ré-entraîne pas** : les données (`assets/`) ne sont pas versionnées
  (confidentialité + volume). Le gate valide le rapport committé, il ne recalcule pas les
  métriques. Le ré-entraînement reste une action **locale** explicite.
- Le simulateur rejoue un jeu **fini** (C-MAPSS) : il démontre le chemin temps réel, il ne
  génère pas de dérive nouvelle.
