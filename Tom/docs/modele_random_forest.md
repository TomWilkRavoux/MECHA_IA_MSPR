# Modèle — Random Forest (cœur de la solution)

> Notebook d'entraînement : `notebooks/01_random_forest.py`
> Notebooks d'évaluation : `notebooks/analyse/eval_rf_classifier.py`,
> `notebooks/analyse/eval_rf_regressor.py`
> Modèles sauvegardés : `models/rf_classifier.joblib`, `models/rf_regressor.joblib`

## 1. Rôle & pourquoi ce choix

Le CDC désigne explicitement les **arbres de décision et ensembles d'arbres (Random Forest)**
comme **cœur de la solution**. C'est un choix pertinent pour la maintenance prédictive :

- **Robuste** au bruit, aux valeurs manquantes et aux échelles hétérogènes des capteurs
  (aucune normalisation strictement nécessaire) ;
- capture les **non-linéarités et interactions** entre capteurs, contrairement à la baseline ;
- **interprétable** via l'**importance des variables** : on identifie les capteurs qui
  pilotent la prédiction → information directement utile aux équipes de maintenance ;
- peu de risque de sur-apprentissage grâce au **bagging** (moyenne de nombreux arbres
  décorrélés).

La Random Forest sert de **référence** à laquelle XGBoost et le LSTM sont comparés.

## 2. Données en entrée

Via `ml/prep.py` : 24 features, **split par machine** (80/20), RUL borné à 125.
La normalisation `StandardScaler` est appliquée par cohérence avec les autres modèles, mais
n'est pas critique pour un modèle à base d'arbres.

## 3. Construction technique

Deux forêts distinctes, une par tâche :

| Tâche | Modèle | Hyperparamètres | Justification |
|---|---|---|---|
| Classification `at_risk` | `RandomForestClassifier` | `n_estimators=200`, `min_samples_leaf=5`, `max_depth=None`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42` | 200 arbres = bon compromis variance/temps ; `min_samples_leaf=5` limite le sur-apprentissage ; `balanced` gère le déséquilibre (~14 % de positifs). |
| Régression `RUL` | `RandomForestRegressor` | `n_estimators=200`, `min_samples_leaf=5`, `max_depth=None`, `n_jobs=-1`, `random_state=42` | mêmes réglages ; profondeur libre pour capter la dynamique de dégradation. |

- **`random_state=42`** : reproductibilité (exigée pour la soutenance).
- **`n_jobs=-1`** : entraînement parallélisé sur tous les cœurs.
- **Importance des variables** exportée en graphe (`reports/figures/rf_importance.png`).

## 4. Métriques obtenues (jeu de test officiel)

**Classification `at_risk`**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.934 | 0.804 | 0.931 | 0.863 | 0.986 |

**Régression `RUL`**

| RMSE | MAE | R² | Score NASA |
|---|---|---|---|
| 27.30 | 19.64 | 0.71 | 22 570 |

## 5. Lecture & limites

- **Net gain sur la baseline** : F1 0.82 → 0.86, RUL RMSE 31.5 → 27.3, score NASA divisé
  par ~1.7. Les interactions entre capteurs paient.
- **Recall = 0.93** : la grande majorité des machines à risque sont détectées — critère
  clé pour éviter les pannes non planifiées.
- **Limites** : modèle **lourd en mémoire** (le régresseur fait ~295 Mo) car il stocke
  200 arbres profonds ; il traite chaque cycle **isolément** et n'exploite pas la
  **séquence temporelle** — c'est ce que le LSTM viendra chercher.
