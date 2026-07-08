# Documentation technique - Partie IA (MECHA / MSPR TPRE841)

Cette documentation explique **les données** et **les modèles** de la solution de
maintenance prédictive. Elle complète `../plan.md` (feuille de route) et les notebooks
(`../notebooks/`).

## Sommaire

| Document | Contenu |
|---|---|
| [`architecture.md`](01-architecture/architecture.md) | **Schéma d'architecture global** (sources → centralisation → préparation/modélisation → service IA → exploitation), distinction **réel/cible**, flux temps réel/batch, **justification des choix** et limites d'industrialisation. |
| [`mlops.md`](04-mlops/mlops.md) | **Automatisation de la chaîne** : ré-entraînement reproductible (`ml/train.py`), **gate qualité en CI** (`ml/validate_metrics.py`), **simulation de flux temps réel** (`sim/producer.py`), et perspectives d'industrialisation. |
| [`ci-cd.md`](07-ci-cd/ci-cd.md) | **Intégration & livraison continues** (CircleCI) : les deux workflows (`ci` / `cd`), le détail des jobs, l'**orchestration** (dépendances, parallélisme), la pyramide de tests, la gestion des dépendances (`uv` + extras CPU/GPU) et le **déclenchement / reproduction en local**. |
| [`repartition_donnees.md`](02-donnees/repartition_donnees.md) | Origine du dataset (NASA C-MAPSS), **justification de la répartition** en usines / lignes (`analyseGlobal.py`), et **consolidation** au vocabulaire MECHA (`fusionDatasetFinal.py`). |
| [`dictionnaire_donnees.md`](02-donnees/dictionnaire_donnees.md) | **Dictionnaire de données par colonne** (identité, réglages, 21 capteurs C-MAPSS, cibles) : type, unité, description, domaine. Cohérence vérifiée par test. |
| [`modele_baseline.md`](03-modeles/modele_baseline.md) | Baseline linéaire (régression logistique + linéaire) - plancher de performance exigé par le CDC. |
| [`modele_random_forest.md`](03-modeles/modele_random_forest.md) | Random Forest - cœur de la solution (arbres, importances). |
| [`modele_xgboost.md`](03-modeles/modele_xgboost.md) | Gradient Boosting (XGBoost) - état de l'art tabulaire. |
| [`modele_lstm.md`](03-modeles/modele_lstm.md) | LSTM (Deep Learning, PyTorch/GPU) - exploitation de la temporalité. |

### Exploitation applicative - dashboard

| Document | Contenu |
|---|---|
| [`dashboard_technique.md`](05-exploitation/dashboard_technique.md) | Architecture **frontend ↔ API découplée**, endpoints consommés, fenêtre glissante / calcul de trajectoire, structure du code, lancement, tests. |
| [`dashboard_fonctionnel.md`](05-exploitation/dashboard_fonctionnel.md) | **Guide utilisateur métier** : écrans, lecture des indicateurs, **justification du seuil 30**, règles d'alerte, limites d'usage. |

## Rappel des deux tâches

- **Classification `at_risk`** : état machine *normal* (0) / *à risque* (1), avec
  `at_risk = 1 si RUL ≤ 30 cycles`.
- **Régression `RUL`** : temps restant avant défaillance (Remaining Useful Life).

## Synthèse comparative (jeu de test officiel)

**Classification `at_risk`** (plus haut = mieux) :

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (LogReg) | 0.911 | 0.750 | 0.906 | 0.821 | 0.976 |
| Random Forest | 0.934 | 0.804 | 0.931 | 0.863 | 0.986 |
| XGBoost | 0.926 | 0.774 | 0.950 | 0.853 | 0.986 |
| **LSTM** | **0.943** | **0.815** | **0.969** | **0.885** | **0.993** |

**Régression `RUL`** (RMSE/MAE/NASA : plus bas = mieux ; R² : plus haut = mieux) :

| Modèle | RMSE | MAE | R² | Score NASA |
|---|---|---|---|---|
| Baseline (LinReg) | 31.46 | 24.42 | 0.62 | 39 431 |
| Random Forest | 27.30 | 19.64 | 0.71 | 22 570 |
| XGBoost | 27.16 | 19.48 | 0.72 | **22 385** |
| **LSTM** | **26.28** | **18.52** | **0.74** | 23 802 |

Le **LSTM** est le meilleur sur les deux tâches ; XGBoost/RF conservent un léger avantage
sur le **score NASA** (moins de prédictions en retard). Détails et justifications dans
chaque document dédié.
