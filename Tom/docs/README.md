# Documentation technique — Partie IA (MECHA / MSPR TPRE841)

Cette documentation explique **les données** et **les modèles** de la solution de
maintenance prédictive. Elle complète `../plan.md` (feuille de route) et les notebooks
(`../notebooks/`).

## Sommaire

| Document | Contenu |
|---|---|
| [`repartition_donnees.md`](repartition_donnees.md) | Origine du dataset (NASA C-MAPSS), **justification de la répartition** en usines / lignes (`analyseGlobal.py`), et **consolidation** au vocabulaire MECHA (`fusionDatasetFinal.py`). |
| [`modele_baseline.md`](modele_baseline.md) | Baseline linéaire (régression logistique + linéaire) — plancher de performance exigé par le CDC. |
| [`modele_random_forest.md`](modele_random_forest.md) | Random Forest — cœur de la solution (arbres, importances). |
| [`modele_xgboost.md`](modele_xgboost.md) | Gradient Boosting (XGBoost) — état de l'art tabulaire. |
| [`modele_lstm.md`](modele_lstm.md) | LSTM (Deep Learning, PyTorch/GPU) — exploitation de la temporalité. |

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
