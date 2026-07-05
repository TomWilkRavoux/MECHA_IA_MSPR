# Documentation — Partie IA Samuel-V2 (MECHA / MSPR TPRE841)

## Sommaire

| Document | Contenu |
|---|---|
| [`repartition_usines.md`](repartition_usines.md) | Ma répartition Option B (usine = régime de fonctionnement), sa justification et ses conséquences techniques |
| [`modele_baseline.md`](modele_baseline.md) | Baseline linéaire — le plancher exigé par le CDC |
| [`modele_random_forest.md`](modele_random_forest.md) | Random Forest — cœur du CDC, importances des capteurs |
| [`modele_hgb.md`](modele_hgb.md) | HistGradientBoosting — l'état de l'art tabulaire, en sklearn |
| [`modele_lstm.md`](modele_lstm.md) | LSTM PyTorch — le seul modèle qui voit la tendance |

## Rappel des deux tâches

- **Classification `at_risk`** : machine dans ses 30 derniers cycles ? (`at_risk = 1 si RUL ≤ 30`)
- **Régression `RUL`** : cycles restants avant défaillance (plafonné à 125 à l'entraînement,
  plafond justifié par les données — notebook 01).

## Synthèse (jeu de test officiel, 707 machines, prédictions U1+U2 recombinées)

**Classification `at_risk`** (plus haut = mieux) :

| Modèle            | Accuracy  | Precision | Recall    | F1        | ROC-AUC   |
| ----------------- | --------- | --------- | --------- | --------- | --------- |
| Baseline (LogReg) | 0.919     | 0.755     | 0.950     | 0.841     | 0.982     |
| Random Forest     | 0.939     | **0.841** | 0.899     | 0.869     | 0.987     |
| HGB (boosting)    | 0.928     | 0.781     | 0.943     | 0.855     | 0.988     |
| **LSTM**          | **0.950** | 0.830     | **0.981** | **0.899** | **0.995** |

**Régression `RUL`** (RMSE/MAE/NASA : plus bas = mieux ; R² : plus haut = mieux) :

| Modèle | RMSE | MAE | R² | Score NASA |
|---|---|---|---|---|
| Baseline (LinReg) | 31.14 | 24.06 | 0.63 | 35 965 |
| Random Forest | 26.27 | 18.92 | 0.74 | 17 684 |
| HGB (boosting) | 25.92 | 18.60 | 0.74 | 16 540 |
| **LSTM** | **25.73** | **18.51** | **0.75** | **15 305** |

Détail par usine dans chaque document (l'Usine 1, parc simple, est bien mieux prédite :
RMSE 17.1 contre 28.4 en Usine 2 — c'est un argument de l'approche par usine : on **sait**
où le modèle est fiable et où il l'est moins).

## Recommandation de déploiement

| Besoin | Modèle | Pourquoi |
|---|---|---|
| Alerter (classification) | **LSTM** | recall 0.981 : ~2 machines à risque manquées sur 100 |
| Planifier (régression RUL) | **LSTM**, repli **HGB** | meilleure RMSE **et** meilleur score NASA ; HGB presque au niveau pour une fraction du coût |
| Parc simple (Usine 1) | HGB peut suffire | sur un problème homogène, le tabulaire est quasi au niveau du LSTM |

Garde-fou de production : suivre le **PHM08 « dangereux »** (ma décomposition du score NASA,
notebooks 02-06) — c'est lui qui compte les prédictions trop optimistes, celles qui mènent
à la panne non planifiée.
