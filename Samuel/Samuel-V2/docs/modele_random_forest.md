# Modèle — Random Forest (cœur du CDC)

> Notebook : `03_Random_Forest.ipynb`
> Modèles : `models/rf_{regression,classification}_usine{1,2}.joblib`
> Figure : `figures/rf_importances.png`

## Rôle et pourquoi ce choix

Le modèle central désigné par le CDC. Bien adapté à la maintenance prédictive :
non-linéarités et interactions entre capteurs, robustesse au bruit (bagging : des centaines
d'arbres indépendants votent), et **importance des variables** → on sait dire aux équipes
de maintenance quels capteurs surveiller.

## Construction — choix faits sur la validation par machine

Quatre configurations comparées **sur la validation** (jamais sur le test) pour la
régression Usine 1 :

| Config | RMSE val | NASA moyen val |
|---|---|---|
| V1 : 100 arbres | 16.02 | 1380 |
| V1 : 100 arbres + poids risque | 16.11 | 1377 |
| **V2 : 200 arbres, feuilles ≥ 5** | **15.90** | **1344** |
| V2 : 200 arbres + poids risque | 16.05 | 1370 |

- **Retenue : 200 arbres, `min_samples_leaf=5`, sans poids.** Le `min_samples_leaf=5`
  (une feuille = au moins 5 cycles) limite le sur-apprentissage du bruit.
- **Résultat honnête sur mon idée V1** : le `sample_weight` sur la zone à risque n'améliore
  pas la RMSE ici. Je le garde documenté comme piste écartée par la mesure.
- Classification : `class_weight='balanced'` (équivalent intégré du rééquilibrage, ratio ≈ 6.3).
- `random_state=42`, `n_jobs=-1`. Bug V1 corrigé : chaque usine a **son** scaler.

## Résultats (test officiel)

**Régression RUL**

| Usine | RMSE | MAE | R² | NASA moyen/machine |
|---|---|---|---|---|
| 1 | 18.90 | 14.25 | 0.79 | 6.9 |
| 2 | 28.67 | 20.76 | 0.72 | 32.2 |

**Classification `at_risk`**

| Usine | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.935 | 0.881 | 0.822 | 0.851 | 0.989 |
| 2 | 0.941 | 0.828 | 0.930 | 0.876 | 0.987 |

## Lecture et limites

- Gain massif sur la baseline en régression : RMSE 22.6 → 18.9 (U1) et 33.9 → 28.7 (U2),
  score NASA divisé par ~2 : les interactions entre capteurs paient.
- En classification U1, la forêt fait un F1 légèrement inférieur à la baseline (0.851 vs
  0.884) avec un profil différent : meilleure precision, recall plus bas. Rappel utile :
  sur un problème simple et bien préparé, le modèle simple est dur à battre.
- **Limites** : centaines d'arbres en mémoire, et chaque cycle est traité **isolément** —
  la tendance temporelle est perdue (→ LSTM, notebook 05).
