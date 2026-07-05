# Modèle — HistGradientBoosting (boosting de gradient)

> Notebook : `04_HGB.ipynb`
> Modèles : `models/hgb_{regression,classification}_usine{1,2}.joblib`

## Rôle et pourquoi ce choix

Deuxième famille d'ensembles d'arbres, philosophie opposée à la forêt : les arbres sont
construits **en séquence**, chacun corrige les erreurs résiduelles des précédents (boosting).
C'est l'état de l'art tabulaire (boosting de gradient sur histogrammes, comme LightGBM), mais intégrée à sklearn : pas de
dépendance en plus, et déjà multi-cœurs.

## Construction

| Réglage | Valeur | Justification |
|---|---|---|
| `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10` | — | le nombre d'arbres est choisi automatiquement (68 à 233 selon la tâche) au lieu d'être deviné |
| `max_iter=500` | plafond large | c'est l'early stopping qui décide |
| `class_weight='balanced'` (classif) | ratio ≈ 6.3 | déséquilibre ~14 % de positifs |
| `random_state=42` | — | reproductibilité |

**Limite assumée et documentée** : le découpage interne de l'early stopping sklearn se fait
par lignes (pas par machine) → son score interne est optimiste (fuite). Il ne sert qu'à
arrêter l'empilement d'arbres ; toutes les performances annoncées viennent de la validation
par machine et du test officiel.

Comme pour la forêt, le `sample_weight` zone à risque a été testé sur la validation et
**écarté** (RMSE val 16.43 avec poids contre 15.89 sans).

## Résultats (test officiel)

**Régression RUL**

| Usine | RMSE | MAE | R² | NASA moyen/machine |
|---|---|---|---|---|
| 1 | 18.62 | 13.99 | 0.80 | 7.1 |
| 2 | 28.29 | 20.42 | 0.73 | 29.8 |

**Classification `at_risk`**

| Usine | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.935 | 0.833 | 0.889 | 0.860 | 0.986 |
| 2 | 0.925 | 0.764 | 0.965 | 0.853 | 0.990 |

## Lecture et limites

- **Meilleur modèle tabulaire en régression** sur les deux usines (RMSE 18.6 / 28.3),
  conforme à la théorie : le boosting affine là où le bagging moyenne.
- **Recall 0.965 en Usine 2** : quasiment aucune machine à risque manquée sur le parc
  complexe — précieux pour éviter les pannes non planifiées.
- **Limites** : pas d'importance de variables native (il faudrait une permutation
  importance) ; cycles traités isolément, comme tous les tabulaires.
