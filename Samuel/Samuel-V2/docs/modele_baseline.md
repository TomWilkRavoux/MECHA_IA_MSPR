# Modèle — Baseline linéaire (LinearRegression + LogisticRegression)

> Notebook : `02_Baseline_Lineaire.ipynb`
> Modèles : `models/baseline_{regression,classification}_usine{1,2}.joblib`

## Rôle

Le plancher de comparaison exigé par le CDC : si un modèle avancé ne bat pas une droite,
sa complexité n'est pas justifiée. Totalement interprétable (un coefficient par capteur),
zéro hyperparamètre critique.

## Construction

| Tâche | Modèle | Réglages | Justification |
|---|---|---|---|
| Régression RUL | `LinearRegression` | défauts | référence minimale absolue |
| Classification `at_risk` | `LogisticRegression` | `max_iter=1000`, `class_weight='balanced'` | ~14 % de positifs → sans rééquilibrage le modèle dirait toujours « normal » |

Entrées : features du notebook 01, MinMax [0,1] (indispensable pour un linéaire),
normalisation par régime en Usine 2. Prédictions RUL bornées à [0, 125].

## Résultats (test officiel, prédiction au dernier cycle observé)

**Régression RUL**

| Usine | RMSE | MAE | R² | NASA moyen/machine |
|---|---|---|---|---|
| 1 (200 machines) | 22.55 | 18.13 | 0.70 | 16.7 |
| 2 (507 machines) | 33.94 | 26.39 | 0.61 | 64.3 |

**Classification `at_risk`**

| Usine | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.945 | 0.840 | 0.933 | 0.884 | 0.986 |
| 2 | 0.909 | 0.727 | 0.956 | 0.826 | 0.981 |

## Lecture

- La classification linéaire est déjà étonnamment bonne **grâce à la préparation** :
  sur l'Usine 1 (régime unique), F1 = 0.88. La normalisation par régime porte aussi la
  logistique de l'Usine 2 à un niveau honnête (F1 = 0.83). Autrement dit : une grosse part
  de la performance vient des **données bien préparées**, pas de la sophistication du modèle.
- En régression, la limite linéaire est nette (RMSE 34 cycles en Usine 2, score NASA élevé) :
  la dégradation n'est pas une droite. C'est la marge que les arbres et le LSTM comblent.
