# Modèle - Baseline linéaire (régression logistique + linéaire)

> Notebook d'entraînement : `notebooks/02_gradient_boosting_baseline.py`
> Notebooks d'évaluation : `notebooks/analyse/eval_baseline_logreg.py`,
> `notebooks/analyse/eval_baseline_linreg.py`
> Modèles sauvegardés : `models/baseline_logreg.joblib`, `models/baseline_linreg.joblib`

## 1. Rôle & pourquoi ce choix

Le CDC exige une **régression simple comme baseline de comparaison**. Ce n'est pas le
modèle « performant » visé, mais un **plancher de référence** indispensable :

- il **quantifie l'apport** des modèles avancés (RF, XGBoost, LSTM) - l'écart de métriques
  mesure exactement ce que gagnent les non-linéarités ;
- il est **totalement interprétable** : chaque capteur a un coefficient de signe et de poids
  lisibles, facile à expliquer au métier ;
- il est **rapide** et robuste, sans hyperparamètre critique.

Sa limite est assumée : un modèle linéaire suppose une relation linéaire entre capteurs et
cible, ce qui est faux pour une dégradation mécanique (effets de seuil, interactions).

## 2. Données en entrée

Communes à tous les modèles tabulaires (module `ml/prep.py`) :

- **24 features** : 3 réglages + 21 capteurs.
- **Split par machine** (80/20) pour éviter la fuite entre train et validation.
- **Normalisation `StandardScaler`** (ajustée sur le train) - **indispensable** ici : un
  modèle linéaire est sensible à l'échelle des variables.
- Cible RUL **bornée à 125 cycles** (clipping piecewise).

## 3. Construction technique

| Tâche | Modèle | Hyperparamètres | Justification |
|---|---|---|---|
| Classification `at_risk` | `LogisticRegression` | `max_iter=1000`, `class_weight="balanced"` | `balanced` compense le déséquilibre (~14 % de positifs) en repondérant la classe rare. |
| Régression `RUL` | `LinearRegression` | par défaut (moindres carrés ordinaires) | modèle le plus simple possible → vraie référence minimale. |

Aucune régularisation forte n'est ajoutée : l'objectif est la **simplicité maximale**, pas
l'optimisation.

## 4. Métriques obtenues (jeu de test officiel)

**Classification `at_risk`**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.911 | 0.750 | 0.906 | 0.821 | 0.976 |

**Régression `RUL`**

| RMSE | MAE | R² | Score NASA |
|---|---|---|---|
| 31.46 | 24.42 | 0.62 | 39 431 |

## 5. Lecture & limites

- La classification linéaire est déjà **honnête** (AUC 0.976) : le signal de dégradation est
  en partie linéairement séparable. Mais precision 0.75 = trop de fausses alertes.
- En **RUL**, R² = 0.62 et RMSE = 31 cycles : erreur importante, et **score NASA le plus
  élevé** (le pire) du panel → beaucoup de prédictions en retard.
- **Conclusion** : la baseline valide qu'il y a du signal exploitable, mais laisse une
  marge nette que les modèles à base d'arbres et le LSTM viennent combler.
