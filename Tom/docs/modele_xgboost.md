# Modèle — Gradient Boosting (XGBoost)

> Notebook d'entraînement : `notebooks/02_gradient_boosting_baseline.py`
> Notebooks d'évaluation : `notebooks/analyse/eval_xgb_classifier.py`,
> `notebooks/analyse/eval_xgb_regressor.py`
> Modèles sauvegardés : `models/xgb_classifier.joblib`, `models/xgb_regressor.joblib`

## 1. Rôle & pourquoi ce choix

Le CDC autorise d'aller au-delà de la Random Forest avec d'autres approches de ML. XGBoost
est **l'état de l'art sur données tabulaires** :

- il construit les arbres **séquentiellement** (boosting) : chaque arbre corrige les
  **erreurs résiduelles** des précédents, là où la Random Forest les moyenne (bagging) ;
- très **efficace** et régularisé (`subsample`, `colsample_bytree`, profondeur bornée) ;
- gère nativement le **déséquilibre** via `scale_pos_weight` ;
- souvent **meilleur que la Random Forest** à volume et bruit équivalents.

Il constitue le second modèle de ML, comparé à la RF (notebook 01) et au LSTM (notebook 03).

## 2. Données en entrée

Identiques aux autres modèles tabulaires (`ml/prep.py`) : 24 features, **split par
machine** (80/20 — la validation sert aussi à l'**early stopping**), normalisation
`StandardScaler`, RUL borné à 125.

## 3. Construction technique

| Tâche | Modèle | Hyperparamètres | Justification |
|---|---|---|---|
| Classification `at_risk` | `XGBClassifier` | `n_estimators=600`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight≈6.3`, `eval_metric="auc"`, `early_stopping_rounds=30` | LR faible + beaucoup d'arbres = apprentissage fin ; `subsample`/`colsample` régularisent ; `scale_pos_weight` = ratio négatifs/positifs pour le déséquilibre. |
| Régression `RUL` | `XGBRegressor` | idem, `eval_metric="rmse"` | mêmes principes ; l'early stopping arrête l'ajout d'arbres quand la RMSE de validation stagne. |

- **Early stopping** (`early_stopping_rounds=30`) : le nombre réel d'arbres retenus est
  **choisi automatiquement** sur la validation → évite le sur-apprentissage et fixe le
  budget de calcul.
- **`random_state=42`** : reproductibilité.

## 4. Métriques obtenues (jeu de test officiel)

**Classification `at_risk`**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.926 | 0.774 | 0.950 | 0.853 | 0.986 |

**Régression `RUL`**

| RMSE | MAE | R² | Score NASA |
|---|---|---|---|
| 27.16 | 19.48 | 0.72 | **22 385** |

## 5. Lecture & limites

- **Meilleur recall en classification (0.95)** parmi les modèles tabulaires : très peu de
  machines à risque manquées.
- En **RUL**, performances proches de la Random Forest (RMSE 27.2 vs 27.3) mais **meilleur
  score NASA du panel (22 385)** : moins de prédictions *en retard*, ce qui est
  précieux pour la **sûreté industrielle**.
- **Limites** : comme la RF, il traite chaque cycle **isolément** (pas de mémoire
  temporelle) ; il demande un **réglage d'hyperparamètres** plus soigné et reste moins
  interprétable qu'un modèle linéaire (atténué par les importances de variables).
