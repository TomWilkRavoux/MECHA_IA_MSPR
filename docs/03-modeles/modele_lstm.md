# Modèle - LSTM (Deep Learning, PyTorch / GPU)

> Notebook d'entraînement : `notebooks/03_lstm.py`
> Notebooks d'évaluation : `notebooks/analyse/eval_lstm_classifier.py`,
> `notebooks/analyse/eval_lstm_regressor.py`
> Architecture partagée : `ml/lstm.py` (`LSTMNet`, `load_lstm`)
> Modèles sauvegardés : `models/lstm_classifier.pt`, `models/lstm_regressor.pt`

## 1. Rôle & pourquoi ce choix

Tous les modèles précédents (baseline, RF, XGBoost) traitent **chaque cycle isolément** :
ils ignorent que la donnée est une **série temporelle**. Or la défaillance est un
**processus progressif** - les capteurs dérivent lentement avant la panne.

Le **LSTM** (Long Short-Term Memory) est un réseau récurrent conçu pour les séquences : il
**mémorise la tendance** sur une fenêtre de cycles et apprend la dynamique de dégradation.
C'est l'**approche de référence académique sur C-MAPSS**. On attend un gain surtout sur le
**RUL**, où la temporalité est déterminante.

## 2. Données en entrée - fenêtres glissantes

Spécifique au LSTM (`ml/prep.make_windows`) :

- Pour chaque machine, on découpe la trajectoire en **fenêtres glissantes de `SEQ_LEN = 30`
  cycles** ; la cible (`at_risk` ou `RUL`) est celle du **dernier cycle** de la fenêtre.
- Les machines trop courtes sont **left-paddées** (répétition du premier cycle).
- Features **normalisées** (`StandardScaler` partagé) - essentiel pour la convergence d'un
  réseau de neurones.
- En test : **une fenêtre par machine** (les `SEQ_LEN` derniers cycles observés), via
  `make_test_windows`.
- Cible RUL **bornée à 125**.

## 3. Construction technique

### Environnement (point de vigilance matériel)

Entraînement **sur GPU** avec **PyTorch (build cu128 / CUDA 12.8)**. La carte cible est une
**RTX 5070 (architecture Blackwell, compute capability sm_120)**, **non supportée par les
wheels TensorFlow** (compilés en CUDA 12.5). PyTorch cu128 est le backend fiable pour ce GPU.

### Architecture (`ml/lstm.py::LSTMNet`)

```
Entrée (batch, 30 cycles, 24 features)
  → LSTM(64, return_sequences)   → Dropout(0.2)
  → LSTM(32)                     → Dropout(0.2)   (on garde le dernier pas de temps)
  → Dense(16) + ReLU
  → Dense(1)                     → logit (classif) ou RUL (régression)
```

Deux modèles séparés partagent cette architecture :

| Tâche | Sortie / Perte | Réglages |
|---|---|---|
| Classification `at_risk` | logit + `BCEWithLogitsLoss(pos_weight)` | `pos_weight` = ratio négatifs/positifs (déséquilibre) ; sigmoïde appliquée à l'évaluation. |
| Régression `RUL` | linéaire + `MSELoss` | - |

### Entraînement

- Optimiseur **Adam** (`lr = 1e-3`), **batch = 512**, jusqu'à **30 epochs**.
- **Early stopping maison** : surveillance de la `val_loss`, **patience = 5**, restauration
  des **meilleurs poids** (`restore best weights`) → évite le sur-apprentissage.
- **Courbes d'apprentissage** (loss / val_loss) exportées :
  `reports/figures/lstm_history_clf.png`, `lstm_history_rul.png`.
- `random_seed = 42` (torch + numpy) pour la reproductibilité.

## 4. Métriques obtenues (jeu de test officiel)

**Classification `at_risk`**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **0.943** | **0.815** | **0.969** | **0.885** | **0.993** |

**Régression `RUL`**

| RMSE | MAE | R² | Score NASA |
|---|---|---|---|
| **26.28** | **18.52** | **0.74** | 23 802 |

## 5. Lecture & limites

- **Meilleur modèle sur les deux tâches** pour F1, recall, AUC, RMSE, MAE et R² : la
  temporalité de la dégradation paie clairement.
- **Recall 0.969** en classification : quasiment aucune machine à risque manquée.
- **Point de vigilance métier** : son **score NASA (23 802)** est légèrement **moins bon**
  que celui de XGBoost/RF → il fait un peu plus de prédictions *en retard*. Pour un usage
  sûreté, cela justifie de garder XGBoost/RF comme repli, ou d'ajuster la perte (pénalité
  asymétrique) dans une itération future.
- **Limites** : entraînement **plus coûteux** (GPU requis), modèle **moins interprétable**
  (« boîte noire »), et sensible à la longueur de fenêtre `SEQ_LEN` (hyperparamètre à
  explorer). Dépendance matérielle forte (CUDA/Blackwell).

## 6. Recommandation

Modèle **retenu pour l'exploitation applicative** (meilleures performances globales), avec
**RF / XGBoost en repli** interprétable et robuste, et suivi du **score NASA** comme
garde-fou de sûreté.
