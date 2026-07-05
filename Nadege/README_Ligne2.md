# Maintenance Predictive Industrielle - MSPR TPRE841
## MECHA | NASA C-MAPSS Dataset - Ligne2 (FD002+FD004)

Evaluation comparative LSTM vs Random Forest sur le groupe Ligne2 :
6 conditions operationnelles, 2 types de panne, dataset plus complexe que Ligne1.

---

## Resultats - Ligne2 (FD002+FD004)

359 moteurs dans le jeu de test
114 moteurs a risque (RUL <= 30 cycles) / 245 normaux

### Regression RUL

| Modele | MAE (cycles) | RMSE (cycles) | R2 | Erreur max | % pred <= 30 cycles |
|---|---|---|---|---|---|
| LSTM | 16.48 | 28.49 | 0.561 | 116.6 | 86.4% |
| Random Forest | 16.52 | 21.03 | 0.761 | 68.2 | 82.8% |

Les deux modeles ont une MAE quasi-identique (16.48 vs 16.52 cycles).
Le Random Forest presente un RMSE plus faible (21.03 vs 28.49) et une erreur max
bien plus contenue (68.2 vs 116.6 cycles) — il commet moins de grosses erreurs isolees.
Le R2 du Random Forest (0.761) est nettement superieur a celui du LSTM (0.561) :
sur des donnees multi-conditions, la structure tabulaire du RF capture mieux
les regimes distincts de degradation.

### Classification risque (seuil RUL <= 30 cycles)

| Modele | Accuracy | Precision | Recall | F1 | AUC-ROC | Faux negatifs | Faux positifs |
|---|---|---|---|---|---|---|---|
| LSTM | 0.949 | 0.861 | 0.921 | 0.890 | 0.986 | 9 | 17 |
| Random Forest | 0.945 | 0.884 | 0.868 | 0.876 | 0.980 | 15 | 13 |

Le LSTM detecte 105 moteurs a risque sur 114 (9 rates).
Le Random Forest n en detecte que 99 (15 rates), soit 1.7x plus de pannes non anticipees.
Le LSTM genere plus de faux positifs (17 vs 13) : son Recall eleve se paye
par legerement plus de fausses alarmes, compromis acceptable en maintenance industrielle.

---

## Comparaison Ligne1 vs Ligne2

| Metrique | LSTM Ligne1 | LSTM Ligne2 | Evolution |
|---|---|---|---|
| MAE (cycles) | 9.63 | 16.48 | +71% |
| RMSE (cycles) | 15.29 | 28.49 | +86% |
| R2 | 0.851 | 0.561 | -34% |
| Recall | 0.933 | 0.921 | -1.3% |
| F1 | 0.933 | 0.890 | -4.6% |
| Faux negatifs | 3 / 45 | 9 / 114 | ratio stable |
| AUC-ROC | 0.998 | 0.986 | stable |

La degradation est concentree sur la regression RUL : les 6 conditions
operationnelles de Ligne2 introduisent une variabilite que le LSTM gere moins
bien en regression. En revanche, la classification reste robuste sur les deux lignes.

---

## Recommandation metier

| Seuil | RUL | Action |
|---|---|---|
| Alerte | <= 50 cycles | Notifier l equipe maintenance |
| Critique | <= 30 cycles | Intervention immediate |

Modele recommande : LSTM
- Recall = 0.921 : rate seulement 9 pannes sur 114
- AUC-ROC = 0.986 : discrimination tres elevee Normal / A risque
- Point de vigilance : erreur max 116.6 cycles en regression RUL

Strategie hybride envisageable :
- LSTM pour les alertes de classification (meilleur Recall)
- Random Forest pour affiner l estimation RUL sur conditions extremes (meilleur RMSE/R2)

---

## Dataset - NASA C-MAPSS Ligne2

| Fichier | Description |
|---|---|
| train_FD002.xlsx | Donnees entrainement - 6 conditions, 1 type de panne |
| train_FD004.xlsx | Donnees entrainement - 6 conditions, 2 types de panne |
| test_FD002.xlsx | Donnees de test FD002 |
| test_FD004.xlsx | Donnees de test FD004 |
| RUL_FD002.txt | RUL reelle au dernier cycle - FD002 |
| RUL_FD004.txt | RUL reelle au dernier cycle - FD004 |

| Parametre | Valeur | Description |
|---|---|---|
| RUL_MAX | 125 cycles | Cap piecewise linear |
| SEQ_LEN | 50 cycles | Longueur des sequences LSTM |
| RISK_THRESHOLD | 30 cycles | Seuil Normal / A risque |

---

## Modeles

### LSTM

| Couche | Details |
|---|---|
| LSTM | 64 unites, return_sequences=True |
| Dropout | 0.2 |
| LSTM | 32 unites |
| Dropout | 0.2 |
| Dense | 16 unites, activation relu |
| Dense | 1 unite (sortie RUL) |

| Parametre | Valeur |
|---|---|
| Optimizer | Adam lr=1e-3 |
| Batch size | 128 |
| Epochs max | 80 |
| EarlyStopping | patience=10, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=8 |
| class_weight | 0:1.0 / 1:n_neg/n_pos |

### Random Forest

| Parametre regression | Valeur |
|---|---|
| n_estimators | 100 |
| random_state | 42 |

| Parametre classification | Valeur |
|---|---|
| n_estimators | 300 |
| max_depth | 20 |
| class_weight | balanced |

---

## Prerequis

pip install tensorflow scikit-learn pandas numpy matplotlib openpyxl ipywidgets sweetviz

---

## Auteur

Nadege MATEZERE - EPSI | EISI IA | MSPR TPRE841
