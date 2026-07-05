# Maintenance Predictive Industrielle - MSPR TPRE841
## MECHA | NASA C-MAPSS Dataset

Pipeline complet de maintenance predictive par apprentissage automatique :
prediction de la duree de vie restante (RUL) et classification du risque machine
a partir des donnees capteurs des moteurs d avion NASA C-MAPSS.

---

## Prerequis

Python : 3.12
Environnement recommande : venv ou conda

Installation des dependances :
pip install tensorflow scikit-learn pandas numpy matplotlib openpyxl ipywidgets sweetviz

Fichiers de donnees requis :

| Fichier | Description |
|---|---|
| train_FD001.xlsx | Donnees entrainement - 1 condition, 1 type de panne |
| train_FD002.xlsx | Donnees entrainement - 6 conditions, 1 type de panne |
| train_FD003.xlsx | Donnees entrainement - 1 condition, 2 types de panne |
| train_FD004.xlsx | Donnees entrainement - 6 conditions, 2 types de panne |
| test_FD001.xlsx | Donnees de test FD001 |
| test_FD002.xlsx | Donnees de test FD002 |
| test_FD003.xlsx | Donnees de test FD003 |
| test_FD004.xlsx | Donnees de test FD004 |
| RUL_FD001.txt | RUL reelle au dernier cycle - FD001 |
| RUL_FD002.txt | RUL reelle au dernier cycle - FD002 |
| RUL_FD003.txt | RUL reelle au dernier cycle - FD003 |
| RUL_FD004.txt | RUL reelle au dernier cycle - FD004 |

---

## Structure du dossier

Nadege/
├── MERGE_MODELISATION.ipynb
└── README.md

---

## Dataset - NASA C-MAPSS

Le dataset C-MAPSS simule la degradation de moteurs d avion sous differentes
conditions operationnelles. Chaque moteur est suivi cycle par cycle jusqu a
sa defaillance.

| Groupe | Merge | Conditions operationnelles | Types de panne | Moteurs test |
|---|---|---|---|---|
| Ligne1 | FD001 + FD003 | 1 | 2 | 200 |
| Ligne2 | FD002 + FD004 | 6 | 2 | variable |

Variables principales :
- unit_id : identifiant du moteur
- cycle : cycle de fonctionnement (index temporel)
- op_setting_1/2/3 : conditions operationnelles
- sensor_1 a sensor_21 : mesures capteurs
- RUL : Remaining Useful Life - cycles restants avant defaillance (cible)

| Parametre | Valeur | Description |
|---|---|---|
| RUL_MAX | 125 cycles | Cap piecewise linear |
| SEQ_LEN | 50 cycles | Longueur des sequences LSTM |
| RISK_THRESHOLD | 30 cycles | Seuil Normal / A risque |

---

## Pipeline

Chargement des 8 fichiers xlsx + 4 fichiers RUL txt
Merge par groupe : Ligne1 (FD001+FD003) / Ligne2 (FD002+FD004)
Calcul RUL = max_cycle - cycle_courant (cap a 125)
Normalisation MinMaxScaler : fit sur train / transform sur test
Construction sequences : fenetres glissantes (50 cycles x 24 features)
Entrainement LSTM + Random Forest
Regression RUL : MAE / RMSE / R2
Classification risque : Recall / F1 / AUC-ROC
Dashboard visualisation + Conclusion metier

---

## Modeles

### LSTM

Architecture regression RUL :

| Couche | Details |
|---|---|
| LSTM | 64 unites, return_sequences=True |
| Dropout | 0.2 |
| LSTM | 32 unites |
| Dropout | 0.2 |
| Dense | 16 unites, activation relu |
| Dense | 1 unite (sortie RUL) |

Architecture classification risque :

| Couche | Details |
|---|---|
| LSTM | 64 unites, return_sequences=True |
| Dropout | 0.2 |
| LSTM | 32 unites |
| Dropout | 0.2 |
| Dense | 1 unite, activation sigmoid |

Hyperparametres :

| Parametre | Valeur |
|---|---|
| Optimizer | Adam lr=1e-3 |
| Batch size | 128 |
| Epochs max | 80 |
| Validation split | 0.2 |
| EarlyStopping | patience=10, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=8 |
| class_weight | 0:1.0 / 1:n_neg/n_pos |

Entree : tenseur 3D (samples, 50, 24)
Avantage : capture la dynamique temporelle de la degradation
Limite : entrainement plus long, sensible aux hyperparametres

---

### Random Forest

Regression :

| Parametre | Valeur |
|---|---|
| n_estimators | 100 |
| random_state | 42 |
| n_jobs | -1 |

Classification :

| Parametre | Valeur |
|---|---|
| n_estimators | 300 |
| max_depth | 20 |
| class_weight | balanced |
| random_state | 42 |
| n_jobs | -1 |

Entree : matrice 2D (samples, 1200) - sequences aplaties 50 x 24
Avantage : robuste, rapide, interpretable
Limite : ne modelise pas nativement la structure temporelle

---

## Resultats - Ligne1 (FD001+FD003)

200 moteurs dans le jeu de test (100 FD001 + 100 FD003)
45 moteurs a risque (RUL <= 30 cycles) / 155 normaux

### Regression RUL

| Modele | MAE (cycles) | RMSE (cycles) | R2 | Erreur max | % pred <= 30 cycles |
|---|---|---|---|---|---|
| LSTM | 9.63 | 15.29 | 0.851 | 89.6 | 94.0% |
| Random Forest | 11.98 | 15.90 | 0.839 | 45.5 | 92.5% |

Le LSTM obtient la meilleure MAE (9.63 cycles) et le meilleur R2 (0.851).
Le Random Forest presente une erreur max plus faible (45.5 vs 89.6 cycles).

### Classification risque (seuil RUL <= 30 cycles)

| Modele | Accuracy | Precision | Recall | F1 | AUC-ROC | Faux negatifs | Faux positifs |
|---|---|---|---|---|---|---|---|
| LSTM | 0.970 | 0.933 | 0.933 | 0.933 | 0.998 | 3 | 3 |
| Random Forest | 0.930 | 0.919 | 0.756 | 0.829 | 0.992 | 11 | 3 |

En maintenance predictive, le Recall est la metrique prioritaire.
Un faux negatif = moteur qui tombe en panne sans alerte = arret non planifie.
Le LSTM detecte 42 moteurs a risque sur 45 (3 rates).
Le Random Forest n en detecte que 34 (11 rates), soit 3.7x plus de pannes non anticipees.

---

## Recommandation metier

| Seuil | RUL | Action |
|---|---|---|
| Alerte | <= 50 cycles | Notifier l equipe maintenance |
| Critique | <= 30 cycles | Intervention immediate |

Modele recommande : LSTM
- Recall = 0.933 : rate seulement 3 pannes sur 45
- MAE = 9.63 cycles : erreur moyenne inferieure a 10 cycles
- 94% des predictions sont a moins de 30 cycles de la realite
- AUC-ROC = 0.998 : discrimination quasi-parfaite Normal / A risque

---

## Auteur

Nadege MATEZERE - EPSI | EISI IA | MSPR TPRE841
