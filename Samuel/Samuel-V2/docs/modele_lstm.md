# Modèle — LSTM (réseau récurrent, PyTorch)

> Notebook : `05_LSTM.ipynb`
> Modèles : `models/lstm_{regression,classification}_usine{1,2}.pt` + `lstm_scaler_usine{1,2}.joblib`
> Figures : `figures/lstm_courbes_apprentissage.png`, `figures/lstm_confusion.png`

## Rôle et pourquoi ce choix

Tous les modèles tabulaires regardent chaque cycle isolément : ils ignorent que la panne est
un **processus** (les capteurs dérivent lentement pendant des dizaines de cycles). Le LSTM
lit la **séquence des 30 derniers cycles** et apprend la trajectoire de dégradation — la
vitesse de dérive dit combien de temps il reste. C'est l'approche de référence de la
littérature C-MAPSS. Codé en PyTorch (techno que je pratique déjà : projet DQN).

## Construction

```
Entrée (batch, 30 cycles, 17-18 features)
  → LSTM(64) + Dropout(0.2)
  → LSTM(32) + Dropout(0.2)     (dernier pas de temps = résumé de la fenêtre)
  → Linéaire(16) + ReLU → Linéaire(1)
```

| Réglage | Valeur | Justification |
|---|---|---|
| Fenêtre | 30 cycles, left-padding en début de vie | assez pour voir une tendance, une fenêtre par cycle |
| Sous-échantillonnage | 1 fenêtre sur 2 à l'entraînement | 2 fenêtres voisines se recouvrent à 29/30 → moitié moins de CPU, information quasi identique |
| Perte | `MSELoss` (RUL/125) / `BCEWithLogitsLoss(pos_weight≈6.2)` | cible normalisée (voir piège) / rééquilibrage du déséquilibre |
| Optimiseur | Adam `lr=1e-3`, batch 512, ≤ 20 époques | standard |
| Early stopping | patience 5 sur la perte de **validation par machine**, meilleurs poids restaurés | anti sur-apprentissage, honnête (pas de fuite) |
| Matériel | **CPU**, quelques minutes par modèle | pas de dépendance GPU : argument de coût pour un réentraînement en usine |

**Piège documenté (notebook 05)** : au premier essai, entraîné sur le RUL brut (0-125), le
réseau n'apprenait que la moyenne (perte figée à la variance du RUL, R² ≈ 0). Les activations
d'un LSTM vivent entre -1 et 1 : la cible doit être normalisée **comme les features**.
Entraîner sur RUL/125 et re-multiplier à la prédiction règle tout — la classification, dont
la cible est déjà 0/1, marchait du premier coup.

## Résultats (test officiel)

**Régression RUL**

| Usine | RMSE | MAE | R² | NASA moyen/machine |
|---|---|---|---|---|
| 1 | **17.13** | **13.11** | **0.83** | **6.0** |
| 2 | 28.42 | 20.63 | 0.73 | **27.8** |

**Classification `at_risk`**

| Usine | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | **0.975** | **0.917** | **0.978** | **0.946** | **0.997** |
| 2 | **0.941** | 0.800 | **0.982** | **0.882** | **0.995** |

## Lecture et limites

- **Meilleur modèle du projet** sur quasiment tout, et de loin en classification Usine 1
  (F1 0.946 : 2 % des machines à risque manquées, 8 % de fausses alertes).
- En régression Usine 2, il fait jeu égal avec le HGB en RMSE (28.4 vs 28.3) mais garde le
  **meilleur score NASA** : ses erreurs sont moins souvent du côté dangereux.
- **Limites** : boîte noire (pas d'importance de variables native), entraînement plus long
  qu'une forêt, sensible à la longueur de fenêtre (30 = à explorer si on itère).
