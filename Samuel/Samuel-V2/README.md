# Samuel-V2 — Maintenance prédictive MECHA (MSPR TPRE841)

Prédire la panne des machines MECHA à partir du dataset NASA C-MAPSS, avec **un modèle par
usine** (ma répartition Option B : usine = régime de fonctionnement). Deux tâches :

- **Régression `RUL`** : combien de cycles restent avant la panne ;
- **Classification `at_risk`** : la machine est-elle dans ses 30 derniers cycles ?

## Parcours de lecture (les notebooks se lisent dans l'ordre)

| Notebook | Contenu |
|---|---|
| `01_Preparation_Donnees.ipynb` | Vérifications, justification du plafond RUL=125 **par les données**, split par machine (anti-fuite), choix des features (tracé en JSON), régimes de l'Usine 2 et normalisation par régime |
| `02_Baseline_Lineaire.ipynb` | Plancher CDC : LinearRegression + LogisticRegression |
| `03_Random_Forest.ipynb` | Cœur CDC : configs comparées sur validation, importances des capteurs |
| `04_HGB.ipynb` | Boosting de gradient (même famille que XGBoost, en sklearn) |
| `05_LSTM.ipynb` | Réseau récurrent PyTorch : le seul modèle qui voit la **tendance** temporelle |
| `06_Synthese_Comparative.ipynb` | Tout côte à côte + global recombiné (707 machines) + sensibilité à l'écrêtage du RUL vrai |

## Ce qui est nouveau par rapport à ma V1

1. **`outils_mspr.py`** : module partagé — chargement, split par machine, normalisation par
   régime, scores PHM08/NASA, évaluation standardisée. Tous les modèles sont évalués
   exactement pareil.
2. **Validation par machine** (80/20, graine 42) : les réglages se choisissent sur la
   validation, jamais sur le test officiel. Fuite de l'early stopping par lignes corrigée
   (et là où sklearn ne permet pas de la corriger, elle est documentée).
3. **La classification `at_risk`** (absente en V1) sur les 4 modèles.
4. **La baseline linéaire** exigée par le CDC.
5. **Le LSTM PyTorch** avec fenêtres de 30 cycles, early stopping maison, dimensionné CPU.
6. Corrections V1 : scaler unique par usine (bug), prédictions RUL bornées [0, 125],
   score NASA total en plus de ma décomposition sûr/dangereux.

## Ce qui est gardé de ma V1 (et assumé)

- La **répartition par régime** (Option B) et ses conséquences → `docs/repartition_usines.md` ;
- La **normalisation par régime** de l'Usine 2 (ma trouvaille V1, généralisée à tous les modèles) ;
- La sélection de features par corrélation (automatisée, elle retrouve mes 17 features V1
  en Usine 1 et en découvre une 18ᵉ en Usine 2) ;
- La décomposition **PHM08 sûr / dangereux** (lecture métier du score NASA) ;
- Le style notebooks : chaque choix est justifié dans le notebook où il est fait.

## Dossiers

```
outils_mspr.py    module partagé (le seul .py — tout le reste est notebook)
Data/csv/         les 6 fichiers usines construits en V1
resultats/        metriques.json + prédictions par modèle (relus par la synthèse)
models/           modèles entraînés (joblib / .pt), rejouables sans réentraîner
figures/          toutes les figures exportées (soutenance)
docs/             un document par modèle + répartition usines + synthèse
```

## Reproduire

Environnement : le venv de Samuel-V1 (Python 3.12, sklearn 1.8, PyTorch 2.12 CPU).
Exécuter les notebooks dans l'ordre 01 → 06 (01 doit tourner en premier : il écrit les
listes de features ; 06 en dernier : il relit tous les résultats). Graine 42 partout.
