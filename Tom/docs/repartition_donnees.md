# Répartition des données & consolidation MECHA

Ce document explique **d'où viennent les données**, **pourquoi** elles sont réparties en
usines et lignes de production, et **comment** elles sont consolidées. Il documente deux
scripts :

- `analyseGlobal.py` - exploration qui **justifie** la répartition en usines / lignes ;
- `fusionDatasetFinal.py` - **consolidation** des jeux au vocabulaire MECHA.

---

## 1. Origine des données

Le jeu utilisé est le **NASA C-MAPSS Turbofan Engine Degradation** (public, PCoE / Kaggle),
conforme à la règle CDC « données publiques ou simulées, **aucune donnée réelle** ».

- Chaque **machine = une trajectoire *run-to-failure*** : on observe la machine cycle après
  cycle jusqu'à sa défaillance.
- **21 capteurs** (`T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR,
  farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32`) + **3 réglages opératoires**
  (`setting_1..3`).
- Le dataset est fourni en **4 sous-jeux** : `FD001`, `FD002`, `FD003`, `FD004`, qui
  partagent exactement le **même schéma** mais des **comportements différents**.

Ces 4 sous-jeux sont le point de départ de la répartition MECHA.

---

## 2. Justification de la répartition (`analyseGlobal.py`)

L'objectif : ne pas répartir les données au hasard, mais s'appuyer sur **deux axes réels**
qui distinguent les sous-jeux. `analyseGlobal.py` explore ces deux axes.

### Axe 1 - Régime de fonctionnement → définit la **ligne de production**

Hypothèse : certaines machines tournent sur un **régime unique et stable**, d'autres sur
**plusieurs régimes**. C'est vérifiable sur la dispersion des 3 réglages opératoires.

- **Preuve statistique** : l'écart-type des `setting_*` est quasi nul pour FD001/FD003,
  élevé pour FD002/FD004.
- **Confirmation par clustering** : un `KMeans` sur les réglages normalisés isole
  **1 régime** pour FD001/FD003 et **6 régimes** pour FD002/FD004.
- **Confirmation visuelle** : le nuage `setting_1`×`setting_2` est un point unique
  (FD001/FD003) ou 6 nuages distincts (FD002/FD004) ; l'histogramme du capteur 2 est
  unimodal (FD001/FD003) ou multimodal (FD002/FD004).

➡️ Interprétation MECHA : une **ligne dédiée** (régime unique) vs une **ligne polyvalente**
(régimes multiples).

### Axe 2 - Modes de défaillance → définit l'**usine**

Le **mode de panne n'est pas une colonne** du dataset : c'est une caractéristique
**documentée par la NASA** (PCoE) :

- FD001 & FD002 = **1 mode** de dégradation (compresseur HPC) ;
- FD003 & FD004 = **2 modes** (HPC + ventilateur).

Ne pouvant pas le *prouver* sur les données brutes, `analyseGlobal.py` cherche un **signal
de soutien** : deux mécanismes de panne coexistants produisent des **durées de vie plus
hétérogènes**. Le **coefficient de variation** de la durée de vie est effectivement plus
élevé pour FD003/FD004.

➡️ Interprétation MECHA : une **usine** produisant des pièces simples (1 mode) vs une usine
produisant des pièces complexes (2 modes).

### Grille 2×2 retenue

|  | 1 mode de panne → **Usine A** | 2 modes de panne → **Usine B** |
|---|---|---|
| **1 régime → Ligne 1 (dédiée)** | FD001 | FD003 |
| **6 régimes → Ligne 2 (polyvalente)** | FD002 | FD004 |

Chaque sous-jeu C-MAPSS est ainsi rattaché à un couple **(usine, ligne)** cohérent avec le
contexte industriel MECHA (5 usines, lignes automatisées, hétérogénéité des équipements).

---

## 3. Consolidation (`fusionDatasetFinal.py`)

Ce script **fusionne** les 4 sous-jeux en fichiers exploitables, au vocabulaire MECHA.

### Transformations appliquées

- **`add_ids`** - ajoute les colonnes métier à partir de la grille 2×2 :
  - `usine` : *Usine A* si FD001/FD002, sinon *Usine B* ;
  - `ligne_production` : *Ligne 1 (dédiée)* si FD001/FD003, sinon *Ligne 2 (polyvalente)* ;
  - `machine_id` : identifiant unique `"{subset}_u{unit:03d}"` (ex. `FD001_u012`) ;
  - `subset` : sous-jeu d'origine (traçabilité).
- **`reorder`** - place les colonnes d'identité en tête (`machine_id, subset, usine,
  ligne_production, unit, cycle`) puis les capteurs.
- **`build`** - concatène verticalement les 4 sous-jeux transformés.

### Fichiers produits (dans `assets/KaggleDataset/`)

| Fichier | Rôle | Cible |
|---|---|---|
| `mecha_train_classification.csv` | entraînement classification | `at_risk` (+ `RUL`) |
| `mecha_train_rul.csv` | entraînement régression | `RUL` |
| `mecha_test_classification.csv` | test (machines tronquées, **features seules**) | - |
| `mecha_rul_true.csv` | vérité terrain : RUL au dernier cycle (`RUL_true`) | `RUL_true` |

- **Volumétrie** : ~**160 359 lignes** d'entraînement, **709 machines**.
- La cible **`at_risk`** est dérivée du RUL en amont : `at_risk = 1 si RUL ≤ 30`
  (~**14 %** de positifs → jeu **déséquilibré**, pris en compte côté modèles).

---

## 4. Hypothèses & limites

- **Hypothèse forte** : l'assimilation *sous-jeu C-MAPSS → (usine, ligne) MECHA* est une
  **convention de mise en situation**. Elle est justifiée par des propriétés réelles
  (régimes, modes de panne) mais reste une modélisation pédagogique.
- Le **mode de panne** n'est pas une variable observable : il s'appuie sur la documentation
  NASA, seulement *soutenue* (non prouvée) par la dispersion des durées de vie.
- **Cohérence temporelle** : chaque machine a son propre compteur `cycle` (horodatage
  relatif) ; le split d'entraînement est fait **par machine** pour éviter toute fuite.
- **Limites connues** : capteurs constants sur certains régimes, volumétrie inégale entre
  sous-jeux, RUL borné à 125 cycles pour l'apprentissage (voir docs modèles).

> Pour le détail des colonnes, un **dictionnaire de données par colonne** reste à produire
> (item ouvert de la feuille de route).
