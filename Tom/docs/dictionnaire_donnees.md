# Dictionnaire de données — MECHA (maintenance prédictive)

Description **par colonne** des jeux de données consolidés (`assets/KaggleDataset/mecha_*.csv`).
Complète [`repartition_donnees.md`](repartition_donnees.md) (origine, répartition usines/lignes,
consolidation). La cohérence de ce dictionnaire avec le schéma réel est **vérifiée
automatiquement** par `tests/test_data_dictionary.py`.

- **Origine** : NASA C-MAPSS Turbofan Engine Degradation (public, PCoE / Kaggle) — aucune donnée
  réelle (règle CDC). Les capteurs et réglages suivent la nomenclature C-MAPSS (Saxena et al.).
- **Granularité** : une ligne = **un cycle d'exploitation d'une machine**. Une machine = une
  trajectoire *run-to-failure*.
- **Colonnes** : 6 d'identité + 3 réglages opératoires + 21 capteurs + cibles.

## 1. Colonnes d'identité (vocabulaire MECHA)

| Colonne | Catégorie | Type | Unité | Description / domaine |
|---|---|---|---|---|
| `machine_id` | identité | texte | — | Identifiant machine unique `"{subset}_u{unit:03d}"` (ex. `FD001_u012`). |
| `subset` | identité | catégoriel | — | Sous-jeu C-MAPSS d'origine (traçabilité) : `FD001`, `FD002`, `FD003`, `FD004`. |
| `usine` | identité | catégoriel | — | Usine MECHA (dérivée du **mode de panne**) : *Usine A* (1 mode, FD001/FD002) ou *Usine B* (2 modes, FD003/FD004). |
| `ligne_production` | identité | catégoriel | — | Ligne (dérivée du **régime**) : *Ligne 1 (dédiée)* (1 régime, FD001/FD003) ou *Ligne 2 (polyvalente)* (6 régimes, FD002/FD004). |
| `unit` | identité | entier | — | Numéro du moteur au sein de son sous-jeu (1..N). |
| `cycle` | temporel | entier | cycle | Compteur de cycles depuis la mise en service (horodatage **relatif**), ≥ 1, croissant par machine. |

## 2. Réglages opératoires (conditions de vol)

| Colonne | Catégorie | Type | Unité | Description / domaine |
|---|---|---|---|---|
| `setting_1` | réglage | réel | kft (altitude) | Condition opératoire 1 — **altitude** (≈ 0–42). Dispersion ≈ 0 en régime unique. |
| `setting_2` | réglage | réel | Mach | Condition opératoire 2 — **nombre de Mach** (≈ 0–0.84). |
| `setting_3` | réglage | réel | ° (TRA) | Condition opératoire 3 — **manette des gaz** (Throttle Resolver Angle). |

## 3. Capteurs (21 mesures moteur, nomenclature C-MAPSS)

| Colonne | Catégorie | Type | Unité | Description / domaine |
|---|---|---|---|---|
| `T2` | capteur | réel | °R | Température totale à l'entrée soufflante (fan inlet). |
| `T24` | capteur | réel | °R | Température totale en sortie du compresseur BP (LPC outlet). |
| `T30` | capteur | réel | °R | Température totale en sortie du compresseur HP (HPC outlet). |
| `T50` | capteur | réel | °R | Température totale en sortie de la turbine BP (LPT outlet). |
| `P2` | capteur | réel | psia | Pression à l'entrée soufflante (fan inlet). |
| `P15` | capteur | réel | psia | Pression totale dans le conduit de dérivation (bypass-duct). |
| `P30` | capteur | réel | psia | Pression totale en sortie du compresseur HP (HPC outlet). |
| `Nf` | capteur | réel | rpm | Régime physique de la soufflante (fan speed). |
| `Nc` | capteur | réel | rpm | Régime physique du corps (core speed). |
| `epr` | capteur | réel | — | Rapport de pression moteur (Engine Pressure Ratio, P50/P2). |
| `Ps30` | capteur | réel | psia | Pression **statique** en sortie du compresseur HP. |
| `phi` | capteur | réel | pps/psi | Rapport débit carburant / `Ps30`. |
| `NRf` | capteur | réel | rpm | Régime **corrigé** de la soufflante (corrected fan speed). |
| `NRc` | capteur | réel | rpm | Régime **corrigé** du corps (corrected core speed). |
| `BPR` | capteur | réel | — | Taux de dilution (Bypass Ratio). |
| `farB` | capteur | réel | — | Rapport carburant/air au brûleur (burner fuel-air ratio). |
| `htBleed` | capteur | réel | — | Enthalpie de prélèvement d'air (bleed enthalpy). |
| `Nf_dmd` | capteur | réel | rpm | Régime soufflante **demandé** (demanded fan speed). |
| `PCNfR_dmd` | capteur | réel | rpm | Régime soufflante corrigé **demandé** (demanded corrected fan speed). |
| `W31` | capteur | réel | lbm/s | Débit de refroidissement turbine HP (HPT coolant bleed). |
| `W32` | capteur | réel | lbm/s | Débit de refroidissement turbine BP (LPT coolant bleed). |

> Certains capteurs sont **quasi constants** sur un régime unique (FD001/FD003) et ne portent
> alors pas d'information — d'où l'intérêt des jeux multi-régimes (FD002/FD004).

## 4. Cibles & vérité terrain

| Colonne | Catégorie | Type | Unité | Description / domaine |
|---|---|---|---|---|
| `RUL` | cible | entier | cycles | **Remaining Useful Life** : cycles restants avant défaillance. Borné à **125** à l'apprentissage (convention C-MAPSS piecewise). |
| `at_risk` | cible | binaire | 0/1 | État « à risque », dérivé : `at_risk = 1 si RUL ≤ 30` (~14 % de positifs → déséquilibre). |
| `RUL_true` | vérité terrain | entier | cycles | RUL au **dernier cycle observé** de chaque machine de test (fichier `mecha_rul_true.csv`). |

## 5. Présence des colonnes par fichier

| Fichier | Identité + réglages + capteurs | `RUL` | `at_risk` | `RUL_true` |
|---|:---:|:---:|:---:|:---:|
| `mecha_train_classification.csv` | ✅ | ✅ | ✅ | — |
| `mecha_train_rul.csv` | ✅ | ✅ | — | — |
| `mecha_test_classification.csv` | ✅ | — | — | — |
| `mecha_rul_true.csv` | identité seule (`machine_id, subset, usine, ligne_production, unit`) | — | — | ✅ |

## 6. Hypothèses & limites

- L'assimilation *sous-jeu C-MAPSS → (usine, ligne) MECHA* est une **convention de mise en
  situation** (voir `repartition_donnees.md` §4), justifiée par des propriétés réelles (régimes,
  modes de panne) mais pédagogique.
- Le **mode de panne** n'est pas une colonne observable : il provient de la documentation NASA.
- `cycle` est un **horodatage relatif** propre à chaque machine ; le split d'apprentissage est
  fait **par machine** pour éviter toute fuite.
- `RUL` borné à 125 cycles pour l'apprentissage ; capteurs constants sur certains régimes.
