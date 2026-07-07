# Script de génération du CSV de test avec pannes simulées
## MSPR TPRE841 — MECHA | Nadège MATEZERE

---

## Pourquoi ce script ?

Le jeu de démonstration du dashboard (`mecha_test_classification.csv`) contient
707 machines avec leur historique complet de cycles. Ce script génère un CSV
alternatif `mecha_test_pannes_simulees.csv` pour tester le dashboard avec des
scénarios de pannes plus marqués et vérifier que le modèle LSTM réagit correctement.

---

## Ce que fait le script

### Étape 1 — Chargement des 4 fichiers test

```python
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    df = pd.read_excel(SOURCE / f"test_{fd}.xlsx")
```

| Fichier source | Usine | Ligne | Machines |
|---|---|---|---|
| `test_FD001.xlsx` | Usine A | Ligne 1 (dédiée) | 100 |
| `test_FD002.xlsx` | Usine A | Ligne 2 (polyvalente) | 259 |
| `test_FD003.xlsx` | Usine B | Ligne 1 (dédiée) | 100 |
| `test_FD004.xlsx` | Usine B | Ligne 2 (polyvalente) | 248 |

### Étape 2 — Ajout du mapping MECHA

Chaque ligne reçoit les colonnes d'identification MECHA :

```python
machine_id        → "FD001_u001", "FD002_u042", ...
subset            → "FD001", "FD002", "FD003", "FD004"
usine             → "Usine A" (FD001/FD002) ou "Usine B" (FD003/FD004)
ligne_production  → "Ligne 1 (dédiée)" ou "Ligne 2 (polyvalente)"
```

### Étape 3 — Simulation des pannes (le changement clé)

Pour chaque subset, **20% des machines** sont sélectionnées aléatoirement
et on ne conserve que leurs **5 derniers cycles** :

```python
# 80% des machines → tous leurs cycles conservés (entre 31 et 303 cycles)
# 20% des machines → seulement les 5 derniers cycles
machines_en_panne = np.random.choice(machines, size=int(len(machines)*0.2))
dfs_pannes.append(sub[sub["machine_id"] == m].tail(5))
```

| Subset | Machines totales | Machines simulées en panne | Cycles conservés |
|---|---|---|---|
| FD001 | 100 | 20 | 5 derniers cycles |
| FD002 | 259 | 52 | 5 derniers cycles |
| FD003 | 100 | 20 | 5 derniers cycles |
| FD004 | 248 | 50 | 5 derniers cycles |

---

## Pourquoi 5 cycles ?

Les machines normales du dataset NASA ont entre **31 et 303 cycles** d'historique,
avec une moyenne d'environ **130 cycles** par machine.

Réduire à 5 cycles simule deux scénarios industriels réalistes :

**Scénario 1 — Capteur IoT nouvellement installé**
La machine tourne depuis longtemps mais le capteur vient d'être posé.
On n'a que 5 cycles de mesure disponibles.

**Scénario 2 — Perte de données historiques**
Suite à une migration de système ou une panne serveur, l'historique
a été perdu. On ne dispose que des 5 dernières mesures.

Dans les deux cas, le LSTM voit très peu d'historique et applique
le left-padding (répétition du premier cycle) pour compléter la
fenêtre de 30 cycles attendue — ce qui lui donne très peu d'information
sur la trajectoire réelle de dégradation → il prédit un RUL bas → alerte critique.

---

## Ce qui change dans le dashboard

| Indicateur | Jeu de démo | Pannes simulées | Pourquoi |
|---|---|---|---|
| Machines totales | 707 | 707 | Même nombre de machines |
| Cycles totaux | ~92 000 | moins | 20% des machines ont seulement 5 cycles |
| RUL min (cycles) | 4 | encore plus bas | Machines en toute fin de vie |
| Machines critiques | ~93 | plus élevé | 20% avec historique tronqué |
| Machines warning | variable | variable | Dépend des prédictions LSTM |

Les machines avec 5 cycles apparaissent en **rouge critique** dans le graphique
de priorisation — le LSTM les identifie correctement comme des machines
en fin de vie imminente, même sans historique complet.

---

## Utilisation

### Prérequis

Avoir les fichiers xlsx bruts dans `MSPR  2/` :