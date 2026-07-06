# Guide utilisateur métier - Dashboard de maintenance prédictive

> Interface : `frontend/dashboard.py` (Streamlit)
> Public : équipes **maintenance** et **production** de MECHA
> Volet technique (architecture, endpoints) : [`dashboard_technique.md`](dashboard_technique.md)

Ce dashboard aide à **prioriser les interventions** de maintenance sur le parc de machines, en
s'appuyant sur le modèle LSTM (état `at_risk` + RUL). Il répond à l'objectif du CDC : rendre les
résultats de l'IA **compréhensibles et exploitables** (alertes, scores de dégradation, tableaux
de bord).

## 1. Démarrage

Dans la **barre latérale** :
1. Vérifier que l'API est « OK » (voyant vert). Sinon, démarrer le backend.
2. Choisir la **source de données** : jeu de démonstration ou import d'un CSV (mêmes colonnes).
3. Filtrer par **usine** / **ligne de production** et régler le nombre de machines à analyser.
4. Cliquer sur **▶️ Analyser le parc**.

## 2. Les trois écrans

### 🏭 Vue parc
Vision d'ensemble : KPI (nombre de machines critiques / à surveiller / normales, RUL minimum),
**priorisation des interventions** (machines classées par RUL croissant - la plus urgente en
haut), tableau détaillé et export CSV des alertes.

### 🧵 Par ligne
Agrégation par **ligne de production** (et usine) : répartition des niveaux d'alerte, RUL moyen
et RUL minimum par ligne. Permet de repérer **où concentrer** les moyens de maintenance - une
ligne concentrant plusieurs machines critiques est prioritaire.

### 🔍 Fiche machine
Analyse détaillée d'**une** machine issue de l'analyse du parc, en trois onglets :
- **Régression (RUL)** : trajectoire du RUL prédit **cycle par cycle**.
- **Classification (risque)** : évolution de la **probabilité de risque** cycle par cycle.
- **Capteurs** : évolution de quelques capteurs clés (contexte physique de la dégradation).

### 🔧 Analyse unitaire
Analyse **à la demande** d'une seule machine, **sans avoir à analyser tout le parc**. On
choisit une machine dans la liste, et le modèle est interrogé immédiatement (endpoint
`/predict`) : niveau d'alerte, RUL, probabilité de risque, nombre de cycles utilisés, puis les
mêmes trois onglets de graphes que la fiche machine. Utile pour un **diagnostic ciblé** (une
machine signalée par un technicien) ou une **démonstration rapide**.

## 3. Comment lire les indicateurs

### RUL (Remaining Useful Life)
Nombre de **cycles restants** estimés avant défaillance. Plus il est bas, plus l'intervention
est urgente. Sur la trajectoire, la courbe **décroît** à mesure que la machine se dégrade.

### Probabilité de risque
Probabilité, donnée par la tête de **classification**, que la machine soit « à risque ».
Au-dessus de **0.5**, le modèle classe la machine comme `at_risk`.

### Niveaux d'alerte (règle métier)
| Badge | Signification | Règle |
|---|---|---|
| 🟢 Normal | Fonctionnement nominal | non `at_risk` |
| 🟠 Surveiller | À risque, dégradation en cours | `at_risk`, RUL > 15 et proba < 0.75 |
| 🔴 Critique | Intervention prioritaire | `at_risk` **et** (RUL ≤ 15 **ou** proba ≥ 0.75) |

## 4. Pourquoi le seuil de risque est fixé à 30 cycles

La cible de classification est définie par : **`at_risk = 1 si RUL ≤ 30 cycles`**
(`ml/prep.py`, `RISK_THRESHOLD = 30`). Ce choix se justifie ainsi :

- **Horizon de planification maintenance** : 30 cycles offrent aux équipes une **marge
  suffisante** pour planifier une intervention (commande de pièces, créneau d'atelier) **avant**
  la panne - cohérent avec l'exigence de **continuité de production** de MECHA.
- **Convention C-MAPSS** : un seuil de l'ordre de 30 cycles est une pratique courante sur le
  jeu NASA C-MAPSS dont dérivent nos données (cf. [`repartition_donnees.md`](repartition_donnees.md)),
  ce qui rend nos résultats **comparables** à la littérature.
- **Compromis détection précoce / fausses alertes** : un seuil trop bas alerterait **trop tard**
  (peu de marge d'action) ; un seuil trop haut **noierait** les équipes sous des alertes
  prématurées. 30 cycles équilibre ces deux risques.

> Ce seuil est **paramétrable** (`RISK_THRESHOLD`) : il pourra être ajusté par ligne/usine selon
> le temps réel de réaction des équipes de maintenance. Le RUL borné à 125 (`RUL_CAP`) et les
> seuils d'alerte critique (15 cycles, proba 0.75) sont ajustables de la même manière dans
> `backend/api/inference.py`.

## 5. Lecture des graphes de la fiche machine

- **Trajectoire RUL** : la courbe doit décroître ; deux repères en pointillés matérialisent le
  **seuil à risque (30)** et le **seuil critique (15)**. Le cycle où la courbe franchit la
  ligne 30 indique **quand** la machine est passée « à risque ».
- **Probabilité de risque** : la courbe monte à l'approche de la défaillance ; repères à **0.5**
  (décision) et **0.75** (critique).
- **Capteurs** : les dérives (température, pression, régime…) donnent le **contexte physique** de
  la dégradation et rendent la prédiction plus **crédible** auprès des techniciens.

## 6. Limites & conditions d'usage

- Les prédictions supposent des cycles fournis dans le **bon format** (24 variables, ordonnés) ;
  un CSV mal formé produit des résultats non fiables.
- Le RUL est une **estimation** (RMSE ≈ 26 cycles sur le test) : c'est un outil d'**aide à la
  décision**, pas une garantie. Le niveau d'alerte reste le repère opérationnel principal.
- Sur les **premiers cycles** d'une machine, la fenêtre est complétée par padding : la
  trajectoire y est moins fiable (peu d'historique observé).
- L'outil **priorise** ; la décision finale d'intervention revient aux équipes maintenance, qui
  croisent ces alertes avec leur connaissance terrain.
