# Feuille de route & cahier des charges technique
 
## MSPR TPRE841 — « Conception d'une solution applicative en adéquation avec l'environnement technique étudié »
 
> **Certification :** Expert en Informatique et Système d'Information — RNCP35584 **Bloc 4 :** Concevoir et développer des solutions applicatives métier et spécifiques (mobiles, embarquées et ERP) **Entreprise (fictive) :** MECHA — fabrication de pièces mécaniques de précision (aéronautique / automobile) **Cas d'usage retenu :** Maintenance prédictive industrielle pilotée par l'IA
 
Ce document reformule le sujet en un plan d'action concret. Il sert de référence d'équipe : **ce qu'il faut faire**, **dans quel ordre**, et **ce qu'il faut rendre**.
 
---
 
## 1. L'essentiel en un coup d'œil
 
|Élément|Détail|
|---|---|
|**Équipe**|4 apprenants (5 max si groupe impair)|
|**Préparation**|26 heures|
|**Soutenance**|50 min : 20 min de présentation + 30 min d'entretien collectif|
|**Jury**|2 évaluateurs externes (ne connaissent ni la formation ni l'équipe)|
|**Continuité**|Suite directe de la MSPR TPRE831 (PoC / faisabilité déjà validé)|
|**Données**|Publiques (Kaggle / GitHub) ou **simulées** — **aucune donnée réelle** (confidentialité)|
|**Note finale =**|Qualité du travail + exhaustivité des livrables + qualité de la soutenance|
 
> ⚠️ **Point d'attention majeur :** on passe du _cadrage_ (MSPR 1) à la **conception + réalisation d'un prototype fonctionnel**. Le jury attend du concret : du code qui tourne, des modèles évalués, une appli d'exploitation décrite, le tout conteneurisé et documenté.
 
---
 
## 2. Compétences évaluées (à démontrer explicitement)
 
Chaque compétence du Bloc 4 doit transparaître dans les livrables **et** dans la soutenance. À garder comme grille d'auto-contrôle :
 
1. **Collecter les besoins métiers** via interviews → guide d'entretien simulé + synthèse des besoins.
2. **Concevoir une architecture applicative** (distribuée / micro-services, évolutive, tolérante aux pannes).
3. **Développer une application adéquate** avec un langage approprié, dans le respect du cahier des charges.
4. **Développer une solution applicative intégrée** (paramétrage + langage spécifique de l'éditeur).
5. **Tester la solution** (identification des erreurs, plans de correction avant mise en production).
6. **Appliquer l'intégration continue** (outil de CI).
7. **Vérifier la conformité** entre solution et fonctionnalités attendues → documentation orientée utilisateur.
8. **Conduire le changement** (participation, communication, formation des utilisateurs).
---
 
## 3. Architecture cible (vue technique)
 
Le sujet impose une architecture cohérente avec l'environnement industriel de MECHA, couvrant **collecte → centralisation → exploitation** des données. Enchaînement attendu (d'après le schéma du sujet) :
 
```
[ Sources industrielles ]                 [ Plateforme Data / IA ]
  - Données machines IoT (temps réel)        - Collecte des données
    température, vitesse, état, temps cycle   - Préparation des données
  - Historique maintenance (journalier) ──▶  - Entraînement des modèles d'IA
        │  (Temps réel / Batch)                        │
        ▼                                              ▼
[ Supervision SCADA / MES ]               [ Tests d'intégration / Validation ]
  - Suivi atelier, horodatage, synchro              │
  - Centralisation (entrepôt de données)            ▼
                                          [ Conteneurisation / Déploiement ]
                                                     │
                                                     ▼
                                          [ Exploitation métier ]
                                            - Alertes
                                            - Tableaux de bord
                                            - Maintenance prédictive
```
 
### Contraintes à intégrer et à justifier
 
- **Qualité des données** : bruit, valeurs manquantes, hétérogénéité des capteurs.
- **Historique suffisant** pour l'apprentissage des modèles.
- **Cohérence temporelle** : horodatage et synchronisation (indispensable pour détecter les dérives).
- **Volumétrie** : gestion de gros volumes pour la maintenance prédictive.
- **Centralisation** : entrepôt de données ou stockage intermédiaire pour les analyses transverses.
> Tous les choix d'architecture doivent être **décrits en détail ET justifiés** au regard des contraintes industrielles.
 
---
 
## 4. Volet Data Science / IA
 
### 4.1 Cas d'usage à modéliser
 
Deux objectifs de modélisation imposés :
 
- **Classification** de l'état machine : _fonctionnement normal_ / _à risque_.
- **Prédiction d'un indicateur de dégradation** : ex. temps restant avant défaillance (**RUL — Remaining Useful Life**).
### 4.2 Algorithmes attendus
 
- **Arbres de décision & ensembles d'arbres** (Random Forest) → cœur de la solution.
- **Régression simple** comme baseline de comparaison.
- **Détection d'anomalies / comportements atypiques** (optionnel mais valorisé).
- ✅ **Justifier chaque choix algorithmique** au regard des données et des objectifs fonctionnels.
### 4.3 Jeux de données (3 stratégies possibles)
 
1. **Adapter** un jeu public existant (filtrer, renommer au vocabulaire MECHA, créer des variables dérivées, injecter des pannes).
2. **Générer** un jeu synthétique cohérent (Python + `pandas`/`numpy`, distributions + règles métier).
3. **Combiner** les deux.
**Pistes de jeux publics pertinents** (maintenance prédictive, à vérifier/valider) :
 
- _AI4I 2020 Predictive Maintenance Dataset_ (UCI) — classification de panne + types de défaillance.
- _NASA C-MAPSS / Turbofan Engine Degradation_ — idéal pour le calcul de RUL.
- Recherches Kaggle : « predictive maintenance », « machine sensor data », « industrial IoT ».
**Cadrage à faire avant de coder les données :**
 
- Quel cas d'usage exact ? (maintenance prédictive d'un type de machine, dérive de production…)
- Quelles entités ? (machines, capteurs, lignes, usines, OF, interventions)
- Quelle granularité temporelle ? (seconde / minute / heure / jour)
- Quelles variables réalistes ? (température, vibration, état, codes défaut, temps depuis dernière maintenance…)
- **Schéma cible** : nom des colonnes, types, ordres de grandeur / domaines de valeurs.
### 4.4 Documentation obligatoire du dataset
 
- Fichier de données exploitable (`.csv`) + éventuel fichier « brut » avant nettoyage.
- Scripts/notebooks de génération ou transformation.
- **Dictionnaire de données** (par colonne : description, type, unité, signification).
- Origine des données (public / simulé / mélange).
- Hypothèses de simulation (règles métier, simplifications).
- Limites connues (volume, biais, simplifications).
- Renommage suggéré au vocabulaire MECHA : `machine_id`, `usine`, `type_piece`, etc.
---
 
## 5. Volet Solution applicative intégrée
 
L'IA seule ne suffit pas : ses résultats doivent être **exploitables par les métiers** (maintenance, production, pilotage). À produire :
 
- **Architecture applicative** : composants, flux, interactions avec les modèles et les données.
- **Mécanismes d'intégration IA ↔ appli** : exposition via **API**, accès base de données, fichiers intermédiaires, services.
- **Paramétrage applicatif** :
    - règles de calcul / transformation des résultats IA,
    - **seuils** et déclenchement d'**alertes / notifications**,
    - **fréquence de mise à jour** des données.
- **Langage / formalisme de configuration** illustré : requêtes, scripts, règles métier, formats d'échange.
- **Interfaces / écrans métier** : tableaux de bord, vues synthétiques, indicateurs clés + usages attendus.
- 💡 L'IA générative est **autorisée** pour formaliser des éléments de conception (écrans, règles, scénarios) **à condition de fournir les prompts** et d'expliciter l'analyse / adaptation / intégration.
---
 
## 6. Validation & Déploiement
 
### Validation
 
- Décrire **comment** la validation de la solution IA a été organisée et réalisée.
- S'appuyer sur des **critères techniques ET fonctionnels** adaptés à MECHA.
### Déploiement multi-sites
 
- Contexte industriel multi-usines (disponibilité équipements, sécurité, **continuité de production**).
- **Intégration progressive** dans l'existant, cohérente avec la supervision/pilotage en place.
- Préciser les modalités de déploiement et l'exploitation par les équipes de maintenance.
- Identifier **limites et prérequis** d'un déploiement à l'échelle industrielle.
---
 
## 7. Feuille de route chronologique (26 h de préparation)
 
> Découpage indicatif à adapter. L'idée : avancer en parallèle (Data/IA d'un côté, Appli/Infra de l'autre) puis converger.
>
> **Légende des cases :** `[x]` terminé · `[~]` en cours · `[ ]` à faire.
 
### Étape 0 — Cadrage & organisation (≈ 2 h)
 
- [ ] Répartition des rôles dans l'équipe (Data, Backend/API, Infra/CI-CD, Doc/Soutenance).
- [x] Création du **repository Git** (structure de dossiers, README, branches).
- [ ] Rédaction du **guide d'entretien simulé** + synthèse des besoins métier (compétence 1).
- [x] Choix du cas d'usage précis et du schéma de données cible.
### Étape 1 — Données (≈ 4 h)
 
- [x] Sélection/génération du jeu de données + injection de pannes/dérives.
- [x] Scripts de transformation, nettoyage, variables dérivées.
- [x] Dictionnaire de données + documentation des hypothèses. _(dictionnaire par colonne `docs/dictionnaire_donnees.md` — 33 colonnes : identité, réglages, 21 capteurs C-MAPSS, cibles ; cohérence vérifiée par `tests/test_data_dictionary.py` ; hypothèses dans `docs/repartition_donnees.md`)_
### Étape 2 — Modélisation IA (≈ 6 h)
 
- [x] Préparation (split train/test, gestion valeurs manquantes, normalisation).
- [x] Baseline (régression / modèle simple).
- [x] Modèle principal (Random Forest) : classification état + prédiction RUL.
- [ ] (Optionnel) détection d'anomalies.
- [x] Évaluation : métriques, comparaison, **interprétation** des résultats.
  - _Réalisé aussi : XGBoost + LSTM (Deep Learning, GPU). Voir `plan.md` §8 et `docs/`._
### Étape 3 — Solution applicative & intégration (≈ 5 h)
 
- [x] Exposition du modèle (API REST, ex. FastAPI / Flask). _(FastAPI `backend/api/` : `/predict`, `/predict/batch`, `/health`, `/features` ; LSTM classif `at_risk` + régression `RUL`)_
- [x] Règles métier, seuils, logique d'alertes. _(seuil `at_risk` = proba ≥ 0.5 ; niveaux `ok`/`warning`/`critical` paramétrables dans `backend/api/inference.py`)_
- [x] Maquette d'écran(s) métier / tableau de bord (réel ou décrit + prompts IA générative). _(dashboard Streamlit `frontend/` finalisé : 4 écrans — Vue parc (KPI, priorisation RUL, alertes), Par ligne (agrégats), Fiche machine (trajectoires RUL/risque via `/predict/trajectory`), Analyse unitaire (`/predict`) ; thème pro clair + doc `docs/dashboard_*.md`)_
### Étape 4 — Infrastructure, tests, CI (≈ 4 h)
 
- [x] **Dockerfile** + (idéalement) `docker-compose`. _(`Tom/Dockerfile.backend` (API, torch CPU) + `Tom/Dockerfile.frontend` (Streamlit) + `Tom/docker-compose.yml` (backend healthcheck + frontend) ; modèles/CSV montés en volumes ; `.dockerignore`)_
- [~] Tests (unitaires / intégration) + plan de correction. _(20 tests pytest : intégration API (`test_api.py`), **unitaires** (`test_units.py` : règle d'alerte, fenêtre glissante, mise en forme requêtes), **dictionnaire** (`test_data_dictionary.py`) ; reste : plan de correction formalisé)_
- [x] Pipeline **CI minimal** (GitHub Actions / GitLab CI) : lancer les tests + builder l'image. _(CircleCI `.circleci/config.yml` : jobs `test-ia` (pytest) + `build-images` (docker compose build), en plus de block-merge & gitleaks)_
### Étape 5 — Documentation & conduite du changement (≈ 3 h)
 
- [ ] Documentation technique complète (cf. checklist §8).
- [~] **Guide utilisateur métier** (1–2 pages). _(guide de l'interface d'exploitation rédigé : `docs/dashboard_fonctionnel.md` — écrans, lecture des indicateurs, justification du seuil 30, règles d'alerte, limites ; reste à couvrir la solution au-delà du dashboard)_
- [ ] Plan de **conduite du changement** (acteurs, messages, formation, déploiement progressif).
- [ ] Volet RGPD (à décrire _comme si_ des données réelles étaient utilisées).
### Étape 6 — Soutenance (≈ 2 h)
 
- [ ] Support de présentation (public technique).
- [ ] Répétition + préparation aux 30 min d'entretien.
---
 
## 8. Checklist complète des livrables (à cocher)
 
### 8.1 Livrables techniques
 
- [x] **Code source** versionné sur Git : préparation des données, modélisation (scripts/notebooks), exploitation du modèle. _(prep/fusion `preAnalyse/` + `ml/`, notebooks `notebooks/`, exploitation `backend/` + `frontend/`)_
- [x] **Jeux de données** : données utilisées + données transformées + hypothèses de simulation/modification. _(scripts `fusionDatasetFinal.py`, `analyseGlobal.py` ; doc `docs/repartition_donnees.md`)_
- [x] **Modèles ML** : paramètres retenus, métriques de performance, résultats + interprétation. _(4 modèles ; docs `docs/modele_*.md`)_
- [x] **Conteneurisation** : configuration Docker + instructions d'exécution en environnement standardisé. _(2 Dockerfiles + `docker-compose.yml` : `docker compose up --build` → API `:8000` + dashboard `:8501` ; images code-only, données montées en volumes)_
- [x] **CI** : chaîne d'intégration continue minimale (fichier de config) lançant les tests + build de l'image. _(CircleCI : `test-ia` (pytest) + `build-images` (build des 2 images))_
- [x] **Description de la solution applicative** : mécanismes d'intégration, paramètres, règles métier, interfaces. _(formalisée dans `docs/dashboard_technique.md` (archi front↔API découplée, endpoints, fenêtre glissante) + `docs/dashboard_fonctionnel.md` (écrans, seuils, alertes) ; API REST `backend/` + dashboard `frontend/` en 4 écrans)_
### 8.2 Documentation technique (argumentée)
 
- [ ] **Schéma d'architecture** : sources, flux, composants de traitement/modélisation, restitution.
- [~] **Choix techniques** : outils/langages, structuration du code, gestion du code source. _(structuration du frontend documentée dans `docs/dashboard_technique.md` (package `ui/` : `screen/` / `style/` / `utils/`) ; stack globale + gestion du code source à formaliser)_
- [x] **Modèles étudiés** : critères de sélection + limites identifiées. _(`docs/modele_*.md`)_
- [ ] **Déploiement & exploitation** : prérequis techniques + perspectives d'industrialisation.
- [ ] **RGPD** : type de données, anonymisation/pseudonymisation, rôle du DPO (en hypothèse données réelles).
- [ ] **Compte-rendu client** : guide d'entretien simulé + synthèse des besoins métier.
- [~] **Guide utilisateur métier** (1–2 p.) : ce que fait la solution, comment lire les résultats, limites/conditions d'usage. _(rédigé pour l'interface d'exploitation : `docs/dashboard_fonctionnel.md`)_
- [ ] **Conduite du changement** : acteurs concernés, messages clés, formation/support, déploiement progressif.
### 8.3 Validation & développement
 
- [ ] Description de la **démarche de validation globale**.
- [ ] **Projection de mise en œuvre** dans les usines MECHA : conditions de déploiement, impacts techniques & organisationnels, points de vigilance pour le passage à l'échelle.
### 8.4 Soutenance
 
- [ ] **Support de présentation** synthétisant le travail (public technique).
- [ ] Préparation à l'**entretien collectif** (30 min de questions du jury).
> 📦 Les livrables peuvent être **regroupés dans un seul document** en dissociant clairement les parties. Tout doit être cohérent, professionnel, et rédigé dans une **logique d'aide à la décision**.
 
---
 
## 9. Stack technique suggérée (à justifier dans le rapport)
 
|Besoin|Proposition|Justification courte|
|---|---|---|
|Langage principal|**Python**|Standard data science, écosystème ML mature|
|Manipulation données|`pandas`, `numpy`|Imposés/suggérés par le sujet|
|Modélisation|`scikit-learn`|Random Forest, régression, métriques|
|Notebooks|Jupyter|Exploration + restitution des résultats|
|API / exposition modèle|**FastAPI** ou Flask|Exposer les prédictions aux métiers|
|Dashboard|Streamlit / Dash / Power BI|Tableaux de bord, alertes, KPI|
|Conteneurisation|**Docker** (+ docker-compose)|Imposé par le sujet|
|Versioning|**Git** (GitHub / GitLab)|Imposé par le sujet|
|CI|**GitHub Actions** ou **GitLab CI**|Imposé : tests + build image|
|Tests|`pytest`|Tests unitaires / intégration|
 
---
 
## 10. Pièges à éviter
 
- ❌ Utiliser des **données réelles** → interdit (confidentialité). Public ou simulé uniquement.
- ❌ Livrer un modèle **non interprété** → les métriques doivent être expliquées et reliées au métier.
- ❌ Oublier la **justification** des choix (archi, algo, stack) → c'est explicitement noté.
- ❌ Négliger l'**appli d'exploitation** → l'IA doit être _utilisable_ par la maintenance/production.
- ❌ Sauter la **CI** ou la **conteneurisation** → ce sont des livrables techniques obligatoires.
- ❌ Sous-estimer la **conduite du changement** et le **guide utilisateur** → compétences évaluées à part entière.
- ❌ Si IA générative utilisée → **toujours fournir les prompts** et expliquer l'intégration.
- ❌ Préparer la soutenance au dernier moment → elle compte pour 1/3 de l'évaluation.
---
 
## 11. Mémo soutenance
 
- **Durée :** 20 min de présentation + 30 min d'entretien collectif (chaque membre doit pouvoir répondre).
- **Public :** technique, mais ne connaît ni l'équipe ni la formation → contextualiser.
- **Objectif :** démontrer la **maîtrise technique** _et_ la **capacité à communiquer** auprès d'un client professionnel.
- **Fil rouge conseillé :** problème MECHA → cas d'usage → données → modèle → résultats → appli d'exploitation → déploiement → valeur métier.
---
 
_Rappel périmètre :_ aucun contact direct avec « l'entreprise ». Le cahier des charges est la seule expression officielle du besoin ; toute clarification passe par l'encadrant pédagogique (rôle du client).
