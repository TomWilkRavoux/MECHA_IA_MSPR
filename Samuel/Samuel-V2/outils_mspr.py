# -*- coding: utf-8 -*-
"""
Outils partagés pour les notebooks Samuel-V2 (MSPR MECHA — maintenance prédictive).

Pourquoi ce fichier : dans la V1 je copiais-collais le chargement, le score PHM08
et le calcul des métriques dans chaque notebook. Ici tout est centralisé :
- si je corrige un piège (ex : l'ordre alphabétique des machine_uid), c'est corrigé partout ;
- tous les modèles sont évalués EXACTEMENT pareil, donc comparables entre eux.

Conventions du projet (décidées dans la V1, gardées en V2) :
- Usine 1 = FD001 + FD003 (1 régime de fonctionnement — Option B, regroupement par régime)
- Usine 2 = FD002 + FD004 (6 régimes de fonctionnement)
- RUL plafonné à 125 cycles pour l'entraînement (au-delà, la dégradation est invisible)
- "À risque" = RUL <= 30 cycles
- random_state = 42 partout (reproductibilité pour la soutenance)
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Constantes du projet
# ---------------------------------------------------------------------------
DOSSIER = Path(__file__).resolve().parent          # dossier Samuel-V2/
DOSSIER_DATA = DOSSIER / "Data" / "csv"
DOSSIER_MODELS = DOSSIER / "models"
DOSSIER_FIGURES = DOSSIER / "figures"
DOSSIER_RESULTATS = DOSSIER / "resultats"
for d in (DOSSIER_MODELS, DOSSIER_FIGURES, DOSSIER_RESULTATS):
    d.mkdir(exist_ok=True)

SEED = 42                  # reproductibilité
RUL_MAX = 125              # plafonnement du RUL (convention C-MAPSS)
SEUIL_RISQUE = 30          # at_risk = 1 si RUL <= 30
PART_VALIDATION = 0.2      # 20 % des MACHINES mises de côté pour la validation
LONGUEUR_FENETRE = 30      # fenêtre glissante pour le LSTM (30 derniers cycles)

# Les 3 réglages qui définissent le régime de fonctionnement (utile Usine 2)
REGLAGES = ['Altitude / Mach', 'Throttle Resolver Angle (TRA)', 'Altitude pressurisée']

# Colonnes qui ne sont PAS des capteurs
COLONNES_IDENTITE = ['machine_id', 'cycle', 'usine', 'ligne_production',
                     'machine_uid', 'RUL', 'at_risk', 'regime']

# Une couleur FIXE par modèle dans toutes les figures du projet (palette Okabe-Ito,
# lisible par les daltoniens) : on reconnaît un modèle d'un graphe à l'autre.
COULEURS = {
    "baseline": "#999999",   # gris : le plancher de référence
    "rf":       "#0072B2",   # bleu
    "hgb":      "#E69F00",   # orange
    "lstm":     "#009E73",   # vert
}

# Palette pour les catégories SANS identité fixe dans le projet (ex : les 6
# régimes de l'Usine 2). Même famille Okabe-Ito que COULEURS, ordre fixe.
PALETTE_CATEGORIES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
def charger_usine(num_usine):
    """Charge le train / test / RUL vrai d'une usine (1 ou 2).

    Au passage :
    - re-plafonne le RUL du train à 125 (au cas où le csv ne le serait pas) ;
    - ajoute la colonne `at_risk` (RUL <= 30) au train → cible de la classification.
    """
    train = pd.read_csv(DOSSIER_DATA / f"train_usine{num_usine}.csv")
    test = pd.read_csv(DOSSIER_DATA / f"test_usine{num_usine}.csv")
    rul = pd.read_csv(DOSSIER_DATA / f"rul_usine{num_usine}.csv")

    train["RUL"] = train["RUL"].clip(upper=RUL_MAX)
    train["at_risk"] = (train["RUL"] <= SEUIL_RISQUE).astype(int)
    return train, test, rul


def capteurs_de(df):
    """Liste des colonnes capteurs (tout sauf identité, cibles et réglages).

    Les 3 réglages ne sont pas des capteurs d'usure : ils définissent le régime
    de fonctionnement (ils servent à `ajouter_regime`, pas comme features).
    """
    return [c for c in df.columns if c not in COLONNES_IDENTITE and c not in REGLAGES]


def charger_features(num_usine):
    """Recharge la liste de features décidée dans le notebook 01.

    Les features sont choisies UNE FOIS (corrélation avec le RUL, notebook 01)
    et tous les modèles rechargent la même liste → comparaison à armes égales.
    """
    fichier = DOSSIER_RESULTATS / f"features_usine{num_usine}.json"
    with open(fichier, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Split train / validation PAR MACHINE (anti-fuite)
# ---------------------------------------------------------------------------
def separer_machines(train, part_validation=PART_VALIDATION):
    """Sépare le train en apprentissage / validation en coupant PAR MACHINE.

    Piège corrigé de la V1 : si on coupe par lignes (ex : validation_fraction du HGB),
    des cycles d'une même machine se retrouvent des deux côtés. Comme deux cycles
    voisins se ressemblent énormément, le modèle "reconnaît" la machine au lieu de
    généraliser → score de validation trop optimiste (fuite de données).
    Ici une machine est soit 100 % en apprentissage, soit 100 % en validation.
    """
    machines = sorted(train["machine_uid"].unique())
    machines_app, machines_val = train_test_split(
        machines, test_size=part_validation, random_state=SEED)
    df_app = train[train["machine_uid"].isin(machines_app)].copy()
    df_val = train[train["machine_uid"].isin(machines_val)].copy()
    return df_app, df_val


# ---------------------------------------------------------------------------
# Régimes de fonctionnement (Usine 2)
# ---------------------------------------------------------------------------
def ajouter_regime(df):
    """Identifie le régime de fonctionnement de chaque cycle (Usine 2).

    On arrondit les 3 réglages pour gommer le bruit de mesure : chaque combinaison
    arrondie = 1 régime. Sur l'Usine 2 on doit retrouver exactement 6 régimes.
    (Méthode mise au point dans la V1, notebook Random_Forest.)
    """
    df = df.copy()
    arrondi = df[REGLAGES].round({'Altitude / Mach': 0,
                                  'Throttle Resolver Angle (TRA)': 2,
                                  'Altitude pressurisée': 0})
    df["regime"] = arrondi.apply(tuple, axis=1)
    return df


def calculer_stats_regime(train, capteurs):
    """Moyenne et écart-type de chaque capteur DANS chaque régime (train seulement)."""
    moyennes = train.groupby("regime")[capteurs].mean()
    ecarts = train.groupby("regime")[capteurs].std().replace(0, 1)  # évite /0
    return moyennes, ecarts


def normaliser_par_regime(df, moyennes, ecarts, capteurs):
    """Centre-réduit chaque capteur PAR régime (stats calculées sur le train).

    Pourquoi : sur l'Usine 2, un même capteur saute entre 6 plages de valeurs selon
    le régime → la dérive d'usure est noyée dans le "bruit de régime". En normalisant
    par régime, on ne garde que l'écart à la normale DE CE régime = le vrai signal
    de dégradation. (C'est le "conditioning-based normalization" de la littérature
    C-MAPSS — je l'avais retrouvé par moi-même en V1.)
    """
    df_norm = df.copy()
    df_norm[capteurs] = (df[capteurs].values
                         - moyennes.loc[df["regime"]].values) / ecarts.loc[df["regime"]].values
    return df_norm


# ---------------------------------------------------------------------------
# Vérité terrain du test (avec le piège d'ordre corrigé)
# ---------------------------------------------------------------------------
def dernier_cycle_test(test):
    """Le dernier cycle observé de chaque machine du test (là où on doit prédire)."""
    return test.groupby("machine_uid").last().reset_index()


def verite_terrain(test, rul):
    """RUL vrai + at_risk vrai, ALIGNÉS sur l'ordre du groupby du test.

    Piège découvert en V1 : groupby('machine_uid') trie en ALPHABÉTIQUE
    ('10_1_1' avant '2_1_1') alors que le fichier RUL est en ordre numérique.
    Sans ré-alignement le R² devient négatif alors que le modèle est bon.
    """
    ordre = test.groupby("machine_uid").last().index
    y_rul = rul.set_index("machine_uid").loc[ordre, "RUL"]
    y_risk = (y_rul <= SEUIL_RISQUE).astype(int)
    return y_rul, y_risk


# ---------------------------------------------------------------------------
# Préparation complète d'une usine (le pipeline validé dans le notebook 01)
# ---------------------------------------------------------------------------
def preparer_usine(num_usine):
    """Enchaîne toute la préparation du notebook 01, à l'identique pour tous les modèles :

    1. chargement (RUL plafonné, at_risk ajouté) ;
    2. Usine 2 : identification des 6 régimes puis normalisation par régime
       (stats calculées sur l'APPRENTISSAGE seulement — anti-fuite) ;
    3. split apprentissage/validation PAR MACHINE (80/20, graine 42) ;
    4. vérité terrain du test alignée (piège d'ordre corrigé) ;
    5. rechargement de la liste de features décidée dans le notebook 01.

    Renvoie : (App, Val, Test, y_rul_vrai, y_risk_vrai, Features)
    """
    train, test, rul = charger_usine(num_usine)
    if num_usine == 2:
        train = ajouter_regime(train)
        test = ajouter_regime(test)
    app, val = separer_machines(train)
    if num_usine == 2:
        capteurs = capteurs_de(train)
        moyennes, ecarts = calculer_stats_regime(app, capteurs)
        app = normaliser_par_regime(app, moyennes, ecarts, capteurs)
        val = normaliser_par_regime(val, moyennes, ecarts, capteurs)
        test = normaliser_par_regime(test, moyennes, ecarts, capteurs)
    y_rul_vrai, y_risk_vrai = verite_terrain(test, rul)
    features = charger_features(num_usine)
    return app, val, test, y_rul_vrai, y_risk_vrai, features


# ---------------------------------------------------------------------------
# Scores PHM08 / NASA
# ---------------------------------------------------------------------------
def phm_score_sur(rul_true, rul_pred):
    """Pénalité des prédictions PRUDENTES (modèle prédit moins de vie que la réalité).

    d < 0 → on ferait la maintenance trop tôt : ça coûte, mais c'est "sûr".
    """
    d = np.asarray(rul_pred) - np.asarray(rul_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, 0)))


def phm_score_dang(rul_true, rul_pred):
    """Pénalité des prédictions DANGEREUSES (modèle surestime la durée de vie).

    d > 0 → on croit avoir de la marge alors que la machine va lâcher : panne
    non planifiée, le pire cas pour MECHA. C'est pour ça que l'exponentielle
    est plus raide (d/10 contre d/13 côté sûr).
    """
    d = np.asarray(rul_pred) - np.asarray(rul_true)
    return float(np.sum(np.where(d > 0, np.exp(d / 10) - 1, 0)))


def score_nasa(rul_true, rul_pred):
    """Score NASA C-MAPSS complet = sûr + dangereux (plus bas = mieux).

    C'est la somme de mes deux PHM08 : ça permet de comparer directement
    avec la littérature C-MAPSS.
    """
    return phm_score_sur(rul_true, rul_pred) + phm_score_dang(rul_true, rul_pred)


# ---------------------------------------------------------------------------
# Évaluation (affichage + dictionnaire de métriques)
# ---------------------------------------------------------------------------
def resultats_regression(y_true, y_pred, afficher=True):
    """Toutes les métriques de régression RUL, dans un dictionnaire."""
    nb_machines = len(y_true)
    met = {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "PHM_sur": phm_score_sur(y_true, y_pred),
        "PHM_dangereux": phm_score_dang(y_true, y_pred),
        "NASA_total": score_nasa(y_true, y_pred),
        "nb_machines": nb_machines,
    }
    # NASA moyen par machine : le total dépend du nombre de machines (200 en U1,
    # 507 en U2), la moyenne permet de comparer les usines entre elles.
    met["NASA_moyen"] = met["NASA_total"] / nb_machines
    if afficher:
        print(" ---------- Resultat régression ----------")
        print(f"R² : {met['R2']:.4f}")
        print(f"MAE (erreur moyenne) : {met['MAE']:.2f} cycles")
        print(f"RMSE (erreur écart-type) : {met['RMSE']:.2f} cycles")
        print(f"PHM08 Sûr (prédit trop tôt) : {met['PHM_sur']:.0f}")
        print(f"PHM08 Dangereux (prédit trop tard) : {met['PHM_dangereux']:.0f}")
        print(f"Score NASA total : {met['NASA_total']:.0f}  "
              f"(moyen par machine : {met['NASA_moyen']:.2f})")
    return met


def resultats_classification(y_true, y_pred, y_proba=None, afficher=True):
    """Toutes les métriques de classification at_risk, dans un dictionnaire."""
    met = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "nb_machines": len(y_true),
    }
    if y_proba is not None:
        met["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    if afficher:
        print(" ---------- Resultat classification ----------")
        print(f"Accuracy : {met['accuracy']:.3f}")
        print(f"Precision (alertes justifiées) : {met['precision']:.3f}")
        print(f"Recall (machines à risque détectées) : {met['recall']:.3f}")
        print(f"F1 : {met['f1']:.3f}")
        if y_proba is not None:
            print(f"ROC-AUC : {met['roc_auc']:.3f}")
    return met


# ---------------------------------------------------------------------------
# Sauvegarde des résultats (pour la synthèse et la soutenance)
# ---------------------------------------------------------------------------
FICHIER_METRIQUES = DOSSIER_RESULTATS / "metriques.json"


def sauver_metriques(modele, tache, num_usine, metriques):
    """Empile les métriques dans resultats/metriques.json (relu par la synthèse)."""
    if FICHIER_METRIQUES.exists():
        with open(FICHIER_METRIQUES, encoding="utf-8") as f:
            tout = json.load(f)
    else:
        tout = {}
    tout.setdefault(modele, {}).setdefault(tache, {})[f"usine{num_usine}"] = metriques
    with open(FICHIER_METRIQUES, "w", encoding="utf-8") as f:
        json.dump(tout, f, indent=2, ensure_ascii=False)


def sauver_predictions(modele, tache, num_usine, machine_uid, y_true, y_pred, y_proba=None):
    """Sauve les prédictions machine par machine.

    Indispensable pour la synthèse : recombiner U1 + U2 et calculer des métriques
    GLOBALES sur les 707 machines du parc complet.
    """
    df = pd.DataFrame({"machine_uid": machine_uid,
                       "y_true": np.asarray(y_true),
                       "y_pred": np.asarray(y_pred)})
    if y_proba is not None:
        df["y_proba"] = np.asarray(y_proba)
    fichier = DOSSIER_RESULTATS / f"pred_{modele}_{tache}_usine{num_usine}.csv"
    df.to_csv(fichier, index=False)


def charger_predictions(modele, tache, num_usine):
    """Relit les prédictions sauvées (utilisé par le notebook de synthèse)."""
    fichier = DOSSIER_RESULTATS / f"pred_{modele}_{tache}_usine{num_usine}.csv"
    return pd.read_csv(fichier)


# ---------------------------------------------------------------------------
# Graphiques (partagés par tous les notebooks, sauvés dans figures/)
# ---------------------------------------------------------------------------
# Règles communes : une couleur FIXE par modèle (COULEURS), légende dès qu'il y a
# deux séries, axes discrets, et chaque figure est sauvée en PNG dans figures/
# pour le rapport et la soutenance.

def _preparer_axes(ax, titre, xlabel, ylabel):
    """Mise en forme commune : grille discrète, pas de cadre inutile."""
    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)                    # grille DERRIÈRE les données
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _sauver_figure(fig, nom_fichier):
    """Sauve dans figures/ (300 dpi pour le rapport) puis affiche dans le notebook."""
    chemin = DOSSIER_FIGURES / nom_fichier
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Figure sauvée : figures/{chemin.name}")


def tracer_split_machines(app, val, num_usine):
    """Vérifie visuellement le split par machine de `separer_machines`.

    Deux histogrammes superposés des durées de vie : si le tirage est sain, les
    machines de validation couvrent les mêmes durées de vie que celles
    d'apprentissage (pas toutes les machines "longues" du même côté). Et comme
    chaque machine n'apparaît que dans UN histogramme, on voit le principe
    anti-fuite : une machine est à 100 % d'un côté ou de l'autre.
    """
    vies_app = app.groupby("machine_uid")["cycle"].max()
    vies_val = val.groupby("machine_uid")["cycle"].max()
    bins = np.histogram_bin_edges(np.concatenate([vies_app, vies_val]), bins=25)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vies_app, bins=bins, color="#0072B2", alpha=0.75,
            label=f"Apprentissage ({len(vies_app)} machines)")
    ax.hist(vies_val, bins=bins, color="#E69F00", alpha=0.75,
            label=f"Validation ({len(vies_val)} machines)")
    _preparer_axes(ax, f"Usine {num_usine} — split 80/20 par machine (anti-fuite)",
                   "Durée de vie de la machine (cycles)", "Nombre de machines")
    ax.legend()
    _sauver_figure(fig, f"split_machines_usine{num_usine}.png")


def tracer_score_nasa():
    """Montre POURQUOI le score NASA est asymétrique.

    Même erreur de 30 cycles : prédire trop tôt coûte ~9 points (maintenance
    anticipée), prédire trop tard en coûte ~19 (panne non planifiée, le pire
    cas pour MECHA). C'est l'argument visuel qui justifie de suivre PHM_sur et
    PHM_dangereux séparément.
    """
    d = np.linspace(-40, 40, 801)
    penalite = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d[d <= 0], penalite[d <= 0], color="#0072B2", linewidth=2,
            label="Prudent — maintenance trop tôt (exp(-d/13))")
    ax.plot(d[d >= 0], penalite[d >= 0], color="#D55E00", linewidth=2,
            label="Dangereux — panne non planifiée (exp(d/10))")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.annotate("9", (-30, np.exp(30 / 13) - 1), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)
    ax.annotate("19", (30, np.exp(30 / 10) - 1), textcoords="offset points",
                xytext=(-6, 8), ha="right", fontsize=9)
    _preparer_axes(ax, "Score NASA : la même erreur ne coûte pas le même prix",
                   "Erreur de prédiction (RUL prédit − RUL réel, cycles)",
                   "Pénalité pour une machine")
    ax.legend()
    _sauver_figure(fig, "score_nasa.png")


def tracer_predictions_rul(y_true, y_pred, nom_modele, num_usine):
    """Nuage RUL réel / RUL prédit : chaque point = une machine du test.

    La diagonale = prédiction parfaite. Au-dessus, le modèle surestime la durée
    de vie (zone dangereuse du score NASA) ; en dessous il est prudent.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = RUL_MAX * 1.08
    ax.plot([0, lim], [0, lim], color="#999999", linewidth=1, linestyle="--")
    ax.scatter(y_true, y_pred, s=22, alpha=0.55, edgecolors="none",
               color=COULEURS.get(nom_modele, "#0072B2"))
    # étiquette directe sur la diagonale plutôt qu'une boîte de légende
    # (la boîte cacherait les points en haut à droite)
    ax.text(lim * 0.57, lim * 0.58, "prédiction parfaite", rotation=45,
            rotation_mode="anchor", ha="center", fontsize=8, color="#777777")
    ax.text(0.03, 0.97, "dangereux\n(surestime la vie)", transform=ax.transAxes,
            va="top", fontsize=9, color="#D55E00")
    ax.text(0.97, 0.03, "prudent\n(prédit trop tôt)", transform=ax.transAxes,
            ha="right", fontsize=9, color="#0072B2")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    _preparer_axes(ax, f"{nom_modele} — Usine {num_usine} : RUL prédit vs réel\n"
                       f"({len(y_true)} machines du test)",
                   "RUL réel (cycles)", "RUL prédit (cycles)")
    _sauver_figure(fig, f"pred_rul_{nom_modele}_usine{num_usine}.png")


def tracer_matrice_confusion(y_true, y_pred, nom_modele, num_usine):
    """Matrice de confusion at_risk, avec les effectifs dans chaque case.

    La case critique pour MECHA : réel "À risque" / prédit "Sain" (en bas à
    gauche) = machine qui va lâcher et qu'on n'a pas vue venir.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    noms = ["Sain", "À risque"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")                # séquentiel : + foncé = + de machines
    ax.set_xticks([0, 1], noms)
    ax.set_yticks([0, 1], noms)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    seuil = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > seuil else "#1a1a1a")
    ax.set_title(f"{nom_modele} — Usine {num_usine} : matrice de confusion")
    _sauver_figure(fig, f"confusion_{nom_modele}_usine{num_usine}.png")


def tracer_regimes(df, num_usine=2):
    """Justifie la normalisation par régime : les cycles se groupent en paquets.

    Chaque point = un cycle, placé selon deux des trois réglages ; la couleur =
    le régime détecté par `ajouter_regime`. Six paquets nets et séparés = la
    normalisation par régime a un sens physique (ce ne sont pas des clusters
    artificiels).
    """
    if "regime" not in df.columns:
        df = ajouter_regime(df)
    x, y = REGLAGES[0], REGLAGES[1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (regime, groupe) in enumerate(sorted(df.groupby("regime"))):
        ax.scatter(groupe[x], groupe[y], s=8, alpha=0.5, edgecolors="none",
                   color=PALETTE_CATEGORIES[i % len(PALETTE_CATEGORIES)],
                   label=f"Régime {i + 1} ({len(groupe)} cycles)")
    _preparer_axes(ax, f"Usine {num_usine} — régimes de fonctionnement détectés", x, y)
    ax.legend(markerscale=2, fontsize=8)
    _sauver_figure(fig, f"regimes_usine{num_usine}.png")


def tracer_normalisation_capteur(avant, apres, capteur, machine_uid):
    """L'avant / après de `normaliser_par_regime`, sur une machine de l'Usine 2.

    À gauche le capteur brut : il saute entre les plages de valeurs des régimes,
    la dérive d'usure est noyée. À droite le même capteur normalisé par régime :
    la tendance de dégradation devient visible.
    """
    brut = avant[avant["machine_uid"] == machine_uid].sort_values("cycle")
    norme = apres[apres["machine_uid"] == machine_uid].sort_values("cycle")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(brut["cycle"], brut[capteur], color="#999999", linewidth=1)
    _preparer_axes(ax1, "Brut : le régime masque l'usure", "Cycle", capteur)
    ax2.plot(norme["cycle"], norme[capteur], color="#0072B2", linewidth=1)
    _preparer_axes(ax2, "Normalisé par régime : l'usure apparaît",
                   "Cycle", f"{capteur} (centré-réduit)")
    fig.suptitle(f"Machine {machine_uid} — capteur « {capteur} »")
    nom_capteur = "".join(c if c.isalnum() else "_" for c in capteur)
    _sauver_figure(fig, f"normalisation_{nom_capteur}_{machine_uid}.png")


def tracer_trajectoire_rul(test, machine_uid, scaler, modele, features,
                           rul_final, nom_modele, num_usine):
    """Suit UNE machine du test cycle après cycle : RUL prédit vs RUL réel.

    Le RUL réel se reconstruit en remontant depuis la fin : au dernier cycle
    observé il vaut `rul_final` (fichier RUL), et il augmente de 1 à chaque
    cycle en arrière (plafonné à 125, comme la cible d'entraînement). Très
    parlant en soutenance : on voit la prédiction s'affiner à l'approche de
    la panne, là où elle compte vraiment.
    """
    groupe = test[test["machine_uid"] == machine_uid].sort_values("cycle")
    y_pred = np.clip(modele.predict(scaler.transform(groupe[features])), 0, RUL_MAX)
    rul_reel = np.clip(rul_final + groupe["cycle"].max() - groupe["cycle"],
                       None, RUL_MAX)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(groupe["cycle"], rul_reel, color="#999999", linewidth=2,
            label="RUL réel")
    ax.plot(groupe["cycle"], y_pred, linewidth=1.5,
            color=COULEURS.get(nom_modele, "#0072B2"),
            label=f"RUL prédit ({nom_modele})")
    ax.axhline(SEUIL_RISQUE, color="#D55E00", linewidth=1, linestyle=":",
               label=f"Seuil à risque ({SEUIL_RISQUE} cycles)")
    _preparer_axes(ax, f"Machine {machine_uid} — prédiction le long de la vie",
                   "Cycle", "RUL (cycles)")
    ax.legend()
    _sauver_figure(fig, f"trajectoire_{nom_modele}_usine{num_usine}_{machine_uid}.png")


# ---------------------------------------------------------------------------
# Évaluation finale standardisée des modèles tabulaires
# ---------------------------------------------------------------------------
def evaluer_et_sauver(nom_modele, num_usine, scaler, modele_rul, modele_risk, features,
                      test, y_rul_vrai, y_risk_vrai):
    """Évalue les 2 modèles d'une usine sur le test officiel + sauvegarde tout.

    Centralisé ici pour que la procédure soit STRICTEMENT identique dans les
    notebooks 02 (baseline), 03 (Random Forest) et 04 (HGB) :
    - prédiction au dernier cycle observé de chaque machine (comme en V1) ;
    - RUL borné à [0, 125] : un RUL négatif n'a pas de sens physique, et le modèle
      est entraîné sur une cible plafonnée à 125 (prédire au-delà = extrapolation) ;
    - métriques + prédictions + modèle sauvegardés pour la synthèse (06).

    Renvoie les prédictions de classification (pour les matrices de confusion).
    """
    dernier = test.groupby("machine_uid").last()
    X_test = scaler.transform(dernier[features])

    # --- Régression RUL ---
    y_pred_rul = np.clip(modele_rul.predict(X_test), 0, RUL_MAX)
    print(f"\n===== Usine {num_usine} — Régression RUL =====")
    met_rul = resultats_regression(y_rul_vrai, y_pred_rul)
    sauver_metriques(nom_modele, "regression", num_usine, met_rul)
    sauver_predictions(nom_modele, "regression", num_usine,
                       dernier.index, y_rul_vrai, y_pred_rul)
    tracer_predictions_rul(y_rul_vrai, y_pred_rul, nom_modele, num_usine)

    # --- Classification at_risk ---
    y_proba = modele_risk.predict_proba(X_test)[:, 1]
    y_pred_risk = modele_risk.predict(X_test)
    print(f"\n===== Usine {num_usine} — Classification at_risk =====")
    met_risk = resultats_classification(y_risk_vrai, y_pred_risk, y_proba)
    sauver_metriques(nom_modele, "classification", num_usine, met_risk)
    sauver_predictions(nom_modele, "classification", num_usine,
                       dernier.index, y_risk_vrai, y_pred_risk, y_proba)
    tracer_matrice_confusion(y_risk_vrai, y_pred_risk, nom_modele, num_usine)

    # --- Sauvegarde des modèles (rejouables sans réentraîner) ---
    joblib.dump({"scaler": scaler, "modele": modele_rul, "features": features},
                DOSSIER_MODELS / f"{nom_modele}_regression_usine{num_usine}.joblib")
    joblib.dump({"scaler": scaler, "modele": modele_risk, "features": features},
                DOSSIER_MODELS / f"{nom_modele}_classification_usine{num_usine}.joblib")
    return y_pred_risk


# ---------------------------------------------------------------------------
# Fenêtres glissantes pour le LSTM
# ---------------------------------------------------------------------------
def construire_fenetres(df, features, cible, longueur=LONGUEUR_FENETRE, pas=1):
    """Découpe chaque machine en fenêtres glissantes de `longueur` cycles.

    Pour chaque cycle t d'une machine, la fenêtre = les `longueur` cycles qui se
    terminent en t, et la cible = celle du cycle t (comme les modèles tabulaires,
    mais avec l'historique en plus). Les débuts de trajectoire trop courts sont
    complétés en répétant le premier cycle (left-padding) : une machine neuve
    "avant son premier cycle" ressemble à son premier cycle.

    `pas` : on peut ne garder qu'une fenêtre sur `pas` (deux fenêtres voisines se
    recouvrent à 29/30 cycles, elles sont quasi identiques → sous-échantillonner
    l'entraînement fait gagner du temps CPU sans perdre d'information). Le dernier
    cycle de chaque machine est toujours gardé.
    """
    X_liste, y_liste = [], []
    for _, groupe in df.groupby("machine_uid", sort=False):
        valeurs = groupe[features].to_numpy(dtype=np.float32)
        cibles = groupe[cible].to_numpy(dtype=np.float32)
        n = len(groupe)
        # padding : on répète le 1er cycle pour pouvoir prédire dès le début
        tampon = np.vstack([np.repeat(valeurs[:1], longueur - 1, axis=0), valeurs])
        indices = list(range(0, n, pas))
        if (n - 1) not in indices:          # toujours garder la fin de trajectoire
            indices.append(n - 1)
        for t in indices:
            X_liste.append(tampon[t:t + longueur])
            y_liste.append(cibles[t])
    return np.stack(X_liste), np.array(y_liste, dtype=np.float32)


def fenetres_test(test, features, longueur=LONGUEUR_FENETRE):
    """Une fenêtre par machine du test : ses `longueur` DERNIERS cycles observés.

    Renvoie aussi la liste des machine_uid dans l'ordre du groupby (le même ordre
    que verite_terrain, pour ne pas retomber dans le piège d'alignement de la V1).
    """
    X_liste, uids = [], []
    for uid, groupe in test.groupby("machine_uid", sort=True):
        valeurs = groupe[features].to_numpy(dtype=np.float32)
        if len(valeurs) < longueur:
            valeurs = np.vstack([np.repeat(valeurs[:1], longueur - len(valeurs), axis=0),
                                 valeurs])
        X_liste.append(valeurs[-longueur:])
        uids.append(uid)
    return np.stack(X_liste), uids
