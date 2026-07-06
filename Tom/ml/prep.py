"""Préparation partagée des données pour les notebooks IA MECHA (dataset C-MAPSS).

Ce module centralise le chargement, le split par machine (anti-fuite), la
normalisation, le clipping du RUL et la construction des fenêtres glissantes
pour le LSTM. Il est importé par les 3 notebooks (`01_random_forest`,
`02_gradient_boosting_baseline`, `03_lstm`) afin d'éviter toute duplication.

Convention métier :
- `at_risk = 1 si RUL <= 30` (seuil de risque, cf. fusionDatasetFinal.py).
- Une machine = une trajectoire run-to-failure -> on splitte PAR MACHINE.
- RUL borné (piecewise) à `RUL_CAP` : au-delà, la dégradation n'est pas encore
  observable et un RUL non borné déstabilise l'apprentissage.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

# --- Chemins (robustes au répertoire courant, relatifs à Tom/) ---
ROOT = Path(__file__).resolve().parent.parent  # dossier Tom/
DATA_DIR = ROOT / "assets" / "KaggleDataset"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Colonnes ---
KEYS = ["machine_id", "subset", "usine", "ligne_production", "unit", "cycle"]
SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]
FEATURES = SETTINGS + SENSORS

# --- Hyperparamètres métier ---
RISK_THRESHOLD = 30   # at_risk = 1 si RUL <= 30
RUL_CAP = 125         # clipping piecewise du RUL (convention C-MAPSS)
SEQ_LEN = 30          # longueur de fenêtre pour le LSTM


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load_train_classification() -> pl.DataFrame:
    """Jeu d'entraînement classification (contient `at_risk` et `RUL`)."""
    return pl.read_csv(DATA_DIR / "mecha_train_classification.csv")


def load_train_rul() -> pl.DataFrame:
    """Jeu d'entraînement régression (cible `RUL`)."""
    return pl.read_csv(DATA_DIR / "mecha_train_rul.csv")


def load_test() -> pl.DataFrame:
    """Jeu de test (features uniquement, machines tronquées, aucun label)."""
    return pl.read_csv(DATA_DIR / "mecha_test_classification.csv")


def load_rul_true() -> pl.DataFrame:
    """Vérité terrain : RUL au dernier cycle observé de chaque machine test."""
    return pl.read_csv(DATA_DIR / "mecha_rul_true.csv")


# ----------------------------------------------------------------------------
# Transformations
# ----------------------------------------------------------------------------
def clip_rul(y: np.ndarray, cap: int = RUL_CAP) -> np.ndarray:
    """Clipping piecewise-linéaire du RUL."""
    return np.minimum(np.asarray(y, dtype=float), cap)


def add_at_risk(df: pl.DataFrame, threshold: int = RISK_THRESHOLD) -> pl.DataFrame:
    """Dérive la colonne `at_risk` à partir du RUL (utile pour le jeu de test)."""
    return df.with_columns((pl.col("RUL") <= threshold).cast(pl.Int8).alias("at_risk"))


def split_by_machine(
    df: pl.DataFrame, val_frac: float = 0.2, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split train/validation PAR machine (aucune machine à cheval -> pas de fuite)."""
    machines = df["machine_id"].unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(machines)
    n_val = int(len(machines) * val_frac)
    val_ids = set(machines[:n_val])
    train = df.filter(~pl.col("machine_id").is_in(val_ids))
    val = df.filter(pl.col("machine_id").is_in(val_ids))
    return train, val


def to_xy(df: pl.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Extrait la matrice de features et le vecteur cible (numpy)."""
    x = df.select(FEATURES).to_numpy()
    y = df[target].to_numpy()
    return x, y


# ----------------------------------------------------------------------------
# Normalisation (StandardScaler ajusté sur le TRAIN uniquement)
# ----------------------------------------------------------------------------
def fit_scaler(x_train: np.ndarray, save_as: str = "scaler.joblib") -> StandardScaler:
    """Ajuste un StandardScaler sur le train et le sérialise dans models/."""
    scaler = StandardScaler().fit(x_train)
    if save_as:
        joblib.dump(scaler, MODELS_DIR / save_as)
    return scaler


def load_scaler(name: str = "scaler.joblib") -> StandardScaler:
    return joblib.load(MODELS_DIR / name)


# ----------------------------------------------------------------------------
# Fenêtres glissantes (LSTM)
# ----------------------------------------------------------------------------
def make_windows(
    df: pl.DataFrame,
    target: str,
    seq_len: int = SEQ_LEN,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les fenêtres glissantes par machine pour l'entraînement.

    Retour : X de forme (n_windows, seq_len, n_features) et y aligné sur le
    DERNIER cycle de chaque fenêtre. Les machines trop courtes sont left-paddées
    en répétant leur premier cycle.
    """
    xs, ys = [], []
    for g in df.partition_by("machine_id"):
        g = g.sort("cycle")
        arr = g.select(FEATURES).to_numpy()
        tgt = g[target].to_numpy()
        if scaler is not None:
            arr = scaler.transform(arr)
        n = len(arr)
        if n < seq_len:  # left-pad
            pad = np.repeat(arr[:1], seq_len - n, axis=0)
            arr = np.vstack([pad, arr])
            tgt = np.concatenate([np.repeat(tgt[:1], seq_len - n), tgt])
            n = seq_len
        for i in range(seq_len, n + 1):
            xs.append(arr[i - seq_len : i])
            ys.append(tgt[i - 1])
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="float32")


def make_test_windows(
    df_test: pl.DataFrame,
    seq_len: int = SEQ_LEN,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, list]:
    """Une fenêtre par machine test : les `seq_len` derniers cycles observés.

    Retour : X (n_machines, seq_len, n_features) et la liste des machine_id
    (dans le même ordre) pour la jointure avec la vérité terrain RUL.
    """
    xs, ids = [], []
    for g in df_test.partition_by("machine_id"):
        g = g.sort("cycle")
        arr = g.select(FEATURES).to_numpy()
        if scaler is not None:
            arr = scaler.transform(arr)
        if len(arr) < seq_len:
            pad = np.repeat(arr[:1], seq_len - len(arr), axis=0)
            arr = np.vstack([pad, arr])
        xs.append(arr[-seq_len:])
        ids.append(g["machine_id"][0])
    return np.asarray(xs, dtype="float32"), ids


# ----------------------------------------------------------------------------
# Jeu de test « officiel » (protocole C-MAPSS)
# ----------------------------------------------------------------------------
def test_last_cycle_eval() -> pl.DataFrame:
    """Dernier cycle de chaque machine test + vérité terrain RUL et at_risk.

    Utilisé pour l'évaluation tabulaire (RF, XGBoost, baseline). Colonnes de
    sortie : features + `RUL_true` + `at_risk_true`.
    """
    test = load_test()
    rul_true = load_rul_true().select(["machine_id", "RUL_true"])
    last = (
        test.sort("cycle")
        .group_by("machine_id", maintain_order=True)
        .last()
    )
    merged = last.join(rul_true, on="machine_id", how="inner")
    return merged.with_columns(
        (pl.col("RUL_true") <= RISK_THRESHOLD).cast(pl.Int8).alias("at_risk_true")
    )
