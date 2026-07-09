"""Tests unitaires des transformations de préparation (ml.prep).

Fonctions pures uniquement : clipping, dérivation du label, split anti-fuite,
extraction X/y, normalisation et fenêtres glissantes. Aucun accès disque
(les CSV du dataset ne sont pas requis) : les DataFrames sont construits en mémoire.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml import prep


def _labelled_df(machines, cycles_per_machine, seed=0):
    """Construit un DataFrame train synthétique avec toutes les FEATURES + RUL."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in machines:
        for c in range(1, cycles_per_machine + 1):
            row = {"machine_id": m, "cycle": c, "RUL": cycles_per_machine - c}
            for f in prep.FEATURES:
                row[f] = float(rng.normal())
            rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# clip_rul
# ---------------------------------------------------------------------------
def test_clip_rul_caps_values():
    out = prep.clip_rul([10, 200, 125, 300], cap=125)
    assert out.tolist() == [10.0, 125.0, 125.0, 125.0]


def test_clip_rul_default_cap():
    out = prep.clip_rul([prep.RUL_CAP + 50])
    assert out[0] == prep.RUL_CAP


# ---------------------------------------------------------------------------
# add_at_risk
# ---------------------------------------------------------------------------
def test_add_at_risk_threshold():
    df = pl.DataFrame({"RUL": [10, 30, 31, 200]})
    out = prep.add_at_risk(df, threshold=30)
    assert out["at_risk"].to_list() == [1, 1, 0, 0]  # <= 30 -> à risque


# ---------------------------------------------------------------------------
# split_by_machine
# ---------------------------------------------------------------------------
def test_split_by_machine_no_leakage_and_proportion():
    df = _labelled_df([f"M{i}" for i in range(10)], cycles_per_machine=3)
    train, val = prep.split_by_machine(df, val_frac=0.2, seed=42)

    train_ids = set(train["machine_id"].to_list())
    val_ids = set(val["machine_id"].to_list())
    assert train_ids.isdisjoint(val_ids)  # aucune machine à cheval
    assert len(val_ids) == 2  # 20% de 10 machines
    assert train_ids | val_ids == {f"M{i}" for i in range(10)}
    # Toutes les lignes d'une machine tombent du même côté (split par machine).
    assert train.height + val.height == df.height


# ---------------------------------------------------------------------------
# to_xy
# ---------------------------------------------------------------------------
def test_to_xy_shapes_and_target():
    df = _labelled_df(["M1"], cycles_per_machine=4)
    x, y = prep.to_xy(df, target="RUL")
    assert x.shape == (4, len(prep.FEATURES))
    assert y.tolist() == [3, 2, 1, 0]


# ---------------------------------------------------------------------------
# fit_scaler
# ---------------------------------------------------------------------------
def test_fit_scaler_no_persist_when_save_as_empty():
    df = _labelled_df(["M1"], cycles_per_machine=20)
    x, _ = prep.to_xy(df, target="RUL")
    scaler = prep.fit_scaler(x, save_as="")  # save_as vide -> pas d'écriture disque
    transformed = scaler.transform(x)
    # StandardScaler : moyenne ~0 après transformation.
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# make_windows
# ---------------------------------------------------------------------------
def test_make_windows_shapes():
    df = _labelled_df(["M1", "M2"], cycles_per_machine=10)
    x, y = prep.make_windows(df, target="RUL", seq_len=5)
    # 2 machines * (10 - 5 + 1) = 12 fenêtres.
    assert x.shape == (12, 5, len(prep.FEATURES))
    assert y.shape == (12,)
    assert x.dtype == np.float32


def test_make_windows_left_pads_short_machines():
    df = _labelled_df(["M1"], cycles_per_machine=3)  # plus court que seq_len
    x, y = prep.make_windows(df, target="RUL", seq_len=5)
    assert x.shape == (1, 5, len(prep.FEATURES))  # left-pad -> une seule fenêtre
    assert y.shape == (1,)


def test_make_windows_target_aligned_on_last_cycle():
    df = _labelled_df(["M1"], cycles_per_machine=6)  # RUL: 5,4,3,2,1,0
    _, y = prep.make_windows(df, target="RUL", seq_len=3)
    # Fenêtres finissant aux cycles 3..6 -> RUL des derniers cycles : 3,2,1,0.
    assert y.tolist() == [3.0, 2.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# make_test_windows
# ---------------------------------------------------------------------------
def test_make_test_windows_one_window_per_machine():
    df = _labelled_df(["M1", "M2"], cycles_per_machine=8)
    x, ids = prep.make_test_windows(df, seq_len=4)
    assert x.shape == (2, 4, len(prep.FEATURES))
    assert set(ids) == {"M1", "M2"}


def test_make_test_windows_left_pads_short_machine():
    df = _labelled_df(["M1"], cycles_per_machine=2)  # plus court que seq_len
    x, ids = prep.make_test_windows(df, seq_len=5)
    assert x.shape == (1, 5, len(prep.FEATURES))
    assert ids == ["M1"]
