"""Pipeline d'entraînement reproductible du LSTM MECHA (remplace les notebooks).

Convertit `notebooks/03_lstm.py` en une commande unique et **traçable** : prépare
les données, entraîne les deux têtes LSTM (classification `at_risk` + régression
`RUL`), évalue sur le jeu de test officiel, sauvegarde les artefacts dans
`models/` et **journalise les métriques** dans `models/metrics.json`.

Ce fichier est la brique « ré-entraînement » de la chaîne MLOps : couplé à
`ml/validate_metrics.py` (gate CI), il permet de ré-entraîner puis de garantir
qu'aucune régression de performance n'est committée.

Usage (depuis la racine) :
    uv run python -m ml.train                 # ré-entraîne + évalue + journalise
    uv run python -m ml.train --eval-only     # recharge les artefacts existants, évalue seulement
    uv run python -m ml.train --epochs 10 --seed 7
    uv run python -m ml.train --quick         # smoke : 2 epochs (vérifie que le pipeline tourne)
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime

import joblib
import numpy as np
import torch
from torch import nn

from ml import metrics, prep, registry
from ml.lstm import LSTMNet, load_lstm


# ----------------------------------------------------------------------------
# Boucle d'entraînement (factorisée depuis notebooks/03_lstm.py)
# ----------------------------------------------------------------------------
def _run_epoch(model, x, y, loss_fn, opt=None, batch=512) -> float:
    train = opt is not None
    model.train(train)
    n = len(x)
    idx = torch.randperm(n, device=x.device) if train else torch.arange(n, device=x.device)
    total = 0.0
    with torch.set_grad_enabled(train):
        for i in range(0, n, batch):
            b = idx[i : i + batch]
            loss = loss_fn(model(x[b]), y[b])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(b)
    return total / n


def train_head(xtr, ytr, xva, yva, loss_fn, device, *, epochs=30, patience=5, batch=512) -> LSTMNet:
    """Entraîne une tête LSTM avec early stopping (restaure les meilleurs poids)."""
    model = LSTMNet().to(device)
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    xtr, ytr, xva, yva = to_t(xtr), to_t(ytr), to_t(xva), to_t(yva)
    # weight_decay explicite (régularisation L2) : 0.0 conserve la baseline entraînée
    # committée telle quelle, tout en fixant l'hyperparamètre au lieu de l'implicite.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        tr = _run_epoch(model, xtr, ytr, loss_fn, opt, batch)
        va = _run_epoch(model, xva, yva, loss_fn, None, batch)
        print(f"    epoch {ep + 1:2d}/{epochs}  loss={tr:.4f}  val_loss={va:.4f}")
        if va < best:
            best, best_state, wait = (
                va,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"    early stopping (patience={patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit(device: str, *, epochs: int, seed: int, out_dir) -> tuple[LSTMNet, LSTMNet, object]:
    """Entraîne les deux têtes et écrit tous les artefacts dans `out_dir`.

    N'écrase **jamais** la baseline plate `models/*` : les artefacts vont dans le
    dossier de run versionné. Retourne aussi le scaler ajusté (utilisé tel quel
    pour l'évaluation, sans rechargement).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_clf = prep.load_train_classification()
    df_rul = prep.load_train_rul()
    tr_clf, va_clf = prep.split_by_machine(df_clf, val_frac=0.2, seed=seed)
    tr_rul, va_rul = prep.split_by_machine(df_rul, val_frac=0.2, seed=seed)

    # Scaler ajusté UNIQUEMENT sur le train de classification (anti-fuite).
    # save_as=None : on le sérialise nous-mêmes dans le dossier de run.
    x_fit, _ = prep.to_xy(tr_clf, target="at_risk")
    scaler = prep.fit_scaler(x_fit, save_as=None)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    xw_tr_clf, yw_tr_clf = prep.make_windows(tr_clf, "at_risk", scaler=scaler)
    xw_va_clf, yw_va_clf = prep.make_windows(va_clf, "at_risk", scaler=scaler)
    xw_tr_rul, yw_tr_rul = prep.make_windows(tr_rul, "RUL", scaler=scaler)
    xw_va_rul, yw_va_rul = prep.make_windows(va_rul, "RUL", scaler=scaler)
    yw_tr_rul, yw_va_rul = prep.clip_rul(yw_tr_rul), prep.clip_rul(yw_va_rul)

    print(f"  fenêtres classif {xw_tr_clf.shape} · RUL {xw_tr_rul.shape}")

    # Classification : BCEWithLogits pondéré (déséquilibre at_risk).
    pos_w = float((yw_tr_clf == 0).sum() / max((yw_tr_clf == 1).sum(), 1))
    print(f"  [classif] pos_weight={pos_w:.2f}")
    clf = train_head(
        xw_tr_clf,
        yw_tr_clf,
        xw_va_clf,
        yw_va_clf,
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device)),
        device,
        epochs=epochs,
    )
    torch.save(clf.state_dict(), out_dir / "lstm_classifier.pt")

    print("  [régression] MSE")
    reg = train_head(
        xw_tr_rul,
        yw_tr_rul,
        xw_va_rul,
        yw_va_rul,
        nn.MSELoss(),
        device,
        epochs=epochs,
    )
    torch.save(reg.state_dict(), out_dir / "lstm_regressor.pt")
    return clf, reg, scaler


# ----------------------------------------------------------------------------
# Évaluation (jeu de test officiel C-MAPSS)
# ----------------------------------------------------------------------------
def evaluate(clf: LSTMNet, reg: LSTMNet, scaler, device: str) -> dict:
    """Évalue les deux têtes sur la dernière fenêtre de chaque machine test."""
    ev = prep.test_last_cycle_eval()
    rul_by_id = dict(zip(ev["machine_id"].to_list(), ev["RUL_true"].to_list(), strict=True))
    xw, ids = prep.make_test_windows(prep.load_test(), scaler=scaler)
    y_rul = np.array([rul_by_id[i] for i in ids], dtype=float)
    y_atrisk = (y_rul <= prep.RISK_THRESHOLD).astype(int)

    xt = torch.as_tensor(xw, dtype=torch.float32, device=device)
    clf.eval()
    reg.eval()
    with torch.no_grad():
        proba = torch.sigmoid(clf(xt)).cpu().numpy()
        pred_rul = reg(xt).cpu().numpy()
    pred_clf = (proba >= 0.5).astype(int)

    return {
        "classification": metrics.classification_metrics(y_atrisk, pred_clf, proba),
        "regression": metrics.regression_metrics(y_rul, pred_rul),
        "n_test_machines": int(len(ids)),
    }


def write_metrics(result: dict, out_dir, *, mode: str, device: str, seed: int, epochs: int) -> None:
    """Journalise les métriques + métadonnées de run dans `out_dir/metrics.json`."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "device": device,
        "python": platform.python_version(),
        "seed": seed,
        "epochs": epochs,
        "seq_len": prep.SEQ_LEN,
        "risk_threshold": prep.RISK_THRESHOLD,
        **result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  métriques → {path.relative_to(prep.ROOT)}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Entraînement/évaluation du LSTM MECHA.")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Recharge les artefacts existants et évalue seulement (pas de ré-entraînement).",
    )
    p.add_argument("--epochs", type=int, default=30, help="Nombre d'epochs (défaut 30).")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire (défaut 42).")
    p.add_argument("--quick", action="store_true", help="Smoke : 2 epochs (vérifie le pipeline).")
    p.add_argument(
        "--no-promote",
        action="store_true",
        help="Enregistre le run sans le promouvoir courant (la baseline reste servie).",
    )
    args = p.parse_args(argv)

    epochs = 2 if args.quick else args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    if args.eval_only:
        print("Mode : évaluation seule (artefacts existants)")
        scaler = prep.load_scaler()
        clf, _ = load_lstm("lstm_classifier.pt", device)
        reg, _ = load_lstm("lstm_regressor.pt", device)
        mode = "eval-only"
        # Journalise à côté des artefacts servis (run courant, sinon baseline plate).
        out_dir = registry.resolve("metrics.json").parent
        run_id = None
    else:
        run_id = registry.new_run_id()
        out_dir = registry.run_dir(run_id)
        print(f"Mode : entraînement ({epochs} epochs, seed {args.seed}) — run {run_id}")
        clf, reg, scaler = fit(device, epochs=epochs, seed=args.seed, out_dir=out_dir)
        mode = "quick" if args.quick else "train"

    result = evaluate(clf, reg, scaler, device)
    write_metrics(result, out_dir, mode=mode, device=device, seed=args.seed, epochs=epochs)
    if run_id is not None:
        registry.register_run(run_id, metrics=result, promote=not args.no_promote)
        served = registry.current_run_id() or "baseline plate"
        print(f"  run enregistré → registry.json (courant : {served})")

    c, r = result["classification"], result["regression"]
    print(
        f"\n  Classification  F1={c['f1']:.3f}  recall={c['recall']:.3f}  AUC={c.get('roc_auc', float('nan')):.3f}"
    )
    print(
        f"  Régression      RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}  R²={r['R2']:.3f}  NASA={r['NASA']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
