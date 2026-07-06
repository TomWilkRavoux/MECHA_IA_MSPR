import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # Tom/
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import marimo as mo
    import numpy as np
    import torch

    from ml import metrics, prep
    from ml.lstm import load_lstm

    return load_lstm, metrics, mo, np, prep, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Évaluation - LSTM (classification `at_risk`)

    Modèle rechargé : `models/lstm_classifier.pt` (PyTorch, entraîné sur GPU dans
    `03_lstm.py`). Évaluation sur une **fenêtre glissante** des derniers `SEQ_LEN`
    cycles de chaque machine test (via `make_test_windows`), label `at_risk` dérivé
    de la vérité terrain RUL.
    """)
    return


@app.cell
def _(load_lstm, np, prep, torch):
    scaler = prep.load_scaler()
    model, device = load_lstm("lstm_classifier.pt")

    ev = prep.test_last_cycle_eval()
    rul_by_id = dict(zip(ev["machine_id"].to_list(), ev["RUL_true"].to_list()))
    xw, ids = prep.make_test_windows(prep.load_test(), scaler=scaler)
    y = (
        np.array([rul_by_id[i] for i in ids], dtype=float) <= prep.RISK_THRESHOLD
    ).astype(int)

    xt = torch.as_tensor(xw, dtype=torch.float32, device=device)
    with torch.no_grad():
        proba = torch.sigmoid(model(xt)).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    return device, pred, proba, y


@app.cell
def _(device, metrics, mo, pred, proba, y):
    m = metrics.classification_metrics(y, pred, proba)
    mo.md(
        f"## Métriques (test · device `{device}`)\n\n"
        + "\n".join(f"- **{k}** : {v:.3f}" for k, v in m.items())
    )
    return (m,)


@app.cell
def _(metrics, pred, y):
    metrics.plot_confusion(y, pred, title="LSTM - Matrice de confusion (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_roc(y, proba, title="LSTM - Courbe ROC (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_pr(y, proba, title="LSTM - Précision/Rappel (at_risk)")
    return


@app.cell
def _(m, mo):
    mo.md(f"""
    ## Conclusion
    - **Recall = {m["recall"]:.2f}**, **Precision = {m["precision"]:.2f}**,
      **F1 = {m["f1"]:.2f}**, **ROC-AUC = {m.get("roc_auc", float("nan")):.2f}**.
    - Le LSTM exploite la **dynamique temporelle** (dérive des capteurs sur la
      fenêtre) que les modèles tabulaires ignorent : gain typique sur le recall
      (détection plus fine des machines en fin de vie).
    """)
    return


if __name__ == "__main__":
    app.run()
