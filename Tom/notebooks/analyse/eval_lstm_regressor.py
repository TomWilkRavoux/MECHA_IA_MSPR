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
    mo.md(
        r"""
        # Évaluation - LSTM (régression `RUL`)

        Modèle rechargé : `models/lstm_regressor.pt` (PyTorch, entraîné sur GPU dans
        `03_lstm.py`). Prédiction du RUL sur la fenêtre des derniers `SEQ_LEN` cycles de
        chaque machine test, comparée à la vérité terrain `RUL_true`.
        """
    )
    return


@app.cell
def _(load_lstm, np, prep, torch):
    scaler = prep.load_scaler()
    model, device = load_lstm("lstm_regressor.pt")

    ev = prep.test_last_cycle_eval()
    rul_by_id = dict(zip(ev["machine_id"].to_list(), ev["RUL_true"].to_list()))
    xw, ids = prep.make_test_windows(prep.load_test(), scaler=scaler)
    y = np.array([rul_by_id[i] for i in ids], dtype=float)

    xt = torch.as_tensor(xw, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(xt).cpu().numpy()
    return device, pred, y


@app.cell
def _(device, metrics, mo, pred, y):
    m = metrics.regression_metrics(y, pred)
    mo.md(
        f"## Métriques (test · device `{device}`)\n\n"
        + "\n".join(f"- **{k}** : {v:.2f}" for k, v in m.items())
    )
    return (m,)


@app.cell
def _(metrics, pred, y):
    metrics.plot_rul_scatter(y, pred, title="LSTM - RUL prédit vs réel (test)")
    return


@app.cell
def _(metrics, pred, y):
    metrics.plot_error_hist(y, pred, title="LSTM - Distribution de l'erreur de RUL")
    return


@app.cell
def _(m, mo):
    mo.md(
        f"""
        ## Conclusion
        - **MAE = {m["MAE"]:.1f}**, **RMSE = {m["RMSE"]:.1f}**, **R² = {m["R2"]:.2f}**,
          **Score NASA = {m["NASA"]:.0f}**.
        - En captant la trajectoire de dégradation, le LSTM obtient généralement les
          **meilleurs RMSE/MAE/R²** du panel. Point de vigilance métier : surveiller le
          **score NASA** (retards de prédiction) pour la sûreté industrielle.
        """
    )
    return


if __name__ == "__main__":
    app.run()
