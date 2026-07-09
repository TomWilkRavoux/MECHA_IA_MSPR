import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from ml import metrics, prep

    torch.manual_seed(42)
    np.random.seed(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE, metrics, mo, nn, np, prep, torch


@app.cell
def _(DEVICE, mo, torch):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun"
    mo.md(
        f"""
        # 03 - Deep Learning : LSTM sur fenêtres glissantes (PyTorch / GPU)

        **Pourquoi un LSTM ?** Contrairement aux modèles tabulaires (RF, XGBoost) qui
        traitent chaque cycle isolément, le LSTM exploite la **dynamique temporelle** de
        la dégradation. Approche de référence sur C-MAPSS : chaque trajectoire est
        découpée en **fenêtres glissantes** de `SEQ_LEN` cycles et le réseau apprend les
        tendances menant à la défaillance.

        **Backend** : PyTorch (build CUDA 12.8) - device détecté : **`{DEVICE}`** ({gpu}).

        Deux modèles séparés : **classification** (`at_risk`, sortie logit + sigmoïde) et
        **régression** (`RUL`, sortie linéaire).
        """
    )
    return


@app.cell
def _(mo, prep):
    # --- Fenêtres glissantes par machine (features normalisées) ---
    scaler = prep.load_scaler()
    df_clf = prep.load_train_classification()
    df_rul = prep.load_train_rul()

    tr_clf, va_clf = prep.split_by_machine(df_clf, val_frac=0.2, seed=42)
    tr_rul, va_rul = prep.split_by_machine(df_rul, val_frac=0.2, seed=42)

    xw_tr_clf, yw_tr_clf = prep.make_windows(tr_clf, target="at_risk", scaler=scaler)
    xw_va_clf, yw_va_clf = prep.make_windows(va_clf, target="at_risk", scaler=scaler)

    xw_tr_rul, yw_tr_rul = prep.make_windows(tr_rul, target="RUL", scaler=scaler)
    xw_va_rul, yw_va_rul = prep.make_windows(va_rul, target="RUL", scaler=scaler)
    yw_tr_rul, yw_va_rul = prep.clip_rul(yw_tr_rul), prep.clip_rul(yw_va_rul)

    mo.md(
        f"Fenêtres (seq_len={prep.SEQ_LEN}) - classif train {xw_tr_clf.shape} · "
        f"RUL train {xw_tr_rul.shape}."
    )
    return (
        scaler,
        xw_tr_clf,
        xw_tr_rul,
        xw_va_clf,
        xw_va_rul,
        yw_tr_clf,
        yw_tr_rul,
        yw_va_clf,
        yw_va_rul,
    )


@app.cell
def _(prep):
    # --- Architecture (source unique : ml/lstm.py) ---
    from ml.lstm import LSTMNet

    n_features = len(prep.FEATURES)
    return LSTMNet, n_features


@app.cell
def _(DEVICE, np, torch):
    # --- Boucle d'entraînement générique avec early stopping ---
    def _to_gpu(x):
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

    def run_epoch(model, x, y, loss_fn, opt=None, batch=512):
        train = opt is not None
        model.train(train)
        n = len(x)
        idx = (
            torch.randperm(n, device=x.device)
            if train
            else torch.arange(n, device=x.device)
        )
        total = 0.0
        with torch.set_grad_enabled(train):
            for i in range(0, n, batch):
                b = idx[i : i + batch]
                out = model(x[b])
                loss = loss_fn(out, y[b])
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total += loss.item() * len(b)
        return total / n

    def train_model(
        model, xtr, ytr, xva, yva, loss_fn, epochs=30, patience=5, batch=512
    ):
        model.to(DEVICE)
        xtr, ytr = _to_gpu(xtr), _to_gpu(ytr)
        xva, yva = _to_gpu(xva), _to_gpu(yva)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist = {"loss": [], "val_loss": []}
        best, best_state, wait = float("inf"), None, 0
        for _ in range(epochs):
            tr_loss = run_epoch(model, xtr, ytr, loss_fn, opt, batch)
            va_loss = run_epoch(model, xva, yva, loss_fn, None, batch)
            hist["loss"].append(tr_loss)
            hist["val_loss"].append(va_loss)
            if va_loss < best:
                best, best_state, wait = (
                    va_loss,
                    {k: v.detach().clone() for k, v in model.state_dict().items()},
                    0,
                )
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)  # restore best weights
        return hist

    return train_model


@app.cell
def _(
    LSTMNet,
    n_features,
    nn,
    prep,
    torch,
    train_model,
    xw_tr_clf,
    xw_va_clf,
    yw_tr_clf,
    yw_va_clf,
):
    # --- Entraînement LSTM classification (BCEWithLogits + pos_weight) ---
    pos_w = float((yw_tr_clf == 0).sum() / (yw_tr_clf == 1).sum())
    loss_clf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    lstm_clf = LSTMNet(n_features)
    hist_clf = train_model(
        lstm_clf, xw_tr_clf, yw_tr_clf, xw_va_clf, yw_va_clf, loss_clf
    )
    torch.save(lstm_clf.state_dict(), prep.MODELS_DIR / "lstm_classifier.pt")
    return hist_clf, lstm_clf


@app.cell
def _(
    LSTMNet,
    n_features,
    nn,
    prep,
    torch,
    train_model,
    xw_tr_rul,
    xw_va_rul,
    yw_tr_rul,
    yw_va_rul,
):
    # --- Entraînement LSTM régression RUL (MSE) ---
    lstm_reg = LSTMNet(n_features)
    hist_reg = train_model(
        lstm_reg, xw_tr_rul, yw_tr_rul, xw_va_rul, yw_va_rul, nn.MSELoss()
    )
    torch.save(lstm_reg.state_dict(), prep.MODELS_DIR / "lstm_regressor.pt")
    return hist_reg, lstm_reg


@app.cell
def _(hist_clf, metrics):
    fig_hc = metrics.plot_history(hist_clf, title="LSTM classif - apprentissage (BCE)")
    metrics.save_fig(fig_hc, "lstm_history_clf.png")
    fig_hc
    return


@app.cell
def _(hist_reg, metrics):
    fig_hr = metrics.plot_history(hist_reg, title="LSTM RUL - apprentissage (MSE)")
    metrics.save_fig(fig_hr, "lstm_history_rul.png")
    fig_hr
    return


@app.cell
def _(mo):
    mo.md(r"## Évaluation sur le jeu de test officiel")
    return


@app.cell
def _(DEVICE, lstm_clf, lstm_reg, metrics, np, prep, scaler, torch):
    # Une fenêtre (derniers cycles) par machine test + vérité terrain alignée
    ev = prep.test_last_cycle_eval()
    rul_by_id = dict(zip(ev["machine_id"].to_list(), ev["RUL_true"].to_list()))

    xw_test, ids = prep.make_test_windows(prep.load_test(), scaler=scaler)
    y_rul_true = np.array([rul_by_id[i] for i in ids], dtype=float)
    y_atrisk_true = (y_rul_true <= prep.RISK_THRESHOLD).astype(int)

    xt = torch.as_tensor(xw_test, dtype=torch.float32, device=DEVICE)
    lstm_clf.eval()
    lstm_reg.eval()
    with torch.no_grad():
        proba = torch.sigmoid(lstm_clf(xt)).cpu().numpy()
        pred_rul = lstm_reg(xt).cpu().numpy()
    pred_clf = (proba >= 0.5).astype(int)

    m_lstm_clf = metrics.classification_metrics(y_atrisk_true, pred_clf, proba)
    m_lstm_rul = metrics.regression_metrics(y_rul_true, pred_rul)
    return m_lstm_clf, m_lstm_rul, pred_rul, y_rul_true


@app.cell
def _(m_lstm_clf, m_lstm_rul, metrics, mo):
    table = metrics.format_metrics_table(
        {"LSTM - classif (at_risk)": m_lstm_clf, "LSTM - RUL (test)": m_lstm_rul}
    )
    mo.md("### Résultats LSTM (test)\n\n" + table)
    return


@app.cell
def _(metrics, pred_rul, y_rul_true):
    fig_sc = metrics.plot_rul_scatter(
        y_rul_true, pred_rul, title="LSTM - RUL prédit vs réel (test)"
    )
    metrics.save_fig(fig_sc, "lstm_rul_scatter.png")
    fig_sc
    return


@app.cell
def _(m_lstm_clf, m_lstm_rul, mo):
    mo.md(
        f"""
        ### Conclusion - LSTM
        - **Classification** : F1 = {m_lstm_clf["f1"]:.2f}, recall =
          {m_lstm_clf["recall"]:.2f}, AUC = {m_lstm_clf.get("roc_auc", float("nan")):.2f}.
        - **RUL** : RMSE = {m_lstm_rul["RMSE"]:.1f}, MAE = {m_lstm_rul["MAE"]:.1f},
          R² = {m_lstm_rul["R2"]:.2f}, score NASA = {m_lstm_rul["NASA"]:.0f}.
        - Les **courbes d'apprentissage** (loss/val_loss) permettent de vérifier
          l'absence de sur-apprentissage : l'early stopping restaure les meilleurs poids.
        - Le LSTM exploite la temporalité que les modèles tabulaires ignorent : gain
          attendu surtout sur le **RUL** (dynamique de dégradation). À comparer aux
          notebooks 01 (RF) et 02 (XGBoost) dans le tableau de synthèse.
        """
    )
    return


if __name__ == "__main__":
    app.run()
