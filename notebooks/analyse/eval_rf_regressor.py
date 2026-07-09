import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # racine du projet
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import joblib
    import marimo as mo

    from ml import metrics, prep

    return joblib, metrics, mo, prep


@app.cell
def _(mo):
    mo.md(r"""
    # Évaluation - Random Forest (régression `RUL`)

    Modèle rechargé : `models/rf_regressor.joblib` (entraîné dans `01_random_forest.py`).
    Prédiction du RUL sur le dernier cycle de chaque machine test, comparée à la
    vérité terrain `RUL_true`.
    """)
    return


@app.cell
def _(joblib, prep):
    scaler = prep.load_scaler()
    model = joblib.load(prep.MODELS_DIR / "rf_regressor.joblib")
    ev = prep.test_last_cycle_eval()
    x = scaler.transform(ev.select(prep.FEATURES).to_numpy())
    y = ev["RUL_true"].to_numpy()
    pred = model.predict(x)
    return model, pred, y


@app.cell
def _(metrics, mo, pred, y):
    m = metrics.regression_metrics(y, pred)
    mo.md(
        "## Métriques (test)\n\n"
        + "\n".join(f"- **{k}** : {v:.2f}" for k, v in m.items())
    )
    return (m,)


@app.cell
def _(metrics, pred, y):
    metrics.plot_rul_scatter(y, pred, title="RF - RUL prédit vs réel (test)")
    return


@app.cell
def _(metrics, pred, y):
    metrics.plot_error_hist(y, pred, title="RF - Distribution de l'erreur de RUL")
    return


@app.cell
def _(metrics, model, prep):
    metrics.plot_feature_importance(
        prep.FEATURES,
        model.feature_importances_,
        title="RF - Importance des variables (RUL)",
    )
    return


@app.cell
def _(m, mo):
    mo.md(f"""
    ## Conclusion
    - **MAE = {m["MAE"]:.1f} cycles** : erreur moyenne lisible par la maintenance.
    - **RMSE = {m["RMSE"]:.1f}** : pénalise les grosses erreurs.
    - **R² = {m["R2"]:.2f}** : part de variance expliquée.
    - **Score NASA = {m["NASA"]:.0f}** (plus bas = mieux) : pénalise les prédictions
      **en retard** (RUL surestimé = panne détectée trop tard).
    """)
    return


if __name__ == "__main__":
    app.run()
