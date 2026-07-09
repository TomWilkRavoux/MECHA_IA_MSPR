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
    # Évaluation - XGBoost (classification `at_risk`)

    Modèle rechargé : `models/xgb_classifier.joblib` (entraîné dans
    `02_gradient_boosting_baseline.py`). Gradient Boosting d'arbres avec
    `scale_pos_weight` pour compenser le déséquilibre de classes.
    """)
    return


@app.cell
def _(joblib, prep):
    scaler = prep.load_scaler()
    model = joblib.load(prep.MODELS_DIR / "xgb_classifier.joblib")
    ev = prep.test_last_cycle_eval()
    x = scaler.transform(ev.select(prep.FEATURES).to_numpy())
    y = ev["at_risk_true"].to_numpy()
    proba = model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return model, pred, proba, y


@app.cell
def _(metrics, mo, pred, proba, y):
    m = metrics.classification_metrics(y, pred, proba)
    mo.md(
        "## Métriques (test)\n\n"
        + "\n".join(f"- **{k}** : {v:.3f}" for k, v in m.items())
    )
    return (m,)


@app.cell
def _(metrics, pred, y):
    metrics.plot_confusion(y, pred, title="XGBoost - Matrice de confusion (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_roc(y, proba, title="XGBoost - Courbe ROC (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_pr(y, proba, title="XGBoost - Précision/Rappel (at_risk)")
    return


@app.cell
def _(metrics, model, prep):
    metrics.plot_feature_importance(
        prep.FEATURES,
        model.feature_importances_,
        title="XGBoost - Importance des variables",
    )
    return


@app.cell
def _(m, mo):
    mo.md(f"""
    ## Conclusion
    - **Recall = {m["recall"]:.2f}**, **Precision = {m["precision"]:.2f}**,
      **F1 = {m["f1"]:.2f}**, **ROC-AUC = {m.get("roc_auc", float("nan")):.2f}**.
    - XGBoost pousse en général le **recall** (moins de pannes manquées) grâce au
      boosting séquentiel. À comparer à la Random Forest et au LSTM.
    """)
    return


if __name__ == "__main__":
    app.run()
