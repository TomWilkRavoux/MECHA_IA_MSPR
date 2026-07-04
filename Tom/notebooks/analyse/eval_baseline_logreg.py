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

    import joblib
    import marimo as mo
    import numpy as np

    from ml import metrics, prep

    return joblib, metrics, mo, np, prep


@app.cell
def _(mo):
    mo.md(r"""
    # Évaluation — Baseline régression logistique (classification `at_risk`)

    Modèle rechargé : `models/baseline_logreg.joblib` (baseline exigée par le CDC,
    entraînée dans `02_gradient_boosting_baseline.py`). Modèle linéaire simple :
    **plancher de performance** et point de comparaison très interprétable.
    """)
    return


@app.cell
def _(joblib, prep):
    scaler = prep.load_scaler()
    model = joblib.load(prep.MODELS_DIR / "baseline_logreg.joblib")
    ev = prep.test_last_cycle_eval()
    x = scaler.transform(ev.select(prep.FEATURES).to_numpy())
    y = ev["at_risk_true"].to_numpy()
    proba = model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return model, pred, proba, y


@app.cell
def _(metrics, mo, pred, proba, y):
    m = metrics.classification_metrics(y, pred, proba)
    mo.md("## Métriques (test)\n\n" + "\n".join(f"- **{k}** : {v:.3f}" for k, v in m.items()))
    return (m,)


@app.cell
def _(metrics, pred, y):
    metrics.plot_confusion(y, pred, title="Baseline LogReg — Confusion (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_roc(y, proba, title="Baseline LogReg — Courbe ROC (at_risk)")
    return


@app.cell
def _(metrics, model, np, prep):
    # Coefficients linéaires = poids de chaque capteur (interprétabilité directe)
    metrics.plot_feature_importance(
        prep.FEATURES, np.abs(model.coef_[0]), title="Baseline LogReg — |coefficients|"
    )
    return


@app.cell
def _(m, mo):
    mo.md(f"""
    ## Conclusion
    - **F1 = {m['f1']:.2f}**, **Recall = {m['recall']:.2f}**,
      **ROC-AUC = {m.get('roc_auc', float('nan')):.2f}**.
    - Modèle linéaire : incapable de capturer les interactions non linéaires entre
      capteurs. L'écart avec RF / XGBoost / LSTM **mesure l'apport** de ces modèles.
    - Avantage : les **coefficients** sont directement lisibles (sens et poids de
      chaque capteur), utile pour expliquer la décision au métier.
    """)
    return


if __name__ == "__main__":
    app.run()
