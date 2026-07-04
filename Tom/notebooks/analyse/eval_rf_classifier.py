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

    from ml import metrics, prep

    return joblib, metrics, mo, prep


@app.cell
def _(mo):
    mo.md(
        r"""
        # Évaluation — Random Forest (classification `at_risk`)

        Modèle rechargé : `models/rf_classifier.joblib` (entraîné dans `01_random_forest.py`).
        Évaluation sur le jeu de test officiel : dernier cycle observé de chaque machine,
        label `at_risk` dérivé de la vérité terrain RUL (seuil 30).
        """
    )
    return


@app.cell
def _(joblib, prep):
    scaler = prep.load_scaler()
    model = joblib.load(prep.MODELS_DIR / "rf_classifier.joblib")
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
    metrics.plot_confusion(y, pred, title="RF — Matrice de confusion (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_roc(y, proba, title="RF — Courbe ROC (at_risk)")
    return


@app.cell
def _(metrics, proba, y):
    metrics.plot_pr(y, proba, title="RF — Précision/Rappel (at_risk)")
    return


@app.cell
def _(metrics, model, prep):
    metrics.plot_feature_importance(
        prep.FEATURES, model.feature_importances_, title="RF — Importance des variables"
    )
    return


@app.cell
def _(m, mo):
    mo.md(
        f"""
        ## Conclusion
        - **Recall = {m['recall']:.2f}** : détection des machines réellement à risque —
          métrique critique (un faux négatif = panne manquée).
        - **Precision = {m['precision']:.2f}** : fiabilité des alertes émises.
        - **F1 = {m['f1']:.2f}**, **ROC-AUC = {m.get('roc_auc', float('nan')):.2f}** (séparabilité globale).
        - L'accuracy est peu informative ici (~14 % de positifs seulement).
        """
    )
    return


if __name__ == "__main__":
    app.run()
