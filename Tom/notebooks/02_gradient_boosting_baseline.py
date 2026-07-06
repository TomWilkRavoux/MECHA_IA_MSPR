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

    import joblib
    import marimo as mo
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from xgboost import XGBClassifier, XGBRegressor

    from ml import metrics, prep

    return (
        LinearRegression,
        LogisticRegression,
        XGBClassifier,
        XGBRegressor,
        joblib,
        metrics,
        mo,
        prep,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 02 - Baseline (régression simple) + Gradient Boosting (XGBoost)

    Deux modèles, comparés à la Random Forest du notebook 01 :

    1. **Baseline exigée par le CDC** - `LogisticRegression` (classification) et
       `LinearRegression` (RUL). Modèles linéaires simples : point de comparaison
       minimal, très interprétables mais incapables de capturer les non-linéarités.
    2. **XGBoost** - Gradient Boosting d'arbres, état de l'art sur données
       tabulaires. Construit les arbres séquentiellement pour corriger les erreurs
       résiduelles ; attendu au-dessus de la Random Forest.

    Les deux tâches (`at_risk` + `RUL`) sont traitées.
    """)
    return


@app.cell
def _(mo, prep):
    # --- Chargement + split par machine + normalisation ---
    df_clf = prep.load_train_classification()
    df_rul = prep.load_train_rul()
    tr_clf, va_clf = prep.split_by_machine(df_clf, val_frac=0.2, seed=42)
    tr_rul, va_rul = prep.split_by_machine(df_rul, val_frac=0.2, seed=42)

    x_tr_clf, y_tr_clf = prep.to_xy(tr_clf, "at_risk")
    x_va_clf, y_va_clf = prep.to_xy(va_clf, "at_risk")
    x_tr_rul, y_tr_rul = prep.to_xy(tr_rul, "RUL")
    x_va_rul, y_va_rul = prep.to_xy(va_rul, "RUL")
    y_tr_rul, y_va_rul = prep.clip_rul(y_tr_rul), prep.clip_rul(y_va_rul)

    scaler = prep.load_scaler()  # scaler ajusté au notebook 01 (mêmes features)
    xs_tr_clf, xs_va_clf = scaler.transform(x_tr_clf), scaler.transform(x_va_clf)
    xs_tr_rul, xs_va_rul = scaler.transform(x_tr_rul), scaler.transform(x_va_rul)

    # Déséquilibre -> scale_pos_weight pour XGBoost
    pos_weight = float((y_tr_clf == 0).sum() / (y_tr_clf == 1).sum())
    mo.md(
        f"`scale_pos_weight` (XGBoost) = **{pos_weight:.2f}** (ratio négatifs/positifs)."
    )
    return (
        pos_weight,
        xs_tr_clf,
        xs_tr_rul,
        xs_va_clf,
        xs_va_rul,
        y_tr_clf,
        y_tr_rul,
        y_va_clf,
        y_va_rul,
    )


@app.cell
def _(
    LinearRegression,
    LogisticRegression,
    joblib,
    prep,
    xs_tr_clf,
    xs_tr_rul,
    y_tr_clf,
    y_tr_rul,
):
    # --- Baseline linéaire (CDC) ---
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        xs_tr_clf, y_tr_clf
    )
    lr_reg = LinearRegression().fit(xs_tr_rul, y_tr_rul)
    joblib.dump(lr_clf, prep.MODELS_DIR / "baseline_logreg.joblib")
    joblib.dump(lr_reg, prep.MODELS_DIR / "baseline_linreg.joblib")
    return lr_clf, lr_reg


@app.cell
def _(
    XGBClassifier,
    XGBRegressor,
    joblib,
    pos_weight,
    prep,
    xs_tr_clf,
    xs_tr_rul,
    xs_va_clf,
    xs_va_rul,
    y_tr_clf,
    y_tr_rul,
    y_va_clf,
    y_va_rul,
):
    # --- XGBoost (early stopping sur la validation) ---
    xgb_clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
    )
    xgb_clf.fit(xs_tr_clf, y_tr_clf, eval_set=[(xs_va_clf, y_va_clf)], verbose=False)

    xgb_reg = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
    )
    xgb_reg.fit(xs_tr_rul, y_tr_rul, eval_set=[(xs_va_rul, y_va_rul)], verbose=False)

    joblib.dump(xgb_clf, prep.MODELS_DIR / "xgb_classifier.joblib")
    joblib.dump(xgb_reg, prep.MODELS_DIR / "xgb_regressor.joblib")
    return xgb_clf, xgb_reg


@app.cell
def _(mo, xgb_clf, xgb_reg):
    mo.md(f"""
    **XGBoost - arbres retenus (early stopping)** : classif "
        f"{xgb_clf.best_iteration + 1} · RUL {xgb_reg.best_iteration + 1}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Évaluation sur le jeu de test officiel
    """)
    return


@app.cell
def _(lr_clf, lr_reg, metrics, prep, xgb_clf, xgb_reg):
    # Jeu de test : dernier cycle par machine, features normalisées
    ev = prep.test_last_cycle_eval()
    xt = prep.load_scaler().transform(ev.select(prep.FEATURES).to_numpy())
    y_atrisk = ev["at_risk_true"].to_numpy()
    y_rul = ev["RUL_true"].to_numpy()

    m_base_clf = metrics.classification_metrics(
        y_atrisk, lr_clf.predict(xt), lr_clf.predict_proba(xt)[:, 1]
    )
    m_xgb_clf = metrics.classification_metrics(
        y_atrisk, xgb_clf.predict(xt), xgb_clf.predict_proba(xt)[:, 1]
    )
    m_base_rul = metrics.regression_metrics(y_rul, lr_reg.predict(xt))
    m_xgb_rul = metrics.regression_metrics(y_rul, xgb_reg.predict(xt))
    return m_base_clf, m_base_rul, m_xgb_clf, m_xgb_rul, xt, y_rul


@app.cell
def _(m_base_clf, m_base_rul, m_xgb_clf, m_xgb_rul, metrics, mo):
    table = metrics.format_metrics_table(
        {
            "Baseline - classif": m_base_clf,
            "XGBoost - classif": m_xgb_clf,
            "Baseline - RUL": m_base_rul,
            "XGBoost - RUL": m_xgb_rul,
        }
    )
    mo.md("### Résultats (test)\n\n" + table)
    return


@app.cell
def _(metrics, prep, xgb_clf):
    fig_imp = metrics.plot_feature_importance(
        prep.FEATURES,
        xgb_clf.feature_importances_,
        title="XGBoost - Importance des variables (classif)",
    )
    metrics.save_fig(fig_imp, "xgb_importance.png")
    fig_imp
    return


@app.cell
def _(metrics, xgb_reg, xt, y_rul):
    fig_sc = metrics.plot_rul_scatter(
        y_rul, xgb_reg.predict(xt), title="XGBoost - RUL prédit vs réel (test)"
    )
    metrics.save_fig(fig_sc, "xgb_rul_scatter.png")
    fig_sc
    return


@app.cell
def _(m_base_clf, m_base_rul, m_xgb_clf, m_xgb_rul, mo):
    mo.md(f"""
    ### Conclusion - comparaison
    - **Classification** : XGBoost (F1 = {m_xgb_clf["f1"]:.2f}, AUC =
      {m_xgb_clf.get("roc_auc", float("nan")):.2f}) contre la baseline logistique
      (F1 = {m_base_clf["f1"]:.2f}). L'écart mesure l'apport des non-linéarités et
      interactions entre capteurs que le modèle linéaire ne capte pas.
    - **RUL** : XGBoost (RMSE = {m_xgb_rul["RMSE"]:.1f}, score NASA =
      {m_xgb_rul["NASA"]:.0f}) contre la régression linéaire (RMSE =
      {m_base_rul["RMSE"]:.1f}). La baseline pose le plancher de performance ;
      le boosting réduit nettement l'erreur.
    - À reporter dans le **tableau de synthèse final** face à la Random Forest (01)
      et au LSTM (03).
    """)
    return


if __name__ == "__main__":
    app.run()
