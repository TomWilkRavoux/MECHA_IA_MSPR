import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    # Rendre le package `ml` (ml/) importable depuis notebooks/
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import joblib
    import marimo as mo
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from ml import metrics, prep

    return RandomForestClassifier, RandomForestRegressor, joblib, metrics, mo, np, prep


@app.cell
def _(mo):
    mo.md(
        r"""
        # 01 - Random Forest (cœur de la solution)

        **Pourquoi la Random Forest ?** Le CDC la désigne explicitement comme cœur de
        la solution : ensemble d'arbres robuste au bruit et aux valeurs manquantes,
        peu sensible à l'échelle des variables, et surtout **interprétable** via
        l'importance des variables - atout majeur pour la maintenance (identifier les
        capteurs prédictifs). Elle sert de **référence** pour les notebooks suivants.

        Ce notebook traite les **deux tâches** imposées :
        - **Classification** de l'état machine (`at_risk`, 1 si RUL ≤ 30) ;
        - **Régression** du RUL (temps restant avant défaillance, borné à 125 cycles).
        """
    )
    return


@app.cell
def _(mo, prep):
    # --- Chargement + split PAR MACHINE (anti-fuite) ---
    df_clf = prep.load_train_classification()
    df_rul = prep.load_train_rul()

    tr_clf, va_clf = prep.split_by_machine(df_clf, val_frac=0.2, seed=42)
    tr_rul, va_rul = prep.split_by_machine(df_rul, val_frac=0.2, seed=42)

    mo.md(
        f"**Données** - classif : {df_clf.height:,} lignes · "
        f"{df_clf['machine_id'].n_unique()} machines "
        f"(train {tr_clf['machine_id'].n_unique()} / val {va_clf['machine_id'].n_unique()}). "
        f"Taux de positifs `at_risk` : {100 * df_clf['at_risk'].mean():.1f} %."
    )
    return tr_clf, va_clf, tr_rul, va_rul


@app.cell
def _(prep, tr_clf, tr_rul):
    # --- Features / cibles + normalisation (ajustée sur le train uniquement) ---
    x_tr_clf, y_tr_clf = prep.to_xy(tr_clf, "at_risk")
    x_tr_rul, y_tr_rul = prep.to_xy(tr_rul, "RUL")
    y_tr_rul = prep.clip_rul(y_tr_rul)  # clipping piecewise à 125

    scaler = prep.fit_scaler(x_tr_clf, save_as="scaler.joblib")
    return scaler, x_tr_clf, x_tr_rul, y_tr_clf, y_tr_rul


@app.cell
def _(
    RandomForestClassifier,
    RandomForestRegressor,
    joblib,
    prep,
    scaler,
    x_tr_clf,
    x_tr_rul,
    y_tr_clf,
    y_tr_rul,
):
    # --- Construction + entraînement ---
    # class_weight="balanced" pour compenser le déséquilibre (~14 % de positifs).
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ).fit(scaler.transform(x_tr_clf), y_tr_clf)

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ).fit(scaler.transform(x_tr_rul), y_tr_rul)

    # --- Sauvegarde des modèles entraînés ---
    joblib.dump(rf_clf, prep.MODELS_DIR / "rf_classifier.joblib")
    joblib.dump(rf_reg, prep.MODELS_DIR / "rf_regressor.joblib")
    return rf_clf, rf_reg


@app.cell
def _(mo):
    mo.md(r"## Évaluation - Classification `at_risk`")
    return


@app.cell
def _(metrics, mo, prep, rf_clf, scaler, va_clf):
    # Validation (trajectoires complètes de machines jamais vues à l'entraînement)
    x_va_clf, y_va_clf = prep.to_xy(va_clf, "at_risk")
    x_va_clf_s = scaler.transform(x_va_clf)
    proba_clf = rf_clf.predict_proba(x_va_clf_s)[:, 1]
    pred_clf = rf_clf.predict(x_va_clf_s)

    m_clf = metrics.classification_metrics(y_va_clf, pred_clf, proba_clf)
    mo.md(
        "**Métriques classification (validation)**\n\n"
        + "\n".join(f"- **{k}** : {v:.3f}" for k, v in m_clf.items())
    )
    return m_clf, pred_clf, proba_clf, y_va_clf


@app.cell
def _(metrics, pred_clf, proba_clf, y_va_clf):
    fig_cm = metrics.plot_confusion(
        y_va_clf, pred_clf, title="RF - Confusion (at_risk)"
    )
    metrics.save_fig(fig_cm, "rf_confusion.png")
    fig_cm
    return


@app.cell
def _(metrics, proba_clf, y_va_clf):
    fig_roc = metrics.plot_roc(y_va_clf, proba_clf, title="RF - ROC (at_risk)")
    metrics.save_fig(fig_roc, "rf_roc.png")
    fig_roc
    return


@app.cell
def _(metrics, proba_clf, y_va_clf):
    fig_pr = metrics.plot_pr(
        y_va_clf, proba_clf, title="RF - Précision/Rappel (at_risk)"
    )
    metrics.save_fig(fig_pr, "rf_pr.png")
    fig_pr
    return


@app.cell
def _(metrics, prep, rf_clf):
    fig_imp = metrics.plot_feature_importance(
        prep.FEATURES,
        rf_clf.feature_importances_,
        title="RF - Importance des variables (classif)",
    )
    metrics.save_fig(fig_imp, "rf_importance.png")
    fig_imp
    return


@app.cell
def _(m_clf, mo):
    mo.md(
        f"""
        ### Conclusion - classification
        - **Recall = {m_clf["recall"]:.2f}** : part des machines réellement à risque
          correctement détectées. C'est la métrique **critique** pour MECHA - un faux
          négatif = panne manquée (arrêt non planifié, coûteux).
        - **Precision = {m_clf["precision"]:.2f}** : quand le modèle alerte, fiabilité de
          l'alerte. Trop bas ⇒ maintenance inutile.
        - **F1 = {m_clf["f1"]:.2f}** synthétise les deux ; **ROC-AUC = {m_clf.get("roc_auc", float("nan")):.2f}**
          mesure la séparabilité globale, indépendamment du seuil.
        - Le déséquilibre (~14 % de positifs) rend l'**accuracy** peu informative : on
          privilégie F1 / recall.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## Évaluation - Régression `RUL`")
    return


@app.cell
def _(metrics, mo, prep, rf_reg, scaler, va_rul):
    # Validation (rows) + jeu de test officiel (dernier cycle par machine vs RUL_true)
    x_va_rul, y_va_rul = prep.to_xy(va_rul, "RUL")
    y_va_rul_c = prep.clip_rul(y_va_rul)
    pred_va_rul = rf_reg.predict(scaler.transform(x_va_rul))
    m_rul_val = metrics.regression_metrics(y_va_rul_c, pred_va_rul)

    ev = prep.test_last_cycle_eval()
    x_test = ev.select(prep.FEATURES).to_numpy()
    y_test = ev["RUL_true"].to_numpy()
    pred_test_rul = rf_reg.predict(scaler.transform(x_test))
    m_rul_test = metrics.regression_metrics(y_test, pred_test_rul)

    mo.md(
        "**Régression RUL - validation** : "
        + " · ".join(f"{k} {v:.2f}" for k, v in m_rul_val.items())
        + "\n\n**Régression RUL - test officiel (dernier cycle)** : "
        + " · ".join(f"{k} {v:.2f}" for k, v in m_rul_test.items())
    )
    return m_rul_test, pred_test_rul, y_test


@app.cell
def _(metrics, pred_test_rul, y_test):
    fig_sc = metrics.plot_rul_scatter(
        y_test, pred_test_rul, title="RF - RUL prédit vs réel (test)"
    )
    metrics.save_fig(fig_sc, "rf_rul_scatter.png")
    fig_sc
    return


@app.cell
def _(metrics, pred_test_rul, y_test):
    fig_err = metrics.plot_error_hist(
        y_test, pred_test_rul, title="RF - Erreur de RUL (test)"
    )
    metrics.save_fig(fig_err, "rf_rul_error.png")
    fig_err
    return


@app.cell
def _(m_rul_test, mo):
    mo.md(
        f"""
        ### Conclusion - régression RUL
        - **MAE = {m_rul_test["MAE"]:.1f} cycles** : erreur moyenne en valeur absolue,
          directement lisible par la maintenance (« ± X cycles »).
        - **RMSE = {m_rul_test["RMSE"]:.1f}** : pénalise davantage les grosses erreurs.
        - **R² = {m_rul_test["R2"]:.2f}** : part de variance expliquée.
        - **Score NASA = {m_rul_test["NASA"]:.0f}** (plus bas = mieux) : pénalise les
          **retards** de prédiction (RUL surestimé = panne détectée trop tard). C'est
          l'indicateur le plus aligné sur le risque industriel.

        La Random Forest fixe la **référence**. Les notebooks 02 (baseline + XGBoost) et
        03 (LSTM) seront comparés à ces valeurs dans le tableau de synthèse final.
        """
    )
    return


@app.cell
def _(m_clf, m_rul_test, metrics, mo):
    table = metrics.format_metrics_table(
        {"RF - classif (at_risk)": m_clf, "RF - RUL (test)": m_rul_test}
    )
    mo.md("### Récapitulatif Random Forest\n\n" + table)
    return


if __name__ == "__main__":
    app.run()
