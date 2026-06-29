import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import sweetviz as sv

    return mo, np, pl, sv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset train FD002 analyse
    """)
    return


@app.cell
def _():
    columns = [
        "unit", "cycle",
        "setting_1", "setting_2", "setting_3",
        "T2", "T24", "T30", "T50",
        "P2", "P15", "P30",
        "Nf", "Nc", "epr", "Ps30", "phi",
        "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd",
        "W31", "W32",
    ]
    return (columns,)


@app.cell
def _(columns, pl):
    train = pl.read_csv("./assets/KaggleDataset/CMaps/train_FD002.txt", separator=" ", has_header=False, truncate_ragged_lines=True).drop(["column_27", "column_28"])

    train.columns = columns

    return (train,)


@app.cell
def _(train):
    train
    return


@app.cell
def _(mo, train):
    mo.ui.table(train.describe())
    return


@app.cell
def _(pl, train):
    def NewColumnsMaxCycle():
        max_cycles = train.group_by("unit").agg(pl.col("cycle").max().alias("max_cycle"))

        train_rul = (
            train.join(max_cycles, on="unit").with_columns((pl.col("max_cycle") - pl.col("cycle")).alias("RUL")).drop("max_cycle")
        )
        return train_rul

    train_rul = NewColumnsMaxCycle()
    train_rul
    return (train_rul,)


@app.cell
def _():
    return


@app.cell
def _(mo, train_rul):
    train_rul.write_csv("./assets/KaggleDataset/train_rul_FD002_clean.csv")
    mo.md("**Fichier exporté :** train_rul_FD002_clean.csv")
    return


@app.cell
def _(sv, train_rul):
    reporTrainNasaFD002 = sv.analyze(train_rul.to_pandas())
    reporTrainNasaFD002.show_html(filepath="./reports/reporTrainNasaFD002.html", open_browser=False)
    reporTrainNasaFD002.show_notebook()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse Dataset test FD002
    """)
    return


@app.cell
def _(columns, pl):
    test = pl.read_csv("./assets/KaggleDataset/CMaps/test_FD002.txt", separator=" ", has_header=False, truncate_ragged_lines=True).drop(["column_27", "column_28"])

    test.columns = columns
    rul_true = pl.read_csv("./assets/KaggleDataset/CMaps/RUL_FD002.txt", separator=" ", has_header=False, truncate_ragged_lines=True)

    rul_true = rul_true.select(pl.col("column_1").alias("RUL_true"))
    rul_true = rul_true.with_row_index("unit", offset=1)

    test, rul_true
    return rul_true, test


@app.cell
def _():
    return


@app.cell
def _(mo, test):
    test.write_csv("./assets/KaggleDataset/test_FD002_clean.csv")
    mo.md("**FIchier exporte :** test_FD002_clean.csv")
    return


@app.cell
def _(mo, rul_true):
    rul_true.write_csv("./assets/KaggleDataset/rul_true_FD002.csv")
    mo.md("**Fichier exporté :** rul_rue_FD002.csv")
    return


@app.cell
def _(columns, mo, train):
    # Certains capteurs sont quasi-constants sur FD001 donc à virer
    sensor_cols = [c for c in columns if c not in ["unit", "cycle", "setting_1", "setting_2", "setting_3"]]

    # Écart-type par capteur
    stds = train.select(sensor_cols).std()
    mo.md(f"**Écart-types des capteurs :**")
    mo.ui.table(stds)
    return (sensor_cols,)


@app.cell
def _(mo, pl, sensor_cols, train):
    # Garder uniquement les capteurs avec std > 0.01
    useful_sensors = [
        col for col in sensor_cols
        if train.select(pl.col(col).std()).item() > 0.01
    ]
    mo.md(f"**Capteurs retenus ({len(useful_sensors)}) :** {useful_sensors}")
    return (useful_sensors,)


@app.cell
def _(mo, pl, train_rul):
    SEUIL_RUL = 30  # cycles

    train_clf = train_rul.with_columns((pl.col("RUL") <= SEUIL_RUL).cast(pl.Int8).alias("at_risk"))

    # Distribution de la cible
    counts = train_clf.group_by("at_risk").len()
    mo.md(f"**Distribution de la cible (seuil={SEUIL_RUL} cycles) :**")
    mo.ui.table(counts)
    return SEUIL_RUL, train_clf


@app.cell
def _(mo, train_clf):
    train_clf
    train_clf.write_csv("./assets/KaggleDataset/train_clf_FD002.csv")
    mo.md("**FIchier exporté :** train_clf_FD002")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Préparation features pour Random Forest
    """)
    return


@app.cell
def _(mo, train_clf, useful_sensors):
    feature_cols = useful_sensors + ["setting_1", "setting_2", "setting_3"]

    X_train = train_clf.select(feature_cols).to_numpy()
    y_train_rul = train_clf.select("RUL").to_numpy().ravel()  #Regression
    y_train_cls = train_clf.select("at_risk").to_numpy().ravel()  #Classification

    mo.md(f"**X_train shape:** {X_train.shape}  \n**y_train RUL:** {y_train_rul.shape}  \n**y_train classification:** {y_train_cls.shape}")
    return X_train, feature_cols, y_train_cls, y_train_rul


@app.cell
def _(X_train, mo, y_train_cls, y_train_rul):
    #Train/Test Split
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_cls_tr, y_cls_val, y_rul_tr, y_rul_val = train_test_split(
        X_train, y_train_cls, y_train_rul,
        test_size=0.2,
        random_state=42
    )
    mo.md(f"**Train:** {X_tr.shape[0]} lignes — **Val:** {X_val.shape[0]} lignes")
    return X_tr, X_val, y_cls_tr, y_cls_val, y_rul_tr, y_rul_val


@app.cell
def _(SEUIL_RUL, X_tr, X_val, mo, y_cls_tr, y_cls_val):
    # Random Forest Classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf_clf.fit(X_tr, y_cls_tr)

    y_cls_pred = rf_clf.predict(X_val)

    report = classification_report(y_cls_val, y_cls_pred)
    mo.md(f"### Classification (seuil RUL={SEUIL_RUL} cycles)\n```\n{report}\n```")
    return confusion_matrix, rf_clf, y_cls_pred


@app.cell
def _(confusion_matrix, mo, pl, y_cls_pred, y_cls_val):
    cm = confusion_matrix(y_cls_val, y_cls_pred)
    cm_df = pl.DataFrame({
        "": ["Prédit Normal", "Prédit À risque"],
        "Réel Normal": [cm[0][0], cm[1][0]],
        "Réel À risque": [cm[0][1], cm[1][1]],
    })
    mo.ui.table(cm_df)
    return


@app.cell
def _(pl, train_clf):
    RUL_CAP = 125

    train_capped = train_clf.with_columns(
        pl.col("RUL").clip(upper_bound=RUL_CAP)
    )

    y_train_rul_capped = train_capped.select("RUL").to_numpy().ravel()
    return


@app.cell
def _(X_tr, X_val, mo, np, y_rul_tr, y_rul_val):
    # Random forest Regression (pred RUL)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    y_rul_tr_cap = np.clip(y_rul_tr, a_min=None, a_max=125)
    y_rul_val_cap = np.clip(y_rul_val, a_min=None, a_max=125)

    rf_reg.fit(X_tr, y_rul_tr_cap)
    y_rul_pred = rf_reg.predict(X_val)

    mae = mean_absolute_error(y_rul_val_cap, y_rul_pred)
    rmse = root_mean_squared_error(y_rul_val_cap, y_rul_pred)
    r2 = r2_score(y_rul_val_cap, y_rul_pred)

    mo.md(f"""### Régression RUL
    - **MAE :** {mae:.1f} cycles
    - **RMSE :** {rmse:.1f} cycles
    - **R² :** {r2:.3f}
    """)
    return (rf_reg,)


@app.cell
def _(feature_cols, mo, pl, rf_reg):
    importances = rf_reg.feature_importances_

    fi_df = (
        pl.DataFrame({"feature": feature_cols, "importance": importances})
        .sort("importance", descending=True)
    )

    mo.ui.table(fi_df)
    return


@app.cell
def _(mo, rf_clf, rf_reg):
    import joblib

    joblib.dump(rf_clf, "rf_classifier_FD002.joblib")
    joblib.dump(rf_reg, "rf_regressor_FD002.joblib")

    mo.md("**Modèles exportés :** `rf_classifier.joblib`, `rf_regressor.joblib`")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
