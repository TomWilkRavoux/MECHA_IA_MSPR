import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path

    return Path, mo, pl


@app.cell
def _(Path, mo, pl):
    DATA_DIR = Path("./assets/KaggleDataset/CMapsCsv/")

    # subset -> fichier source
    CLF_FILES = {
        "FD001": "train_clf_clean.csv",
        "FD002": "train_clf_FD002.csv",
        "FD003": "train_clf_FD003_clean.csv",
        "FD004": "train_clf_FD004_clean.csv",
    }
    RUL_FILES = {
        "FD001": "train_rul_FD001_clean.csv",
        "FD002": "train_rul_FD002_clean.csv",
        "FD003": "train_rul_FD003_clean.csv",
        "FD004": "train_rul_FD004_clean.csv",
    }

    TEST_FILES = {
        "FD001": "test_FD001_clean.csv",
        "FD002": "test_FD002_clean.csv",
        "FD003": "test_FD003_clean.csv", 
        "FD004": "test_FD004_clean.csv", 
    }

    RUL_TRUE = {
        "FD001": "rul_true_FD001_clean.csv",
        "FD002": "rul_true_FD002.csv",
        "FD003": "rul_true_FD003_clean.csv", 
        "FD004": "rul_true_FD004_clean.csv", 
    }

    def add_ids(df: pl.DataFrame, name: str) -> pl.DataFrame:
        usine = "Usine A" if name in ("FD001", "FD002") else "Usine B"
        ligne = "Ligne 1 (dédiée)" if name in ("FD001", "FD003") else "Ligne 2 (polyvalente)"
        return df.with_columns([
            pl.lit(name).alias("subset"),
            pl.lit(usine).alias("usine"),
            pl.lit(ligne).alias("ligne_production"),
            (pl.lit(name) + "_u" + pl.col("unit").cast(pl.Utf8).str.zfill(3)).alias("machine_id"),
        ])

    def reorder(df: pl.DataFrame) -> pl.DataFrame:
        front = [c for c in ["machine_id", "subset", "usine", "ligne_production", "unit", "cycle"] if c in df.columns]
        return df.select(front + [c for c in df.columns if c not in front])

    def build(files: dict) -> pl.DataFrame:
        return pl.concat(
            [reorder(add_ids(pl.read_csv(DATA_DIR / f), name)) for name, f in files.items()],
            how="vertical",
        )

    mo.md("### Consolidation des jeux d'entraînement MECHA")
    return CLF_FILES, RUL_FILES, RUL_TRUE, TEST_FILES, build


@app.cell
def _(CLF_FILES, build):
    df_clf = build(CLF_FILES)
    df_clf
    return (df_clf,)


@app.cell
def _(RUL_FILES, build):
    df_rul = build(RUL_FILES)
    df_rul
    return (df_rul,)


@app.cell
def _(df_clf, df_rul, mo, pl):
    controle = (
        df_clf.group_by(["usine", "ligne_production"]).agg(
            machines=pl.col("machine_id").n_unique(),
            lignes=pl.len(),
            taux_at_risk=(100 * pl.col("at_risk").mean()).round(1),
        )
        .sort(["usine", "ligne_production"])
    )
    mo.vstack([
        mo.md(f"**Classification** : {df_clf.height} lignes · {df_clf['machine_id'].n_unique()} machines · "
              f"RUL {df_rul['RUL'].min()}→{df_rul['RUL'].max()}"),
        controle,
    ])
    return


@app.cell
def _(df_clf, df_rul, mo):
    df_clf.write_csv("./assets/KaggleDataset/mecha_train_classification.csv")
    df_rul.write_csv("./assets/KaggleDataset/mecha_train_rul.csv")
    mo.md("`mecha_train_classification.csv` et `mecha_train_rul.csv` écrits.")
    return


@app.cell
def _(TEST_FILES, build):
    df_test = build(TEST_FILES)
    df_test
    return (df_test,)


@app.cell
def _(RUL_TRUE, build):
    df_rul_true = build(RUL_TRUE)
    df_rul_true
    return (df_rul_true,)


@app.cell
def _(df_rul_true, df_test, mo):
    df_test.write_csv("./assets/KaggleDataset/mecha_test_classification.csv")
    df_rul_true.write_csv("./assets/KaggleDataset/mecha_rul_true.csv")
    mo.md("`mecha_test_classification.csv` et `mecha_rul_true.csv` écrits.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
