import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import sweetviz as sv
    import os
    from pathlib import Path
    import altair as alt

    return Path, mo, pl, sv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse dataset donné
    """)
    return


@app.cell
def _(pl):
    df = pl.read_csv("./assets/2025-2026 EISI IA - Jeu donnees MSPR TPRE831.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, sv):
    report = sv.analyze(df.to_pandas())
    return (report,)


@app.cell
def _(report):
    report.show_notebook()
    return


@app.cell
def _(report):
    # save html
    report.show_html("./reports/sweetvizDataset100KBrut.html", open_browser=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse dataset clean generer
    """)
    return


@app.cell
def _():
    CSV_FOLDER = "./assets/output/"
    return (CSV_FOLDER,)


@app.cell
def _(CSV_FOLDER, Path, pl):
    csv_files = list(Path(CSV_FOLDER).glob("*.csv"))

    dfs = {}
    for f in csv_files:
        try:
            dfs[f.stem] = pl.read_csv(f, try_parse_dates=True, infer_schema_length=1000)
        except Exception as e:
            print(f"Erreur sur {f.name} : {e}")

    print(f"{len(dfs)} fichiers chargés : {list(dfs.keys())}")
    return (dfs,)


@app.cell
def _(dfs, mo):
    dataset_selector = mo.ui.dropdown(
        options=list(dfs.keys()),
        label="Choisir un CSV"
    )
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(dataset_selector, dfs, mo):
    def _():
        df = dfs[dataset_selector.value]
        return mo.md(f"##`{dataset_selector.value}` — {df.shape[0]:,} lignes × {df.shape[1]} colonnes")


    _()
    return


@app.cell
def _(dfs, mo):
    report_selector = mo.ui.dropdown(
        options=list(dfs.keys()),
        label="Choisir un CSV pour le rapport Sweetviz"
    )
    report_selector
    return (report_selector,)


@app.cell
def _(dfs, report_selector, sv):
    def _():
        df_pandas = dfs[report_selector.value].to_pandas()

        report = sv.analyze([df_pandas, report_selector.value])
        return report.show_notebook()


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
