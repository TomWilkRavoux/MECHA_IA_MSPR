import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    df_prod = pl.read_csv("./assets/output/production_clean.csv")
    df_alerte = pl.read_csv("./assets/output/alertes.csv")
    df_approviso = pl.read_csv("./assets/output/approvisionnement_clean.csv")
    df_energie = pl.read_csv("./assets/output/energie_clean.csv")
    df_etat_machine = pl.read_csv("./assets/output/etat_machine_clean.csv")
    df_maintenance = pl.read_csv("./assets/output/maintenance_clean.csv")
    df_qualite = pl.read_csv("./assets/output/qualite_clean.csv")
    return (
        df_alerte,
        df_approviso,
        df_energie,
        df_etat_machine,
        df_maintenance,
        df_prod,
        df_qualite,
    )


@app.cell
def _(
    df_alerte,
    df_approviso,
    df_energie,
    df_etat_machine,
    df_maintenance,
    df_prod,
    df_qualite,
):
    dfs = [df_alerte, df_prod, df_approviso, df_energie, df_etat_machine, df_maintenance, df_qualite]
    return (dfs,)


@app.cell
def _(dfs):
    for i in range(7):
        dfAf = dfs[i]
        print(dfAf.columns)
    

    return


if __name__ == "__main__":
    app.run()
