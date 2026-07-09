import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    return KMeans, Path, StandardScaler, mo, pl, plt


@app.cell
def _(Path, mo):
    DATA_DIR = Path("./assets/KaggleDataset/CMaps/")
    SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
    COLS = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    mo.md("### Exploration C-MAPSS - justification du découpage usines / lignes")
    return COLS, DATA_DIR, SUBSETS


@app.cell
def _(COLS, DATA_DIR, SUBSETS, pl):
    def load_subset(name: str) -> pl.DataFrame:
        raw = pl.read_csv(
            DATA_DIR / f"train_{name}.txt",
            separator=" ",
            has_header=False,
            truncate_ragged_lines=True,
        )
        # les fichiers ont un espace en fin de ligne -> on garde les 26 vraies colonnes
        raw = raw.select(raw.columns[:26])
        raw = raw.rename({old: new for old, new in zip(raw.columns, COLS)})
        return raw.with_columns(pl.lit(name).alias("subset"))

    df_all = pl.concat([load_subset(n) for n in SUBSETS])
    df_all.head()
    return (df_all,)


@app.cell
def _(mo):
    mo.md("""
    **Point de départ :** les 4 fichiers ont le même schéma (1 machine = 1 trajectoire
    *run-to-failure*, 21 capteurs, 3 réglages opératoires). Ce qui les différencie n'est
    donc *pas* la structure, mais leur **comportement**. On va isoler les deux axes qui
    fondent notre découpage.
    """)
    return


@app.cell
def _(df_all, pl):
    unit_life = df_all.group_by(["subset", "unit"]).agg(vie=pl.col("cycle").max())

    overview = (
        df_all.group_by("subset")
        .agg(
            n_lignes=pl.len(),
            n_machines=pl.col("unit").n_unique(),
        )
        .join(
            unit_life.group_by("subset").agg(
                vie_moy=pl.col("vie").mean().round(1),
                vie_min=pl.col("vie").min(),
                vie_max=pl.col("vie").max(),
                vie_std=pl.col("vie").std().round(1),
            ),
            on="subset",
        )
        .sort("subset")
    )
    overview
    return (unit_life,)


@app.cell
def _(mo):
    mo.md("""
    #### Axe 1 Régime de fonctionnement (justifie le découpage en *lignes*)
    Hypothèse : certaines machines tournent sur un **régime unique et stable**, d'autres
    sur **plusieurs régimes**. Si c'est vrai, ça se voit sur la dispersion des 3 réglages
    opératoires : quasi nulle pour un régime unique, forte pour des régimes multiples.
    """)
    return


@app.cell
def _(df_all, pl):
    op_var = (
        df_all.group_by("subset")
        .agg(
            std_op1=pl.col("op_setting_1").std().round(3),
            std_op2=pl.col("op_setting_2").std().round(3),
            std_op3=pl.col("op_setting_3").std().round(3),
        )
        .sort("subset")
    )
    op_var
    return


@app.cell
def _(KMeans, StandardScaler, df_all, pl):
    # Confirmation par cluster
    op_X = df_all.select(["op_setting_1", "op_setting_2", "op_setting_3"]).to_numpy()
    op_Xs = StandardScaler().fit_transform(op_X)
    op_km = KMeans(n_clusters=6, n_init=10, random_state=42).fit(op_Xs)

    df_reg = df_all.with_columns(pl.Series("regime", op_km.labels_))

    regimes_par_subset = (
        df_reg.group_by(["subset", "regime"])
        .agg(eff=pl.len())
        .with_columns(
            (pl.col("eff") / pl.col("eff").sum().over("subset")).alias("part")
        )
        .filter(pl.col("part") >= 0.01)  # on ignore les régimes marginaux (<1%)
        .group_by("subset")
        .agg(nb_regimes=pl.col("regime").n_unique())
        .sort("subset")
    )
    regimes_par_subset
    return


@app.cell
def _(SUBSETS, df_all, pl, plt):
    regime_sample = df_all.sample(n=8000, seed=1) if df_all.height > 8000 else df_all

    fig_regimes, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), SUBSETS):
        d = regime_sample.filter(pl.col("subset") == name)
        ax.scatter(d["op_setting_1"], d["op_setting_2"], s=4, alpha=0.4)
        ax.set_title(name)
        ax.set_xlabel("op_setting_1")
        ax.set_ylabel("op_setting_2")
    fig_regimes.suptitle(
        "Régimes opératoires : 1 nuage (FD001/FD003) vs 6 (FD002/FD004)"
    )
    fig_regimes.tight_layout()
    fig_regimes
    return


@app.cell
def _(SUBSETS, df_all, pl, plt):
    fig_sensor, axs = plt.subplots(2, 2, figsize=(9, 7))
    for axEff, nameEff in zip(axs.ravel(), SUBSETS):
        vals = df_all.filter(pl.col("subset") == nameEff)["sensor_2"].to_numpy()
        axEff.hist(vals, bins=60)
        axEff.set_title(nameEff)
    fig_sensor.suptitle(
        "Capteur 2 : unimodal (FD001/FD003) vs multimodal (FD002/FD004)"
    )
    fig_sensor.tight_layout()
    fig_sensor  # change l'indice du capteur pour explorer (sensor_3, sensor_4...)
    return


@app.cell
def _(mo):
    mo.md("""
    #### Axe 2 Modes de défaillance (justifie le découpage en *usines*)
    Contrairement au régime, **le mode de panne n'est pas une colonne du dataset** : il
    n'y a pas de label "type de panne". C'est une **caractéristique documentée** par la NASA
    (PCoE) : FD001 & FD002 = **1 mode** (dégradation HPC), FD003 & FD004 = **2 modes**
    (HPC + ventilateur).

    On ne peut donc pas le *prouver* sur les données brutes, mais on peut chercher un
    **signal de soutien** : deux mécanismes de panne coexistants devraient produire des
    durées de vie plus hétérogènes (dispersion plus forte) que pour un seul mode.
    """)
    return


@app.cell
def _(pl, unit_life):
    fault_signal = (
        unit_life.group_by("subset")
        .agg(
            vie_moy=pl.col("vie").mean().round(1),
            vie_std=pl.col("vie").std().round(1),
            cv=(pl.col("vie").std() / pl.col("vie").mean()).round(
                3
            ),  # coeff. de variation
        )
        .sort("subset")
    )
    fault_signal
    return


@app.cell
def _(mo):
    mo.md("""
    #### Conclusion : la grille 2×2 qui fonde le mapping

    |                | 1 mode de panne | 2 modes de panne |
    |----------------|-----------------|------------------|
    | **1 régime**   | FD001           | FD003            |
    | **6 régimes**  | FD002           | FD004            |

    - **Axe régime** : prouvé par les données (1 vs 6 régimes) -> caractéristique d'une
      **ligne** : ligne dédiée (régime unique) vs ligne polyvalente (régimes multiples).
    - **Axe modes de panne** = documenté NASA + signal de dispersion -> caractéristique d'une
      **usine** : site produisant des pièces simples (1 mode) vs complexes (2 modes).

    **Mapping retenu :**
    - **Usine A** (1 mode) -> Ligne 1 = FD001 (dédiée), Ligne 2 = FD002 (polyvalente)
    - **Usine B** (2 modes) -> Ligne 1 = FD003 (dédiée), Ligne 2 = FD004 (polyvalente)
    """)
    return


@app.cell
def _(df_all, pl):
    df_mecha = df_all.with_columns(
        [
            pl.when(pl.col("subset").is_in(["FD001", "FD002"]))
            .then(pl.lit("Usine A"))
            .otherwise(pl.lit("Usine B"))
            .alias("usine"),
            pl.when(pl.col("subset").is_in(["FD001", "FD003"]))
            .then(pl.lit("Ligne 1 (dédiée)"))
            .otherwise(pl.lit("Ligne 2 (polyvalente)"))
            .alias("ligne_production"),
        ]
    )
    df_mecha.select(["subset", "usine", "ligne_production"]).unique().sort(
        ["usine", "ligne_production"]
    )
    return


if __name__ == "__main__":
    app.run()
