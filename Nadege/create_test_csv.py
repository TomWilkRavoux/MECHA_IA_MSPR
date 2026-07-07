# create_test_csv.py
import pandas as pd
import numpy as np
from pathlib import Path

SOURCE = Path("/home/nadege/Downloads/MSPR  2")
DEST   = Path("/home/nadege/Downloads/MECHA_IA_MSPR/Tom/assets/KaggleDataset")

# Mapping complet — 4 subsets
MAPPING = {
    "FD001": ("Usine A", "Ligne 1 (dédiée)"),
    "FD002": ("Usine A", "Ligne 2 (polyvalente)"),
    "FD003": ("Usine B", "Ligne 1 (dédiée)"),
    "FD004": ("Usine B", "Ligne 2 (polyvalente)"),
}

dfs = []
for fd, (usine, ligne) in MAPPING.items():
    df = pd.read_excel(SOURCE / f"test_{fd}.xlsx")
    df.insert(0, "machine_id", fd + "_u" + df["unit_id"].astype(str).str.zfill(3))
    df.insert(1, "subset", fd)
    df.insert(2, "usine", usine)
    df.insert(3, "ligne_production", ligne)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Simule des pannes sur 20% des machines de chaque subset
np.random.seed(42)
dfs_normales = []
dfs_pannes   = []

for fd in MAPPING:
    sub = df_all[df_all["subset"] == fd]
    machines = sub["machine_id"].unique()
    n_panne  = max(1, int(len(machines) * 0.2))
    machines_en_panne = np.random.choice(machines, size=n_panne, replace=False)

    dfs_normales.append(sub[~sub["machine_id"].isin(machines_en_panne)])

    for m in machines_en_panne:
        # Garde seulement les 5 derniers cycles → machine en fin de vie
        dfs_pannes.append(sub[sub["machine_id"] == m].tail(5))

    print(f"{fd} — {len(machines)} machines · {n_panne} en fin de vie simulée")

df_export = pd.concat(dfs_normales + dfs_pannes, ignore_index=True)

out = DEST / "mecha_test_pannes_simulees.csv"
df_export.to_csv(out, index=False)

print(f"\nCSV exporté : {out}")
print(f"Machines totales  : {df_export['machine_id'].nunique()}")
print(f"Lignes totales    : {len(df_export):,}")
print(f"Usines            : {df_export['usine'].unique()}")
print(f"Lignes production : {df_export['ligne_production'].unique()}")
