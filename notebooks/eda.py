"""
Task 2 - EDA (analyse exploratoire) et nettoyage des données.

Ce script (équivalent d'un notebook) :
  1. Charge data/raw_data.csv et fait un premier état des lieux (info, describe, NaN, doublons).
  2. Produit les graphiques d'EDA dans notebooks/figures/ :
       - équilibre des classes (type principal)
       - heatmap de corrélation des variables numériques
       - boxplots + scatter (relations feature / classe)
  3. Nettoie les données AVEC justification (voir NOTES.md) :
       - suppression des lignes sans cible
       - imputation médiane des features numériques manquantes
       - suppression des doublons
       - ingénierie de variables (total_stats, is_dual_type)
       - regroupement des types rares dans "other" (sinon le split stratifié casse)
  4. Enregistre data/clean_data.csv

Lancer :  python notebooks/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans fenêtre : on sauvegarde les figures en PNG
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAT_COLS = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
NUMERIC_FEATURES = STAT_COLS + ["height", "weight", "base_experience"]
MIN_CLASS_COUNT = 20  # en dessous, un type est jugé "rare" et regroupé dans "other"


def first_look(df):
    print("=" * 60)
    print("1) PREMIER ÉTAT DES LIEUX")
    print("=" * 60)
    print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes\n")
    print("--- df.info() ---")
    df.info()
    print("\n--- df.describe() ---")
    print(df.describe())
    print("\n--- Valeurs manquantes par colonne ---")
    print(df.isna().sum())
    print(f"\n--- Doublons exacts : {df.duplicated().sum()}")


def make_plots(df):
    print("\n" + "=" * 60)
    print("2) VISUALISATIONS (sauvegardées dans notebooks/figures/)")
    print("=" * 60)
    sns.set_style("whitegrid")

    # a) Équilibre des classes
    plt.figure(figsize=(10, 5))
    order = df["type_1"].value_counts().index
    sns.countplot(data=df, x="type_1", order=order, color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.title("Équilibre des classes - type principal (avant nettoyage)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png", dpi=110)
    plt.close()

    # b) Heatmap de corrélation
    plt.figure(figsize=(8, 6))
    corr = df[NUMERIC_FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Corrélations des variables numériques")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=110)
    plt.close()

    # c) Boxplot : attaque par type (top 8 types les plus fréquents)
    top_types = df["type_1"].value_counts().head(8).index
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=df[df["type_1"].isin(top_types)], x="type_1", y="attack",
                order=top_types, color="#DD8452")
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribution de l'attaque par type principal (top 8)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplot_attack_by_type.png", dpi=110)
    plt.close()

    # d) Scatter attaque vs défense coloré par quelques types
    subset = df[df["type_1"].isin(["water", "fire", "grass", "rock"])]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=subset, x="attack", y="defense", hue="type_1", alpha=0.8)
    plt.title("Attaque vs Défense (4 types) - les stats se chevauchent beaucoup")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_attack_defense.png", dpi=110)
    plt.close()

    print("4 figures enregistrées :",
          "class_balance.png, correlation_heatmap.png,",
          "boxplot_attack_by_type.png, scatter_attack_defense.png")


def clean(df):
    print("\n" + "=" * 60)
    print("3) NETTOYAGE (justifications détaillées dans NOTES.md)")
    print("=" * 60)
    n0 = len(df)

    # 3.1 Cible manquante -> inutilisable pour l'entraînement, on supprime
    df = df.dropna(subset=["type_1"]).copy()
    print(f"- Lignes sans type_1 supprimées : {n0 - len(df)}")

    # 3.2 Ingénierie de variables
    df["total_stats"] = df[STAT_COLS].sum(axis=1)              # puissance globale
    df["is_dual_type"] = df["type_2"].notna().astype(int)      # 1 si double type
    print("- Nouvelles features : total_stats, is_dual_type")

    # 3.3 Imputation médiane des features numériques manquantes (distributions asymétriques)
    for col in NUMERIC_FEATURES:
        n_missing = df[col].isna().sum()
        if n_missing:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"- {col}: {n_missing} valeurs imputées par la médiane ({median})")

    # 3.4 Doublons exacts
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"- Doublons supprimés : {n_before - len(df)}")

    # 3.5 Outliers : on GARDE les stats extrêmes (légendaires) -> info utile, pas des erreurs
    print("- Outliers : conservés (les légendaires ont des stats hautes légitimes)")

    # 3.6 Regroupement des types rares
    counts = df["type_1"].value_counts()
    rare = counts[counts < MIN_CLASS_COUNT].index.tolist()
    if rare:
        df["type_1"] = df["type_1"].where(~df["type_1"].isin(rare), other="other")
        print(f"- Types rares (<{MIN_CLASS_COUNT}) regroupés dans 'other' : {rare}")

    print(f"\nRésultat : {len(df)} lignes, {df['type_1'].nunique()} classes cibles")
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "raw_data.csv")
    first_look(df)
    make_plots(df)
    df_clean = clean(df)
    out = DATA_DIR / "clean_data.csv"
    df_clean.to_csv(out, index=False)
    print(f"\nOK -> données nettoyées enregistrées dans {out}")
