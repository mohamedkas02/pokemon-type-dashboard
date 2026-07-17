"""
Task 4 - Dashboard interactif Streamlit.

5 sections (navigation dans la barre latérale) :
  1. Accueil / Intro
  2. Aperçu des données (brut vs nettoyé)
  3. EDA interactive (graphiques Plotly + filtre latéral)
  4. Performance des modèles (table, matrice de confusion, importance des features)
  5. Prédiction en direct (sliders -> type prédit + probabilités)

Lancer en local :  streamlit run app.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_SOURCE_URL = "https://pokeapi.co/"
LAST_UPDATED = "2026-07-17"

st.set_page_config(page_title="Classification de types Pokémon", page_icon="🎮", layout="wide")


# --------------------------------------------------------------------------
# Chargements mis en cache (une seule fois par session)
# --------------------------------------------------------------------------
@st.cache_data
def load_data(name):
    return pd.read_csv(APP_DIR / "data" / name)


@st.cache_data
def load_metrics():
    with open(APP_DIR / "model_metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_artifacts():
    model = joblib.load(APP_DIR / "model.pkl")
    scaler = joblib.load(APP_DIR / "scaler.pkl")
    le = joblib.load(APP_DIR / "label_encoder.pkl")
    return model, scaler, le


raw_df = load_data("raw_data.csv")
clean_df = load_data("clean_data.csv")
metrics = load_metrics()
model, scaler, label_encoder = load_artifacts()

FEATURES = metrics["features"]
CLASS_LABELS = metrics["class_labels"]


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
st.sidebar.title("🎮 Navigation")
section = st.sidebar.radio(
    "Aller à la section :",
    ["1. Accueil", "2. Aperçu des données", "3. EDA interactive",
     "4. Performance des modèles", "5. Prédiction en direct"],
)


# ==========================================================================
# 1. ACCUEIL
# ==========================================================================
if section == "1. Accueil":
    st.title("Prédire le type d'un Pokémon à partir de ses statistiques")
    st.markdown(
        """
        Ce dashboard illustre un pipeline data science complet, de bout en bout :
        **API publique → nettoyage → modèle de classification → application déployée**.

        Les données proviennent de la [**PokéAPI**](https://pokeapi.co/) (aucune clé requise).
        On cherche à prédire le **type principal** d'un Pokémon (eau, feu, plante, …)
        uniquement à partir de ses caractéristiques numériques (HP, attaque, défense, etc.).
        """
    )
    best = metrics["best_model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pokémon (nettoyés)", metrics["n_rows"])
    c2.metric("Classes cibles", len(CLASS_LABELS))
    c3.metric("Meilleur modèle", best.replace("_", " "))
    c4.metric("F1 macro", f"{metrics['all_model_results'][best]['f1']:.2f}")

    st.info(
        "Prédire un type à partir des seules stats est volontairement difficile : "
        "les statistiques se chevauchent beaucoup entre types. Une précision modeste "
        "est donc attendue — l'objectif est le pipeline, pas un score record."
    )


# ==========================================================================
# 2. APERÇU DES DONNÉES
# ==========================================================================
elif section == "2. Aperçu des données":
    st.title("Aperçu des données")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes (brut)", raw_df.shape[0])
    c2.metric("Lignes (nettoyé)", clean_df.shape[0])
    c3.metric("Colonnes (nettoyé)", clean_df.shape[1])

    tab1, tab2 = st.tabs(["Données brutes", "Données nettoyées"])
    with tab1:
        st.dataframe(raw_df.head(50), width="stretch")
    with tab2:
        st.dataframe(clean_df.head(50), width="stretch")

    st.subheader("Valeurs manquantes : avant vs après nettoyage")
    miss = pd.DataFrame({
        "avant (brut)": raw_df.isna().sum(),
        "après (nettoyé)": clean_df.reindex(columns=raw_df.columns).isna().sum(),
    }).fillna(0).astype(int)
    st.dataframe(miss, width="stretch")


# ==========================================================================
# 3. EDA INTERACTIVE
# ==========================================================================
elif section == "3. EDA interactive":
    st.title("Analyse exploratoire interactive")

    # Filtre latéral : sélection des types affichés (met les graphiques à jour en direct)
    all_types = sorted(clean_df["type_1"].unique())
    selected = st.sidebar.multiselect(
        "Filtrer par type principal :", all_types, default=all_types
    )
    view = clean_df[clean_df["type_1"].isin(selected)] if selected else clean_df

    st.subheader("Équilibre des classes")
    counts = view["type_1"].value_counts().reset_index()
    counts.columns = ["type_1", "count"]
    fig_bar = px.bar(counts, x="type_1", y="count", color="type_1",
                     title="Nombre de Pokémon par type principal")
    st.plotly_chart(fig_bar, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Heatmap de corrélation")
        num_cols = [c for c in FEATURES if c != "is_dual_type"]
        corr = view[num_cols].corr()
        fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_heat, width="stretch")
    with col2:
        st.subheader("Attaque vs Défense")
        fig_sc = px.scatter(view, x="attack", y="defense", color="type_1",
                            hover_data=["name"], opacity=0.75)
        st.plotly_chart(fig_sc, width="stretch")

    st.subheader("Distribution d'une statistique par type")
    stat = st.selectbox("Statistique :",
                        ["attack", "defense", "hp", "speed",
                         "special_attack", "special_defense", "total_stats"])
    fig_box = px.box(view, x="type_1", y=stat, color="type_1")
    st.plotly_chart(fig_box, width="stretch")


# ==========================================================================
# 4. PERFORMANCE DES MODÈLES
# ==========================================================================
elif section == "4. Performance des modèles":
    st.title("Performance des modèles")

    st.subheader("Comparaison des modèles")
    res = pd.DataFrame(metrics["all_model_results"]).T
    res.index.name = "modèle"
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#b6e3b6"),
                 width="stretch")
    st.caption(f"Modèle retenu (meilleur F1 macro) : **{metrics['best_model']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matrice de confusion")
        cm = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(cm, x=CLASS_LABELS, y=CLASS_LABELS, text_auto=True,
                          color_continuous_scale="Blues",
                          labels=dict(x="Prédit", y="Réel", color="Nombre"))
        st.plotly_chart(fig_cm, width="stretch")
    with col2:
        st.subheader("Importance des features")
        fi = pd.DataFrame(
            {"feature": list(metrics["feature_importance"].keys()),
             "importance": list(metrics["feature_importance"].values())}
        ).sort_values("importance", ascending=True)
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig_fi, width="stretch")


# ==========================================================================
# 5. PRÉDICTION EN DIRECT
# ==========================================================================
elif section == "5. Prédiction en direct":
    st.title("Prédiction en direct")
    st.markdown("Réglez les statistiques puis cliquez sur **Prédire le type**.")

    stats_info = metrics["feature_stats"]
    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        target_col = cols[i % 2]
        info = stats_info[feat]
        if feat == "is_dual_type":
            choice = target_col.selectbox("Double type ?", ["Non", "Oui"])
            inputs[feat] = 1 if choice == "Oui" else 0
        else:
            inputs[feat] = target_col.slider(
                feat, min_value=float(info["min"]), max_value=float(info["max"]),
                value=float(info["median"]),
            )

    if st.button("Prédire le type", type="primary"):
        x = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
        x_scaled = scaler.transform(x)
        pred_idx = model.predict(x_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.success(f"Type principal prédit : **{pred_label}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_scaled)[0]
            proba_df = pd.DataFrame({"type": CLASS_LABELS, "probabilité": proba}) \
                .sort_values("probabilité", ascending=False).head(10)
            fig = px.bar(proba_df, x="probabilité", y="type", orientation="h",
                        title="Probabilités par type (top 10)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Pied de page
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Source : [PokéAPI]({DATA_SOURCE_URL}) · Dernière mise à jour : {LAST_UPDATED}"
)
