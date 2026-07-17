# Classification du type d'un Pokémon (PokéAPI → Dashboard déployé)

Mini-projet data science de bout en bout : récupération de données via une **API publique
sans clé** (la [PokéAPI](https://pokeapi.co/)), nettoyage/EDA, entraînement d'un modèle de
classification, et **dashboard interactif Streamlit** déployé en ligne.

**Objectif du modèle :** prédire le *type principal* d'un Pokémon (eau, feu, plante, …)
à partir de ses statistiques de base (HP, attaque, défense, vitesse, etc.).

## Structure du projet

```
api-classification-dashboard/
├── data/
│   ├── raw_data.csv          # données brutes récupérées via l'API (reproductible)
│   └── clean_data.csv        # données nettoyées
├── notebooks/
│   ├── eda.py                # EDA + nettoyage (équivalent notebook)
│   └── figures/              # graphiques d'EDA (PNG)
├── fetch_data.py             # Task 1 : récupération via l'API
├── train_model.py            # Task 3 : entraînement + comparaison des modèles
├── app.py                    # Task 4 : dashboard Streamlit
├── model.pkl                 # meilleur modèle entraîné
├── scaler.pkl                # StandardScaler
├── label_encoder.pkl         # LabelEncoder de la cible
├── model_metrics.json        # métriques + infos pour le dashboard
├── requirements.txt
├── NOTES.md                  # justification des décisions de nettoyage
├── WRITEUP.md                # synthèse d'une page
└── README.md
```

## Installation et exécution en local

```bash
# 1. (optionnel) créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. installer les dépendances
pip install -r requirements.txt

# 3. reconstruire tout le pipeline (dans l'ordre)
python fetch_data.py          # -> data/raw_data.csv
python notebooks/eda.py       # -> data/clean_data.csv + figures
python train_model.py         # -> model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json

# 4. lancer le dashboard
streamlit run app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`.

## Déploiement

Application déployable gratuitement sur **Streamlit Community Cloud** ou **Hugging Face Spaces**
(voir la procédure dans `WRITEUP.md`). Tous les artefacts (`.pkl`, `.json`, CSV) sont versionnés
pour que l'app se lance sans réentraîner le modèle.

> URL publique : _(à compléter après déploiement)_

## Source des données

[PokéAPI](https://pokeapi.co/) — API REST publique, gratuite, sans authentification.
