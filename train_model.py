"""
Task 3 - Entraînement et comparaison de modèles de classification.

Objectif : prédire le TYPE PRINCIPAL d'un Pokémon à partir de ses stats.

Étapes :
  1. Préparation X / y + encodage de la cible (LabelEncoder).
  2. Split train/test stratifié.
  3. Standardisation des features (utile pour la régression logistique).
  4. Entraînement de 2 modèles : régression logistique (baseline) + random forest.
  5. Évaluation (accuracy + précision/rappel/F1 macro + matrice de confusion).
  6. Sauvegarde des artefacts pour le dashboard :
       model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json

Lancer :  python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent

# 10 features numériques (total_stats volontairement exclu car colinéaire avec les 6 stats)
FEATURES = ["hp", "attack", "defense", "special_attack", "special_defense", "speed",
            "height", "weight", "base_experience", "is_dual_type"]
TARGET = "type_1"


def main():
    df = pd.read_csv(ROOT / "data" / "clean_data.csv")

    # 1) Features / cible ---------------------------------------------------
    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"{len(le.classes_)} classes : {list(le.classes_)}")

    # 2) Split stratifié ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 3) Standardisation ----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Modèles ------------------------------------------------------------
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, random_state=42, class_weight="balanced"
        ),
    }

    # 5) Entraînement + évaluation -----------------------------------------
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        fitted[name] = model
        print(f"\n{name}")
        print(f"  accuracy={acc:.3f}  precision(macro)={prec:.3f}  "
              f"recall(macro)={rec:.3f}  f1(macro)={f1:.3f}")

    # Meilleur modèle = meilleur F1 macro (mieux que l'accuracy si classes déséquilibrées)
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted[best_name]
    print(f"\n>>> Meilleur modèle (F1 macro) : {best_name}")

    # Matrice de confusion (labels forcés = une ligne/colonne par classe)
    best_preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_preds, labels=range(len(le.classes_)))

    # Importance des features (feature_importances_ pour RF, |coef| moyen pour LR)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_).mean(axis=0)
    importances = importances / importances.sum()  # normalisées pour comparer
    feature_importance = {f: float(v) for f, v in zip(FEATURES, importances)}

    # Bornes des sliders du dashboard (min/max/médiane par feature)
    feature_stats = {
        f: {"min": float(df[f].min()), "max": float(df[f].max()),
            "median": float(df[f].median())}
        for f in FEATURES
    }

    # 6) Sauvegarde des artefacts ------------------------------------------
    joblib.dump(best_model, ROOT / "model.pkl")
    joblib.dump(scaler, ROOT / "scaler.pkl")
    joblib.dump(le, ROOT / "label_encoder.pkl")

    with open(ROOT / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "all_model_results": results,
            "best_model": best_name,
            "confusion_matrix": cm.tolist(),
            "class_labels": le.classes_.tolist(),
            "feature_importance": feature_importance,
            "feature_stats": feature_stats,
            "features": FEATURES,
            "n_rows": int(len(df)),
        }, f, indent=2)

    print("\nArtefacts enregistrés : model.pkl, scaler.pkl, label_encoder.pkl, "
          "model_metrics.json")


if __name__ == "__main__":
    main()
