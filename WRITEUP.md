# Synthèse (une page) — Classification du type d'un Pokémon

## 1. Source des données et choix
**PokéAPI** (https://pokeapi.co/) — API REST publique, gratuite, **sans clé ni inscription**.
Je l'ai choisie car elle est bien documentée, fiable, et illustre bien les points techniques
demandés : la liste est **paginée** (champ `next`), et chaque Pokémon nécessite un **second
appel de détail** dont il faut **aplatir le JSON imbriqué** (`stats[i].base_stat` → colonnes
`hp`, `attack`, …). J'ai récupéré **1000 Pokémon** (bien au-delà des 200 requis).

**Tâche :** prédire le **type principal** d'un Pokémon (18 classes) à partir de 10 features
numériques (les 6 stats de base + taille, poids, expérience de base, double-type ou non).

## 2. Décisions de nettoyage principales (détail dans `NOTES.md`)
- **Cible :** aucune ligne sans `type_1` (rien à supprimer).
- **Manquants :** `type_2` absent pour ~50 % des Pokémon (mono-type) → transformé en variable
  binaire `is_dual_type` plutôt que gardé tel quel. Les features numériques n'avaient pas de
  manquants ; imputation médiane prévue par sécurité (robuste à l'asymétrie).
- **Doublons :** aucun (id unique).
- **Outliers :** **conservés** — les stats très hautes sont des légendaires légitimes, pas des
  erreurs (jugement métier plutôt que suppression aveugle par IQR).
- **Classes rares :** `flying` est quasi toujours un type secondaire → trop rare comme type
  **principal**, regroupé dans une classe `other` pour ne pas casser le split stratifié.
- **Feature engineering :** `total_stats` (pour l'EDA) et `is_dual_type`. `total_stats` est
  **exclu du modèle** (colinéaire avec les 6 stats).

## 3. Résultats des modèles (jeu de test, 20 %, split stratifié)

| Modèle | Accuracy | Précision (macro) | Rappel (macro) | F1 (macro) |
|---|---|---|---|---|
| Régression logistique | 0.235 | 0.222 | 0.166 | 0.164 |
| **Random Forest** (retenu) | **0.255** | **0.213** | **0.214** | **0.203** |

**Modèle déployé : Random Forest**, choisi sur le **F1 macro** (0.203 vs 0.164) — plus pertinent
que l'accuracy vu le déséquilibre des classes (l'eau et le normal dominent). Les features les
plus importantes sont le **poids**, l'**expérience de base** et l'**attaque spéciale**.

## 4. Limite honnête et piste d'amélioration
**Limite :** l'accuracy (~26 %) est faible, mais **c'est attendu et normal** : les statistiques
de base se chevauchent énormément entre types (un Pokémon eau et un Pokémon plante peuvent avoir
des stats quasi identiques). Le signal utile est intrinsèquement faible — l'objectif du projet
est la **maîtrise du pipeline complet**, pas un score record. À titre de repère, un classifieur
aléatoire pondéré ferait bien pire, et prédire toujours la classe majoritaire plafonnerait vers
~14 %.

**Piste :** enrichir les features au-delà des seules stats — par ex. les **capacités**
(`abilities`), le **groupe d'œufs**, la **génération**, ou des indices dérivés du nom, qui
portent davantage d'information sur le type qu'un simple profil de stats.
