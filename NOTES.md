# Notes de nettoyage (Task 2) — justification des décisions

Ces décisions sont appliquées dans `notebooks/eda.py`.

### 1. Lignes sans cible (`type_1`)
Un classifieur ne peut pas apprendre sans étiquette : les rares lignes sans type principal
sont **supprimées** (elles sont inutilisables pour l'entraînement).

### 2. Valeurs manquantes des features numériques
Quelques Pokémon n'ont pas de `base_experience` (formes spéciales récentes). Comme les
distributions des stats sont **asymétriques** (présence de légendaires très forts), on
impute par la **médiane** plutôt que la moyenne (la médiane est robuste aux valeurs extrêmes).
Aucune colonne n'a assez de valeurs manquantes pour justifier sa suppression (< 40–50 %).

### 3. Doublons
Suppression des **doublons exacts** (`drop_duplicates`). L'`id` étant unique par Pokémon,
on n'en attend pas, mais on vérifie par sécurité.

### 4. Outliers — conservés volontairement
Les stats très élevées correspondent le plus souvent aux **Pokémon légendaires** : ce sont
de vraies valeurs, pas des erreurs de saisie. On les **garde** (jugement métier) plutôt que
de les retirer via l'IQR, car elles portent de l'information utile pour la classification.

### 5. Ingénierie de variables
- `total_stats` = somme des 6 stats (puissance globale) — utilisée pour l'EDA/visualisation.
- `is_dual_type` = 1 si le Pokémon a un second type, 0 sinon — variable indépendante de la
  cible (`type_1`), donc **pas de fuite de données** (data leakage).

### 6. Classes rares regroupées dans « other »
Certains types sont très rares **en tant que type principal** (ex. `flying`, souvent
secondaire). Un modèle ne peut rien apprendre d'une classe à 1–2 exemples, et cela **casse
le split stratifié**. Les types comptant moins de **20** exemples sont donc regroupés dans
une classe `other`. C'est un compromis assumé : `other` devient hétérogène, mais le reste
du pipeline reste stable et évaluable.

### 7. Encodage
La cible `type_1` est encodée séparément avec un `LabelEncoder` au moment de l'entraînement
(Task 3). La seule feature « catégorielle » est `is_dual_type`, déjà binaire (0/1).

### Choix des features du modèle
`total_stats` est **exclu du modèle** car parfaitement colinéaire avec les 6 stats
(c'est leur somme) ; le garder n'apporterait rien et rendrait l'importance des features
moins lisible. Il reste utilisé uniquement pour la visualisation.
