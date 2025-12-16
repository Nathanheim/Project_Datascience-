# Améliorations pour améliorer la précision du modèle

## 📊 Résumé des améliorations recommandées

### 1. Feature Engineering (Impact élevé) ⭐⭐⭐

#### A. Features cycliques pour les données temporelles
**Pourquoi** : Les variables temporelles (heure, jour de semaine) sont cycliques. L'heure 23h est proche de 0h, mais le modèle ne le sait pas sans encodage cyclique.

**Implémentation** : Voir `model_improvements.py` → `add_cyclical_features()`
- `hour_sin` / `hour_cos` (24h cycle)
- `dayofweek_sin` / `dayofweek_cos` (7 jours cycle)
- `month_sin` / `month_cos` (12 mois cycle)

**Impact attendu** : +2-5% R²

#### B. Features de trafic
**Pourquoi** : Les tarifs varient selon le trafic et les habitudes.

**Implémentation** : Voir `model_improvements.py` → `add_traffic_features()`
- `is_rush_hour` (7-9h et 17-19h)
- `is_weekend` (samedi/dimanche)
- `is_night` (0-5h)

**Impact attendu** : +1-3% R²

#### C. Features d'interaction
**Pourquoi** : Les interactions entre variables (ex: distance × passagers) peuvent améliorer la prédiction.

**Implémentation** : Voir `model_improvements.py` → `add_interaction_features()`
- `distance_passengers` (Distance × Nombre de passagers)
- `distance_squared` (Distance² pour capturer la non-linéarité)

**Impact attendu** : +1-2% R²

#### D. Features géographiques avancées
**Pourquoi** : La distance Manhattan (en L) peut être plus pertinente que la distance à vol d'oiseau pour les trajets en ville.

**Implémentation** : Voir `model_improvements.py` → `add_geographical_features()`
- `manhattan_distance` (distance en L)

**Note** : Nécessite de conserver `pickup_longitude` et `dropoff_longitude` dans le preprocessing.

**Impact attendu** : +0.5-2% R²

---

### 2. Amélioration du préprocessing

#### A. RobustScaler au lieu de StandardScaler
**Pourquoi** : Plus robuste aux outliers, ce qui peut améliorer les performances.

**Implémentation** :
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # au lieu de StandardScaler()
```

**Impact attendu** : +0.5-1% R² (modeste mais stable)

#### B. Conservation de certaines coordonnées géographiques
**Pourquoi** : Les coordonnées peuvent être utiles pour identifier des zones à tarifs élevés.

**Suggestion** : Conserver au moins `pickup_latitude` et `dropoff_latitude` pour créer des features de zone.

---

### 3. Amélioration de l'optimisation des hyperparamètres

#### A. Plus d'itérations dans RandomizedSearchCV
**Actuel** : `n_iter=10` (optimisé pour la vitesse)
**Recommandé** : `n_iter=50-100` pour meilleure exploration

**Impact attendu** : +1-3% R²

#### B. Grille de paramètres élargie
**Actuel** :
```python
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [20, 30, 40],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
}
```

**Recommandé** :
```python
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", None],
}
```

**Impact attendu** : +2-4% R²

#### C. Plus de folds dans la validation croisée
**Actuel** : `n_splits=3`
**Recommandé** : `n_splits=5` (standard)

**Impact attendu** : +0.5-1% R² (meilleure estimation)

---

### 4. Optimisation de Gradient Boosting

**Pourquoi** : Gradient Boosting peut être meilleur que Random Forest sur certains datasets.

**Recommandation** : Ajouter un tuning pour GradientBoostingRegressor avec `GridSearchCV` ou `RandomizedSearchCV`.

**Impact attendu** : +2-5% R² si optimal

---

### 5. Validation croisée sur le train set

**Actuel** : Utilisation directe de `train_test_split` sans validation croisée pour l'évaluation initiale.

**Recommandé** : Utiliser `cross_val_score` pour évaluer les modèles sur le train set avant le test final.

**Impact** : Meilleure compréhension de la performance réelle.

---

### 6. Ensemble de modèles (Stacking/Voting)

**Pourquoi** : Combiner plusieurs modèles peut améliorer la précision.

**Implémentation** :
```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Exemple de VotingRegressor
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(...)),
    ('gb', GradientBoostingRegressor(...)),
    ('xgb', XGBRegressor(...))  # Nécessite xgboost
])
```

**Impact attendu** : +1-3% R²

---

## 🎯 Priorisation des améliorations

### Priorité HAUTE (Impact élevé, facile à implémenter)
1. ✅ Features cycliques (`add_cyclical_features`)
2. ✅ Features de trafic (`add_traffic_features`)
3. ✅ Features d'interaction (`add_interaction_features`)
4. ✅ Plus d'itérations dans RandomizedSearchCV (`n_iter=50`)

### Priorité MOYENNE (Impact moyen, effort modéré)
5. RobustScaler
6. Grille de paramètres élargie
7. Optimisation de Gradient Boosting

### Priorité BASSE (Impact faible ou effort élevé)
8. Features géographiques avancées (nécessite changement du preprocessing)
9. Ensemble de modèles (complexité accrue)
10. Validation croisée supplémentaire

---

## 📈 Impact global estimé

Si toutes les améliorations HAUTE priorité sont implémentées :
- **Amélioration R² estimée : +5-10%**
- **Exemple** : Si R² actuel = 0.80, nouveau R² ≈ 0.84-0.88

---

## 🚀 Comment tester les améliorations

1. Créer une branche Git : `git checkout -b improvements`
2. Implémenter les features du fichier `model_improvements.py`
3. Modifier `data_preprocessing.py` pour utiliser les nouvelles features
4. Tester avec `main.py` et comparer les métriques
5. Si amélioration confirmée, merge dans main

---

## 📝 Notes importantes

- Les améliorations peuvent augmenter le temps d'entraînement
- Tester une amélioration à la fois pour identifier celle qui apporte le plus
- Conserver un baseline pour comparer
- Monitorer les métriques sur le test set (éviter le sur-apprentissage)

