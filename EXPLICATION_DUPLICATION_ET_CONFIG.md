# Explication : Duplication et Config

## 1️⃣ Éliminer la duplication dans main.py

### 📌 Le problème en images

**Situation actuelle :**

```
┌─────────────────────────────────┐
│  modeling.py                    │
│  ─────────────────────────────  │
│  train_sklearn_models() {       │
│    models = {                   │  ← Définition 1 des modèles
│      "RF": RandomForest(),      │
│      "GB": GradientBoosting()   │
│    }                            │
│    model.fit()                  │  ← Entraînement 1
│  }                              │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  main.py                        │
│  ─────────────────────────────  │
│  results = train_sklearn_models()│
│                                  │
│  simple_models = {              │  ← Définition 2 (DUPLICATION!)
│    "RF": RandomForest(),        │     Même chose qu'au-dessus
│    "GB": GradientBoosting()     │
│  }                              │
│  best_model.fit()               │  ← Entraînement 2 (DUPLICATION!)
└─────────────────────────────────┘
```

### ❌ Pourquoi c'est mauvais ?

1. **Code dupliqué** : Si vous changez un modèle dans `modeling.py`, vous devez aussi le changer dans `main.py`
2. **Temps perdu** : Le modèle est entraîné deux fois (inutile)
3. **Risque d'erreur** : Les deux définitions peuvent devenir différentes par erreur

### ✅ La solution

**Modifier `modeling.py` pour retourner aussi les modèles entraînés :**

```python
# modeling.py - AVANT (retourne seulement les résultats)
def train_sklearn_models(...):
    models = {...}
    for name, model in models.items():
        model.fit(...)  # Entraîne le modèle
    return results_df  # Retourne seulement les résultats
```

```python
# modeling.py - APRÈS (retourne résultats + modèles entraînés)
def train_sklearn_models(...):
    models = {...}
    trained_models = {}  # Nouveau : stocker les modèles entraînés
    for name, model in models.items():
        model.fit(...)  # Entraîne le modèle
        trained_models[name] = model  # Sauvegarder le modèle entraîné
    return results_df, trained_models  # Retourner les deux
```

**Puis dans `main.py` :**

```python
# main.py - APRÈS (utilise les modèles déjà entraînés)
results_df, trained_models = train_sklearn_models(...)

# Récupérer directement le meilleur modèle (déjà entraîné !)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]  # Pas besoin de réentraîner !

# Utiliser directement
# plot_residuals_best_model(best_model, X_test_scaled, y_test, best_model_name)
```

### 🎯 Résultat

- ✅ Plus de duplication : modèles définis une seule fois
- ✅ Plus rapide : pas de réentraînement inutile
- ✅ Plus sûr : une seule source de vérité

---

## 2️⃣ Créer un config.py

### 📌 Le problème actuel

**Valeurs codées en dur partout dans le code :**

```python
# main.py - ligne 35
csv_path = Path(__file__).parent.parent / "dataset" / "uber.csv"

# data_preprocessing.py - ligne 106
test_size: float = 0.2

# modeling.py - ligne 57
DecisionTreeRegressor(random_state=42)

# tuning_and_saving.py - ligne 29
cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
n_iter=10
```

**Problèmes :**
- Si vous voulez changer `random_state` de 42 à 100, il faut chercher et modifier dans 6 endroits différents
- Risque d'oublier certains endroits
- Difficile de tester différentes configurations

### ✅ La solution : créer un fichier `config.py`

**Créer `Source_code/config.py` :**

```python
# config.py
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "uber.csv"
MODEL_PATH = Path(__file__).parent / "uber_random_forest_model.pkl"

# Paramètres
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ITER_RANDOM_SEARCH = 10
N_SPLITS_CV = 3
```

**Puis utiliser partout :**

```python
# main.py - AVANT
csv_path = Path(__file__).parent.parent / "dataset" / "uber.csv"

# main.py - APRÈS
from config import DATASET_PATH
csv_path = DATASET_PATH
```

```python
# modeling.py - AVANT
DecisionTreeRegressor(random_state=42)

# modeling.py - APRÈS
from config import RANDOM_STATE
DecisionTreeRegressor(random_state=RANDOM_STATE)
```

```python
# tuning_and_saving.py - AVANT
cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
n_iter=10

# tuning_and_saving.py - APRÈS
from config import N_SPLITS_CV, RANDOM_STATE, N_ITER_RANDOM_SEARCH
cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
n_iter=N_ITER_RANDOM_SEARCH
```

### 🎯 Avantages

1. **Un seul endroit à modifier** : Changez `RANDOM_STATE` dans `config.py`, tout le projet utilise la nouvelle valeur
2. **Plus facile à tester** : Créez `config_test.py` avec d'autres valeurs
3. **Plus clair** : On voit tous les paramètres d'un coup d'œil
4. **Plus professionnel** : C'est une bonne pratique en programmation

### 📊 Comparaison

**Sans config.py :**
```
Pour changer random_state de 42 à 100 :
1. Chercher "random_state=42" dans tous les fichiers
2. Modifier 6 endroits différents
3. Vérifier qu'on n'a rien oublié
4. Risque d'erreur : 1 endroit oublié = bug
```

**Avec config.py :**
```
Pour changer random_state de 42 à 100 :
1. Ouvrir config.py
2. Changer RANDOM_STATE = 100
3. C'est tout ! ✅
```

---

## 🎯 Résumé

### 1. Duplication dans main.py
- **Problème** : Les modèles sont définis et entraînés deux fois
- **Solution** : Faire retourner les modèles entraînés par `train_sklearn_models()`
- **Gain** : Code plus propre, plus rapide, moins de bugs

### 2. Config.py
- **Problème** : Valeurs codées en dur partout (42, 0.2, 10, etc.)
- **Solution** : Centraliser dans un fichier `config.py`
- **Gain** : Plus facile à modifier, plus professionnel, moins d'erreurs

---

## 💡 Est-ce urgent ?

Non, ce sont des améliorations de qualité de code, pas des bugs. Votre code fonctionne actuellement.

Mais c'est une bonne pratique qui :
- Facilite la maintenance
- Réduit les risques d'erreurs
- Rend le code plus professionnel

