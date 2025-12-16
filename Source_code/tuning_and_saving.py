"""
Module de tuning et sauvegarde du meilleur modèle RandomForest.

Contient :
- RandomizedSearchCV sur RandomForestRegressor
- évaluation finale sur le test set
- sauvegarde du modèle avec joblib
"""

import numpy as np
from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def tune_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestRegressor:
    """Effectue un RandomizedSearchCV sur RandomForest."""
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [20, 30, 40],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
    }
    rf = RandomForestRegressor(random_state=random_state)
    # Optimisation : réduit de 5 à 3 folds et de 20 à 10 itérations pour aller 3x plus vite
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=10,  # Réduit de 20 à 10 (au lieu de 100 entraînements, maintenant 30)
        cv=cv,
        n_jobs=2,  # Réduit à 2 pour éviter les timeouts sur Mac
        random_state=random_state,
        scoring="r2",
        verbose=1,
    )
    rf_search.fit(X_train, y_train)

    print("Best Parameters:", rf_search.best_params_)
    print("Best Cross-Val R2:", rf_search.best_score_)

    best_rf = rf_search.best_estimator_
    return best_rf


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Calcule R2, MAE, RMSE sur le test set."""
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Final Model Performance on Test Set ---")
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)

    return {"r2": r2, "mae": mae, "rmse": rmse}


def save_model(model, path: str = "random_forest_model.pkl"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé avec succès dans {path}")


if __name__ == "__main__":
    print("Ce module gère le tuning et la sauvegarde du modèle.")


