"""
Module de modèles pour la prédiction du tarif Uber.

Ce module contient les modèles :
- PanelOLS (effets fixes)
- Plusieurs modèles sklearn (arbres, forêts, boosting, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from linearmodels.panel import PanelOLS
import statsmodels.api as sm


def build_panel_ols(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construit et ajuste un modèle PanelOLS (effets fixes).
    Utilise pickup_day et pickup_hour comme index.
    """
    df_panel = df.set_index(["pickup_day", "pickup_hour"])

    y_panel = df_panel["fare_amount"]
    X_panel = df_panel[["Distance", "passenger_count", "pickup_month"]]
    X_panel = sm.add_constant(X_panel)

    fe_model = PanelOLS(y_panel, X_panel, entity_effects=True)
    fe_results = fe_model.fit()

    return {
        "model": fe_model,
        "results": fe_results,
        "y_panel": y_panel,
        "X_panel": X_panel,
    }


def train_sklearn_models(X_train_scaled, X_test_scaled, y_train, y_test) -> pd.DataFrame:
    """
    Entraîne plusieurs modèles sklearn et retourne
    un DataFrame avec R2, MAE, RMSE.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Support Vector Regressor": SVR(),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append([name, r2, mae, rmse])

    results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "MAE", "RMSE"])
    results_df = results_df.sort_values(by="R2_Score", ascending=False).reset_index(drop=True)

    return results_df


if __name__ == "__main__":
    print("Ce module définit les modèles, il est conçu pour être appelé depuis main.py.")


