"""
Module de modèles pour la prédiction du tarif Uber.

Ce module contient les modèles :
- PanelOLS (effets fixes)
- Plusieurs modèles sklearn (arbres, forêts, boosting, etc.)
- Fonctions d'amélioration de features (cycliques, trafic, interactions, etc.)
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
from sklearn.preprocessing import RobustScaler

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


def train_sklearn_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Entraîne plusieurs modèles sklearn et retourne
    un DataFrame avec R2, MAE, RMSE ainsi que les modèles entraînés.
    
    Returns
    -------
    tuple: (results_df, trained_models)
        - results_df: DataFrame avec les résultats (R2, MAE, RMSE)
        - trained_models: Dictionnaire des modèles entraînés {name: model}
    """
    from config import RANDOM_STATE, N_ESTIMATORS_RF
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Support Vector Regressor": SVR(),
    }

    results = []
    trained_models = {}  # Stocker les modèles entraînés

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model  # Sauvegarder le modèle entraîné
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append([name, r2, mae, rmse])

    results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "MAE", "RMSE"])
    results_df = results_df.sort_values(by="R2_Score", ascending=False).reset_index(drop=True)

    return results_df, trained_models


# ============================================================================
# Fonctions d'amélioration de features
# ============================================================================


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features cycliques pour les variables temporelles.
    Améliore la modélisation des patterns temporels (ex: minuit proche de 23h).
    """
    df = df.copy()
    
    # Heure cyclique (24h)
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    
    # Jour de semaine cyclique (7 jours)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['pickup_dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['pickup_dayofweek'] / 7)
    
    # Mois cyclique (12 mois)
    df['month_sin'] = np.sin(2 * np.pi * df['pickup_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['pickup_month'] / 12)
    
    return df


def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features liées au trafic et aux habitudes.
    """
    df = df.copy()
    
    # Heures de pointe (rush hours)
    df['is_rush_hour'] = ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) | \
                         ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19))
    df['is_rush_hour'] = df['is_rush_hour'].astype(int)
    
    # Week-end
    df['is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)
    
    # Heures de nuit (0-5h)
    df['is_night'] = ((df['pickup_hour'] >= 0) & (df['pickup_hour'] <= 5)).astype(int)
    
    return df


def add_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features géographiques avancées.
    Nécessite que pickup_longitude et dropoff_longitude soient encore présents.
    """
    df = df.copy()
    
    # Distance Manhattan (distance en L, approximative)
    # Utilise les latitudes/longitudes si disponibles
    if 'pickup_longitude' in df.columns and 'dropoff_longitude' in df.columns:
        # Approximation de la distance Manhattan
        lat_diff = np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
        lon_diff = np.abs(df['dropoff_longitude'] - df['pickup_longitude'])
        # Conversion approximative en km (1 degré ≈ 111 km)
        df['manhattan_distance'] = (lat_diff * 111) + (lon_diff * 111 * np.cos(np.radians(df['pickup_latitude'])))
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features d'interaction entre variables importantes.
    """
    df = df.copy()
    
    # Distance * Passagers (tarif peut être plus cher avec plus de passagers)
    df['distance_passengers'] = df['Distance'] * df['passenger_count']
    
    # Distance au carré (relation non-linéaire possible)
    df['distance_squared'] = df['Distance'] ** 2
    
    return df


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features basées sur le temps.
    """
    df = df.copy()
    
    # Jour du mois (peut affecter les tarifs)
    df['day_of_month'] = df['pickup_day']
    
    # Saison (si applicable)
    df['season'] = df['pickup_month'].apply(lambda x: (x % 12) // 3 + 1)
    
    return df


def improved_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline amélioré qui ajoute toutes les nouvelles features.
    """
    df = df.copy()
    
    # Features cycliques
    df = add_cyclical_features(df)
    
    # Features de trafic
    df = add_traffic_features(df)
    
    # Features géographiques (si coordonnées disponibles)
    df = add_geographical_features(df)
    
    # Features d'interaction
    df = add_interaction_features(df)
    
    # Features temporelles
    df = add_time_based_features(df)
    
    return df


def use_robust_scaler(X_train, X_test):
    """
    Utilise RobustScaler au lieu de StandardScaler.
    Plus robuste aux outliers.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    print("Ce module définit les modèles et les améliorations de features, il est conçu pour être appelé depuis main.py.")


