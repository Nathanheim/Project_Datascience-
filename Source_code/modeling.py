"""
Module for models for Uber fare prediction.

This module contains:
- PanelOLS (fixed effects)
- Several sklearn models (trees, forests, boosting, etc.)
- Feature improvement functions (cyclical, traffic, interactions, etc.)
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
    Build and fit a PanelOLS model (fixed effects).
    Uses pickup_day and pickup_hour as index.
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
    Train several sklearn models and return
    a DataFrame with R2, MAE, RMSE as well as trained models.
    
    Returns
    -------
    tuple: (results_df, trained_models)
        - results_df: DataFrame with results (R2, MAE, RMSE)
        - trained_models: Dictionary of trained models {name: model}
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
    trained_models = {}  # Store trained models

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model  # Save trained model
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append([name, r2, mae, rmse])

    results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "MAE", "RMSE"])
    results_df = results_df.sort_values(by="R2_Score", ascending=False).reset_index(drop=True)

    return results_df, trained_models


# ============================================================================
# Feature improvement functions
# ============================================================================


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical features for temporal variables.
    Improves modeling of temporal patterns (e.g., midnight close to 23h).
    """
    df = df.copy()
    
    # Cyclical hour (24h)
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    
    # Cyclical day of week (7 days)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['pickup_dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['pickup_dayofweek'] / 7)
    
    # Cyclical month (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['pickup_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['pickup_month'] / 12)
    
    return df


def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features related to traffic and habits.
    """
    df = df.copy()
    
    # Rush hours
    df['is_rush_hour'] = ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)) | \
                         ((df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19))
    df['is_rush_hour'] = df['is_rush_hour'].astype(int)
    
    # Weekend
    df['is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)
    
    # Night hours (0-5h)
    df['is_night'] = ((df['pickup_hour'] >= 0) & (df['pickup_hour'] <= 5)).astype(int)
    
    return df


def add_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced geographical features.
    Requires that pickup_longitude and dropoff_longitude are still present.
    """
    df = df.copy()
    
    # Manhattan distance (L-shaped distance, approximate)
    # Uses latitudes/longitudes if available
    if 'pickup_longitude' in df.columns and 'dropoff_longitude' in df.columns:
        # Manhattan distance approximation
        lat_diff = np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
        lon_diff = np.abs(df['dropoff_longitude'] - df['pickup_longitude'])
        # Approximate conversion to km (1 degree ≈ 111 km)
        df['manhattan_distance'] = (lat_diff * 111) + (lon_diff * 111 * np.cos(np.radians(df['pickup_latitude'])))
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between important variables.
    """
    df = df.copy()
    
    # Distance * Passengers (fare may be more expensive with more passengers)
    df['distance_passengers'] = df['Distance'] * df['passenger_count']
    
    # Distance squared (possible non-linear relationship)
    df['distance_squared'] = df['Distance'] ** 2
    
    return df


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features.
    """
    df = df.copy()
    
    # Day of month (may affect fares)
    df['day_of_month'] = df['pickup_day']
    
    # Season (if applicable)
    df['season'] = df['pickup_month'].apply(lambda x: (x % 12) // 3 + 1)
    
    return df


def improved_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improved pipeline that adds all new features.
    """
    df = df.copy()
    
    # Cyclical features
    df = add_cyclical_features(df)
    
    # Traffic features
    df = add_traffic_features(df)
    
    # Geographical features (if coordinates available)
    df = add_geographical_features(df)
    
    # Interaction features
    df = add_interaction_features(df)
    
    # Temporal features
    df = add_time_based_features(df)
    
    return df


def use_robust_scaler(X_train, X_test):
    """
    Use RobustScaler instead of StandardScaler.
    More robust to outliers.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    print("This module defines models and feature improvements, it is designed to be called from main.py.")

