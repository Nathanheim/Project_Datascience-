"""
Module for loading and preparing data for Uber fare prediction.

This module contains:
- CSV loading
- missing value cleaning
- date/time variable creation
- distance calculation (Haversine formula)
- outlier removal
- feature preparation (X, y) and train/test split + scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV file."""
    df = pd.read_csv(csv_path)
    return df


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing missing values (dropna)."""
    return df.dropna()


def add_datetime_features(df: pd.DataFrame, datetime_col: str = "pickup_datetime") -> pd.DataFrame:
    """Convert datetime column and add year, month, day, hour, day of week."""
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    df["pickup_year"] = df[datetime_col].dt.year
    df["pickup_month"] = df[datetime_col].dt.month
    df["pickup_day"] = df[datetime_col].dt.day
    df["pickup_hour"] = df[datetime_col].dt.hour
    df["pickup_dayofweek"] = df[datetime_col].dt.dayofweek

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unused columns."""
    df = df.copy()
    cols_to_drop = ["Unnamed: 0", "key", "pickup_datetime"]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


def haversine(lon1, lon2, lat1, lat2):
    """Calculate distance in km between two points (Haversine formula)."""
    lon1, lon2, lat1, lat2 = map(np.radians, [lon1, lon2, lat1, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add Distance column."""
    df = df.copy()
    df["Distance"] = haversine(
        df["pickup_longitude"],
        df["dropoff_longitude"],
        df["pickup_latitude"],
        df["dropoff_latitude"],
    ).round(2)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply outlier filtering rules:
    - Distance <= 60
    - fare_amount < 100
    - 0 < passenger_count < 10
    """
    df = df.copy()
    df = df[(df["Distance"] <= 60) & (df["fare_amount"] < 100)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 10)]
    return df


def drop_location_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove some location columns."""
    df = df.copy()
    cols_to_drop = ["pickup_longitude", "dropoff_longitude"]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


def prepare_features_target(df: pd.DataFrame, target_col: str = "fare_amount"):
    """Separate features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Perform train_test_split then apply StandardScaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def full_preprocessing_pipeline(csv_path: str):
    """
    Complete preprocessing pipeline that
    returns objects ready for training.
    Includes advanced features (cyclical, traffic, interactions).
    """
    # Import from same directory (Source_code/)
    import sys
    from pathlib import Path
    # Add Source_code directory to path if necessary
    source_code_dir = Path(__file__).parent
    if str(source_code_dir) not in sys.path:
        sys.path.insert(0, str(source_code_dir))
    
    from modeling import (
        add_cyclical_features,
        add_traffic_features,
        add_geographical_features,
        add_interaction_features,
        add_time_based_features,
    )
    
    # Loading
    df_raw = load_raw_data(csv_path)

    # Missing value cleaning
    df = drop_missing_values(df_raw)

    # Basic temporal features
    df = add_datetime_features(df, datetime_col="pickup_datetime")

    # Remove unused columns
    df = drop_unused_columns(df)

    # Distance (needed before other features)
    df = add_distance_feature(df)

    # Outliers (before adding advanced features to avoid calculating on aberrant data)
    df = remove_outliers(df)

    # Advanced features (before removing coordinates for geographical features)
    df = add_cyclical_features(df)
    df = add_traffic_features(df)
    df = add_geographical_features(df)  # Requires coordinates
    df = add_interaction_features(df)
    df = add_time_based_features(df)

    # Location columns (after using coordinates for geographical features)
    df = drop_location_columns(df)

    # Features / Target
    X, y = prepare_features_target(df, target_col="fare_amount")

    # Split + scaling
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = train_test_scale(
        X, y
    )

    return {
        "df_clean": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


if __name__ == "__main__":
    # Small manual pipeline test
    csv_example = "../dataset/uber.csv"
    data = full_preprocessing_pipeline(csv_example)
    print("Clean data shape:", data["df_clean"].shape)

